// Shared body for uf_init_scan.comp / uf_init_scan_u8.comp; the wrapper declares Thresholded and
// THRESHOLDED_AT(i).
//
// As uf_init_body.glsl, but each pixel points at the start of its run within the workgroup's row
// segment, found with a max-scan over the workgroup; a run that began before the segment points at
// the pixel left of the segment. Each thread covers 4 pixels of one row; dispatched 2D with a
// one-row-tall workgroup.
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x_id = 0, local_size_x = 128) in;

layout(std430, binding = 0) writeonly buffer Parent { uint parent[]; };
// Binding 0 viewed as 4-pixel groups, for rows whose start is a multiple of 4.
layout(std430, binding = 0) writeonly buffer Parent4 { uvec4 parent4[]; };

layout(push_constant) uniform PushConstants {
  uint width;
  uint height;
} pc;

// The last run start of each subgroup; sized for the smallest subgroup in the largest workgroup.
shared uint sg_last[64];

void main() {
  uint x0 = gl_GlobalInvocationID.x * 4u;
  uint y = gl_GlobalInvocationID.y;
  uint row = y * pc.width;
  bool in_row = x0 < pc.width;
  uint x_end = min(x0 + 4u, pc.width);

  // Run starts are held as x + 1, with 0 for none so far.
  // With a width that is a multiple of 4, a thread's 4 pixels are one aligned word and one uvec4.
  bool whole = (pc.width & 3u) == 0u;
  uint start[4] = uint[4](0u, 0u, 0u, 0u);
  uint last = 0u;
  if (in_row) {
    uint prev = (x0 > 0u) ? uint(THRESHOLDED_AT(row + x0 - 1u)) : 0u;
#ifdef THRESHOLDED_WORD
    uint word = whole ? THRESHOLDED_WORD((row + x0) >> 2u) : 0u;
#endif
    for (uint k = 0u; k < 4u; ++k) {
      uint x = x0 + k;
      if (x >= x_end) break;
#ifdef THRESHOLDED_WORD
      uint v = whole ? ((word >> (8u * k)) & 0xFFu) : uint(THRESHOLDED_AT(row + x));
#else
      uint v = uint(THRESHOLDED_AT(row + x));
#endif
      bool join_left = (v != 127u) && (x > 0u) && (prev == v);
      if (!join_left) last = x + 1u;
      start[k] = last;
      prev = v;
    }
  }

  // Every invocation reaches the scan and the barrier.
  uint before = subgroupExclusiveMax(last);
  uint incl = subgroupInclusiveMax(last);
  if (gl_SubgroupInvocationID == gl_SubgroupSize - 1u) sg_last[gl_SubgroupID] = incl;
  memoryBarrierShared();
  barrier();
  for (uint s = 0u; s < gl_SubgroupID; ++s) before = max(before, sg_last[s]);
  if (!in_row) return;

  // Pixel 0 of a row always starts a run, so seg_left is used only for segments after the first.
  uint seg_left = row + gl_WorkGroupID.x * gl_WorkGroupSize.x * 4u - 1u;
  if (whole) {
    uvec4 p;
    for (uint k = 0u; k < 4u; ++k) {
      uint s = max(start[k], before);
      p[k] = (s != 0u) ? row + s - 1u : seg_left;
    }
    parent4[(row + x0) >> 2u] = p;
    return;
  }
  for (uint k = 0u; k < 4u; ++k) {
    uint x = x0 + k;
    if (x >= x_end) break;
    uint s = max(start[k], before);
    parent[row + x] = (s != 0u) ? row + s - 1u : seg_left;
  }
}
