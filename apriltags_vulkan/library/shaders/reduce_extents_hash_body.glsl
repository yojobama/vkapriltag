// Reduces each boundary point into its raw blob's extents accumulator, keyed by hash slot.
layout(local_size_x_id = 0, local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Compacted { uint compacted[]; };
layout(std430, binding = 1) readonly buffer PointSlot { uint point_slot[]; };
layout(std430, binding = 2) readonly buffer SlotDense { uint slot_dense[]; };
layout(std430, binding = 3) buffer Extents { MinMaxExtentsGpu extents[]; };
// Point count read from this buffer when count_from_buffer is set, else from the push constant.
layout(std430, binding = 5) readonly buffer CountBuf { uint count_buf; };
#ifdef USE_INT64_ATOMIC
// Binding 3 viewed as 64-bit words; it reaches only the (count,
// pxgx_plus_pygy_sum) pair, disjoint from the struct view's other fields.
layout(std430, binding = 4) buffer ExtentsU64 { uint64_t extents_u64[]; };
#endif

layout(push_constant) uniform PushConstants {
  uint count;
  uint max_raw_blobs;
  uint count_from_buffer;
} pc;

void main() {
  uint i = gl_GlobalInvocationID.x;
  if (i >= (pc.count_from_buffer != 0u ? count_buf : pc.count)) return;

  uint slot = point_slot[i];
  if (slot == 0xFFFFFFFFu) return;

  uint bi = slot_dense[slot];
  if (bi == 0u || bi > pc.max_raw_blobs) return;
  bi -= 1u;

  // Accumulate into this workgroup's own copy (see ExtentsSlot in common.glsl).
  uint s = ExtentsSlot(bi, gl_WorkGroupID.x & (kExtentsCopies - 1u), pc.max_raw_blobs);

  QBPoint p = UnpackQBPoint(compacted[i]);

  int x = int(p.x);
  int y = int(p.y);
  // Test before the atomic: a blob's extents stop changing after its first few
  // points, so most of these atomics are skipped.
  if (x < extents[s].min_x) atomicMin(extents[s].min_x, x);
  if (y < extents[s].min_y) atomicMin(extents[s].min_y, y);
  if (x > extents[s].max_x) atomicMax(extents[s].max_x, x);
  if (y > extents[s].max_y) atomicMax(extents[s].max_y, y);
#ifdef USE_INT64_ATOMIC
  // One 64-bit add updates count (low half, +1) and pxgx_plus_pygy_sum (high
  // half, modulo 2^32). The low half cannot carry: count < 2^32.
  atomicAdd(extents_u64[s * 4u + 2u],
            (uint64_t(uint(x * p.gx + y * p.gy)) << 32) | 1UL);
#else
  atomicAdd(extents[s].count, 1u);
#endif
  // gx or gy is zero for E and S connections; skip the no-op atomic add.
  if (p.gx != 0) atomicAdd(extents[s].gx_sum, p.gx);
  if (p.gy != 0) atomicAdd(extents[s].gy_sum, p.gy);
#ifndef USE_INT64_ATOMIC
  atomicAdd(extents[s].pxgx_plus_pygy_sum, x * p.gx + y * p.gy);
#endif
}
