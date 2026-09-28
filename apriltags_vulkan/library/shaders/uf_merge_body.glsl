// Shared body for uf_merge.comp / uf_merge_u8.comp; each wrapper declares
// Thresholded and THRESHOLDED_AT before including this.
//
// One hooking pass of parallel union-find over 4-connected same-valued
// neighbours, considering only the down edge (uf_init.comp already joined the
// horizontal runs). Ambiguous pixels (127) never merge. Alternate with
// uf_compress.comp until changed_flag stays 0.
//
// A thread unions only at a run-overlap start (x == 0, or the pixel to its left
// or below-left differs), since the other columns of an overlap repeat the same
// union. Hooking is by atomicMin, so a root is its component's minimum index.
//
// changed_flag is set only when a union joined two distinct components,
// aggregated in shared memory so each workgroup issues at most one global
// atomic.
//
// Dispatched 2D with one-row-tall workgroups so x is available directly.
layout(local_size_x_id = 0, local_size_x = 256) in;

layout(std430, binding = 0) buffer Parent { uint parent[]; };
layout(std430, binding = 2) buffer Changed { uint changed_flag; };

layout(push_constant) uniform PushConstants {
  uint width;
  uint height;
} pc;

shared uint wg_changed;

uint find(uint n) {
  uint p = parent[n];
  while (p != n) {
    n = p;
    p = parent[n];
  }
  return n;
}

// Returns true if the two elements were in distinct components (i.e. this
// call performed real work and the labelling has not converged yet).
bool doUnion(uint a, uint b) {
  bool merged = false;
  bool done = false;
  while (!done) {
    a = find(a);
    b = find(b);
    if (a < b) {
      uint old = atomicMin(parent[b], a);
      done = (old == b);
      merged = true;
      b = old;
    } else if (b < a) {
      uint old = atomicMin(parent[a], b);
      done = (old == a);
      merged = true;
      a = old;
    } else {
      done = true;
    }
  }
  return merged;
}

void main() {
  uint x = gl_GlobalInvocationID.x;
  uint y = gl_GlobalInvocationID.y;

  // Every invocation must reach both barriers, so bounds tests select work rather than returning early.
  if (gl_LocalInvocationID.x == 0u) wg_changed = 0u;
  memoryBarrierShared();
  barrier();

  bool merged = false;
  if (x < pc.width && y + 1u < pc.height) {
    uint i = y * pc.width + x;
    uint v = THRESHOLDED_AT(i);
    if (v != 127u && THRESHOLDED_AT(i + pc.width) == v) {
      // Leftmost column of this run overlap? See the header comment.
      bool overlap_start = (x == 0u) || (THRESHOLDED_AT(i - 1u) != v) ||
                           (THRESHOLDED_AT(i - 1u + pc.width) != v);
      if (overlap_start && doUnion(i, i + pc.width)) merged = true;
    }
  }

  if (merged) atomicOr(wg_changed, 1u);
  memoryBarrierShared();
  barrier();

  if (gl_LocalInvocationID.x == 0u && wg_changed != 0u) {
    atomicOr(changed_flag, 1u);
  }
}
