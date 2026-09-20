// Shared body for uf_merge.comp / uf_merge_u8.comp. The two differ only in
// Thresholded's storage width (the 8-bit-storage axis), which each wrapper
// declares before including this; Parent stays uint32 either way, since a
// union-find index routinely exceeds 255.
//
// One hooking pass of parallel union-find over 4-connected same-valued
// neighbours. Only the DOWN edge is considered: uf_init.comp already joined
// every horizontal run of equal pixels, so the right-hand edges are unioned
// before this shader ever runs. Between them the two still cover every edge
// exactly once. Pixels with value 127 ("ambiguous", neither black nor white)
// never merge with anything, matching the CUDA implementation's "127 means
// I'm my own blob" rule. Call this shader repeatedly, alternating with
// uf_compress.comp, until `changed_flag` reads back as 0.
//
// RUN-LEVEL MERGING. The down edges between two rows are not independent:
// wherever a horizontal run in row y sits above a run of the same value in
// row y+1, every column of that overlap asks for the SAME union of the SAME
// two components, because uf_init.comp has already joined each row's run
// into one. Only the leftmost column of the overlap needs to perform it; the
// rest are provably redundant, and each one costs two find() walks - the
// dependent global loads that the first pass of OPTIMIZATION_NOTES.md
// identified as this stage's actual cost.
//
// So a thread performs its union only at a run-overlap START: x == 0, or the
// pixel to the left differs, or the pixel below-left differs. The test costs
// two extra loads, both adjacent to ones already being made, against a
// find()-walk pair and an atomicMin saved for every interior column of every
// overlap.
//
// This is the transferable half of HA4 (Hennequin & Lacassagne), the
// run-based 4-connected GPU CCL - reached without its warp intrinsics, which
// three separate measurements in this tree say are the wrong tool on both
// device classes here. See "A scan of the literature" in
// OPTIMIZATION_NOTES.md.
//
// Exactly equivalent, not an approximation. If v[i-1] == v[i] then uf_init
// joined i-1 and i; if v[i-1+W] == v[i+W] it joined those two; so once the
// overlap's leftmost column has unioned its pair, i and i+W are already in
// one component. Induction over the overlap gives the rest, the closure
// reached per pass is unchanged, and hooking is still by atomicMin, so a
// component's root is still its minimum index.
//
// CONVERGENCE FLAG: this used to do `atomicAdd(changed_count, 1u)` for every
// pixel pair that merely *had* a matching neighbour - up to two atomics per
// pixel, i.e. ~1M read-modify-writes per pass all targeting a single 4-byte
// address, which serializes on one cache line and dwarfed the actual union
// work. Worse, the result was never read: the host looped a fixed 64 times
// regardless.
//
// Now the flag is (a) set only when a union actually joined two distinct
// components, so it genuinely means "another pass is needed", and (b)
// aggregated in shared memory first, so at most one global atomic is issued
// per workgroup rather than one per pixel.
//
// DISPATCHED 2D, with a workgroup that is still 1 x N: the run-start test
// needs x, and recovering it from a linear index would cost a division by a
// runtime width on parts with no integer divide instruction (Valhall).
// Keeping the workgroup one row tall means consecutive threads still walk
// consecutive columns, so the two row streams stay exactly as coalesced as
// they were under the 1D dispatch.
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

  // NOTE: every invocation must reach both barriers below, so the bounds and
  // "ambiguous pixel" tests select work rather than returning early.
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
