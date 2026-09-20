// reduce_extents.comp, keyed off the hash slot instead of off a run in a
// sorted array. Identical arithmetic; the only difference is where the raw
// blob index comes from - there is no sorted array to offset into,
// scatter_index_points.comp places each point with a per-blob cursor
// instead, and MinMaxExtentsGpu no longer even carries a starting_offset
// field (see common.glsl).
layout(local_size_x_id = 0, local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Compacted { uint compacted[]; };
layout(std430, binding = 1) readonly buffer PointSlot { uint point_slot[]; };
layout(std430, binding = 2) readonly buffer SlotDense { uint slot_dense[]; };
layout(std430, binding = 3) buffer Extents { MinMaxExtentsGpu extents[]; };
// DEVICE-SIDE COUNT. `count` is a boundary-point total that only exists on
// the GPU until the host reads it back, and that readback is what forced a
// mid-frame SubmitAndWait. Taking the bound from a buffer instead lets this
// dispatch be issued indirectly in the same submission that produced the
// count - see GpuDetector's fused_submits_ and build_indirect_args.comp. The
// push constant is kept as the fallback for the unfused path, selected by
// `count_from_buffer`.
layout(std430, binding = 5) readonly buffer CountBuf { uint count_buf; };
#ifdef USE_INT64_ATOMIC
// The SAME buffer as binding 3, viewed as 64-bit words. The two views never
// touch the same bytes - this one only ever reaches the (count,
// pxgx_plus_pygy_sum) pair, the struct view only ever reaches the other six
// fields - so there is no aliasing hazard to reason about.
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

  // Accumulate into this workgroup's own copy; merge_extents.comp folds them
  // back. See common.glsl's ExtentsSlot comment for why.
  uint s = ExtentsSlot(bi, gl_WorkGroupID.x & (kExtentsCopies - 1u), pc.max_raw_blobs);

  QBPoint p = UnpackQBPoint(compacted[i]);

  int x = int(p.x);
  int y = int(p.y);
  // Read before the atomic. atomicMin with a value already >= the stored one
  // is a no-op, so testing first is exactly equivalent - and a blob's extents
  // stop moving after its first handful of points, so almost every one of
  // these four atomics is skipped. Worth 28% of the span on its own; unlike
  // the guards OPTIMIZATION_NOTES.md warns about, this is the scalar path
  // (the only one integrated parts take), where a cheap early-out and
  // contention relief do compose.
  if (x < extents[s].min_x) atomicMin(extents[s].min_x, x);
  if (y < extents[s].min_y) atomicMin(extents[s].min_y, y);
  if (x > extents[s].max_x) atomicMax(extents[s].max_x, x);
  if (y > extents[s].max_y) atomicMax(extents[s].max_y, y);
#ifdef USE_INT64_ATOMIC
  // One atomic instead of two. `count` (+1 every point) and
  // pxgx_plus_pygy_sum (+v every point) are the only two fields EVERY point
  // touches, and common.glsl places them in one 64-bit word - count in the
  // low half, the sum in the high half.
  //
  // Exact, not approximate. Adding `(uint64(uint(v)) << 32) | 1` increments
  // the high half by v modulo 2^32 - identical to what the 32-bit atomicAdd
  // produced - and the low half by one. The low half can never carry into
  // the high half, because count is bounded by the boundary-point capacity
  // and so cannot reach 2^32. Nothing needs unpacking afterwards either:
  // the two struct fields ARE those halves.
  atomicAdd(extents_u64[s * 4u + 2u],
            (uint64_t(uint(x * p.gx + y * p.gy)) << 32) | 1UL);
#else
  atomicAdd(extents[s].count, 1u);
#endif
  // gx and gy are STRUCTURALLY zero for two of the four connection types -
  // blob_diff_body.glsl emits E as (+/-1, 0) and S as (0, +/-1), only the
  // SE/SW diagonals carry both - so roughly half these points would issue an
  // atomic read-modify-write to add nothing. Adding 0 is the identity, so
  // skipping it is provably bit-identical, and unlike a saturating guard this
  // tests a value already in a register rather than loading one.
  if (p.gx != 0) atomicAdd(extents[s].gx_sum, p.gx);
  if (p.gy != 0) atomicAdd(extents[s].gy_sum, p.gy);
#ifndef USE_INT64_ATOMIC
  atomicAdd(extents[s].pxgx_plus_pygy_sum, x * p.gx + y * p.gy);
#endif
}
