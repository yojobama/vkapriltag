// reduce_extents.comp keyed by hash slot rather than by position in a sorted array.
layout(local_size_x_id = 0, local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Compacted { uint compacted[]; };
layout(std430, binding = 1) readonly buffer PointSlot { uint point_slot[]; };
layout(std430, binding = 2) readonly buffer SlotDense { uint slot_dense[]; };
layout(std430, binding = 3) buffer Extents { MinMaxExtentsGpu extents[]; };
// Point count read from this buffer when count_from_buffer is set, otherwise from pc.count.
layout(std430, binding = 5) readonly buffer CountBuf { uint count_buf; };
#ifdef USE_INT64_ATOMIC
// Binding 3 viewed as 64-bit words.
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

  // Accumulates into this workgroup's own copy; merge_extents.comp folds the copies together.
  uint s = ExtentsSlot(bi, gl_WorkGroupID.x & (kExtentsCopies - 1u), pc.max_raw_blobs);

  QBPoint p = UnpackQBPoint(compacted[i]);

  int x = int(p.x);
  int y = int(p.y);
  // Skips each atomic when the stored extent is already at least as tight.
  if (x < extents[s].min_x) atomicMin(extents[s].min_x, x);
  if (y < extents[s].min_y) atomicMin(extents[s].min_y, y);
  if (x > extents[s].max_x) atomicMax(extents[s].max_x, x);
  if (y > extents[s].max_y) atomicMax(extents[s].max_y, y);
#ifdef USE_INT64_ATOMIC
  // One 64-bit atomic adds 1 to count (low half) and the sum term to pxgx_plus_pygy_sum
  // (high half).
  atomicAdd(extents_u64[s * 4u + 2u],
            (uint64_t(uint(x * p.gx + y * p.gy)) << 32) | 1UL);
#else
  atomicAdd(extents[s].count, 1u);
#endif
#ifdef USE_INT64_ATOMIC
  // One 64-bit atomic adds gx + 1 to gx_sum (low half) and gy + 1 to gy_sum (high half); the
  // +1 bias per point is removed in select_blobs.comp.
  atomicAdd(extents_u64[s * 4u + 3u],
            (uint64_t(uint(p.gy + 1)) << 32) | uint64_t(uint(p.gx + 1)));
#else
  // Skips the zero gx/gy components of horizontal and vertical connections.
  if (p.gx != 0) atomicAdd(extents[s].gx_sum, p.gx);
  if (p.gy != 0) atomicAdd(extents[s].gy_sum, p.gy);
  atomicAdd(extents[s].pxgx_plus_pygy_sum, x * p.gx + y * p.gy);
#endif
}
