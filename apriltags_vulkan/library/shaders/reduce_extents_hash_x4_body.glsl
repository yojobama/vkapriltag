// As reduce_extents_hash_body.glsl, with each thread taking 4 consecutive boundary points and
// combining runs of points in the same blob before issuing that blob's atomics. blob_diff.comp
// appends a pixel's points together and they usually share a blob pair.
layout(local_size_x_id = 0, local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Compacted { uint compacted[]; };
layout(std430, binding = 1) readonly buffer PointSlot { uint point_slot[]; };
layout(std430, binding = 2) readonly buffer SlotDense { uint slot_dense[]; };
layout(std430, binding = 3) buffer Extents { MinMaxExtentsGpu extents[]; };
layout(std430, binding = 5) readonly buffer CountBuf { uint count_buf; };
#ifdef USE_INT64_ATOMIC
layout(std430, binding = 4) buffer ExtentsU64 { uint64_t extents_u64[]; };
#endif

layout(push_constant) uniform PushConstants {
  uint count;
  uint max_raw_blobs;
  uint count_from_buffer;
} pc;

struct Acc {
  int min_x, min_y, max_x, max_y;
  uint count;
  int sum, gx, gy;
};

void Flush(uint s, Acc a) {
  if (a.min_x < extents[s].min_x) atomicMin(extents[s].min_x, a.min_x);
  if (a.min_y < extents[s].min_y) atomicMin(extents[s].min_y, a.min_y);
  if (a.max_x > extents[s].max_x) atomicMax(extents[s].max_x, a.max_x);
  if (a.max_y > extents[s].max_y) atomicMax(extents[s].max_y, a.max_y);
#ifdef USE_INT64_ATOMIC
  // gx and gy carry the +1 bias per point, as in the one-point shader.
  atomicAdd(extents_u64[s * 4u + 2u], (uint64_t(uint(a.sum)) << 32) | uint64_t(a.count));
  atomicAdd(extents_u64[s * 4u + 3u], (uint64_t(uint(a.gy)) << 32) | uint64_t(uint(a.gx)));
#else
  atomicAdd(extents[s].count, a.count);
  if (a.gx != 0) atomicAdd(extents[s].gx_sum, a.gx);
  if (a.gy != 0) atomicAdd(extents[s].gy_sum, a.gy);
  atomicAdd(extents[s].pxgx_plus_pygy_sum, a.sum);
#endif
}

void main() {
  uint n = pc.count_from_buffer != 0u ? min(count_buf, pc.count) : pc.count;
  uint first = gl_GlobalInvocationID.x * 4u;
  if (first >= n) return;
  uint copy = gl_WorkGroupID.x & (kExtentsCopies - 1u);

  uint cur = 0xFFFFFFFFu;
  Acc a;
  for (uint k = 0u; k < 4u; ++k) {
    uint i = first + k;
    if (i >= n) break;
    uint slot = point_slot[i];
    if (slot == 0xFFFFFFFFu) continue;
    uint bi = slot_dense[slot];
    if (bi == 0u || bi > pc.max_raw_blobs) continue;
    uint s = ExtentsSlot(bi - 1u, copy, pc.max_raw_blobs);
    QBPoint p = UnpackQBPoint(compacted[i]);
    int x = int(p.x);
    int y = int(p.y);
#ifdef USE_INT64_ATOMIC
    int gx = p.gx + 1;
    int gy = p.gy + 1;
#else
    int gx = p.gx;
    int gy = p.gy;
#endif
    if (s != cur) {
      if (cur != 0xFFFFFFFFu) Flush(cur, a);
      cur = s;
      a.min_x = x; a.min_y = y; a.max_x = x; a.max_y = y;
      a.count = 1u; a.sum = x * p.gx + y * p.gy; a.gx = gx; a.gy = gy;
    } else {
      a.min_x = min(a.min_x, x); a.min_y = min(a.min_y, y);
      a.max_x = max(a.max_x, x); a.max_y = max(a.max_y, y);
      a.count += 1u; a.sum += x * p.gx + y * p.gy; a.gx += gx; a.gy += gy;
    }
  }
  if (cur != 0xFFFFFFFFu) Flush(cur, a);
}
