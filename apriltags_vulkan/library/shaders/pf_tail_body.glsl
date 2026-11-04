// Shared body for pf_tail.comp / pf_tail_atomic64.comp: the stages after blob_diff up to the sort
// (hash_group, reduce_extents_hash_x4, select_blobs, scan_blob_counts, scatter_index_points) as one
// dispatch of a few workgroups that all stay resident, separated by software grid barriers. Each
// stage is the body of the kernel of that name, run over a grid-stride loop. A barrier that
// does not complete (a workgroup is not resident) sets `failed`, every workgroup leaves, and the host
// reruns the tail as separate dispatches.
layout(local_size_x_id = 0, local_size_x = 256) in;

layout(std430, binding = 0) readonly coherent buffer Keys { uvec2 keys[]; };
layout(std430, binding = 1) coherent buffer HashOwner { uint hash_owner[]; };
layout(std430, binding = 2) coherent buffer PointSlot { uint point_slot[]; };
layout(std430, binding = 3) coherent buffer SlotDense { uint slot_dense[]; };
layout(std430, binding = 4) coherent buffer RawBlobCounter { uint raw_blob_counter; };
layout(std430, binding = 5) coherent buffer DropCounter { uint drop_counter; };
layout(std430, binding = 6) readonly coherent buffer CountBuf { uint count_buf; };
layout(std430, binding = 7) coherent buffer Extents { MinMaxExtentsGpu extents[]; };
#ifdef USE_INT64_ATOMIC
layout(std430, binding = 8) coherent buffer ExtentsU64 { uint64_t extents_u64[]; };
#endif
layout(std430, binding = 9) readonly coherent buffer Compacted { uint compacted[]; };
layout(std430, binding = 10) coherent buffer Remap { uint remap[]; };
layout(std430, binding = 11) coherent buffer Selected { MinMaxExtentsGpu selected[]; };
layout(std430, binding = 12) coherent buffer SelectedCounter { uint selected_counter; };
layout(std430, binding = 13) coherent buffer BlobPointOffsets { uint blob_point_offsets[]; };
layout(std430, binding = 14) coherent buffer BlobCursor { uint blob_cursor[]; };
layout(std430, binding = 15) coherent buffer IndexPoints { IPoint index_points[]; };
layout(std430, binding = 16) coherent buffer BlobArgs { uint blob_args[]; };
// arrived counts workgroup arrivals over every barrier; failed is non-zero after a bail-out.
layout(std430, binding = 17) coherent buffer GridSync { uint arrived; uint failed; };

layout(push_constant) uniform PushConstants {
  uint count;          // capacity of the boundary-point buffers
  uint table_mask;
  uint max_probes;
  uint max_raw_blobs;
  uint max_blobs;
  uint min_cluster_pixels;
  uint max_cluster_pixels;
  uint reversed_border;
  uint normal_border;
  uint gx_gy_biased;
  float aspect_max;
  float fill_min;
  float fill_max;
  uint ipoint_capacity;
  uint sort_args_slot;
  uint groups;         // workgroups dispatched; all must be resident
  uint spin_limit;
} pc;

shared uint s_stop;
// Sized for the largest workgroup (1024 threads); only the first gl_WorkGroupSize.x are used.
shared uint chunk_sums[1024];

// Returns after every workgroup has arrived for this round; sets s_stop when the wait gave up.
void GridBarrier(uint round) {
  memoryBarrierBuffer();
  barrier();
  if (gl_LocalInvocationID.x == 0u) {
    atomicAdd(arrived, 1u);
    const uint target_count = pc.groups * (round + 1u);
    uint spins = 0u;
    while (atomicAdd(arrived, 0u) < target_count) {
      if (++spins > pc.spin_limit || atomicAdd(failed, 0u) != 0u) {
        atomicOr(failed, 1u);
        break;
      }
    }
    s_stop = atomicAdd(failed, 0u);
  }
  barrier();
  memoryBarrierBuffer();
}

// Min/max start at +/-INT_MAX, sums at 0, in every copy that select_blobs folds.
void InitExtents(uint bi) {
  MinMaxExtentsGpu e;
  e.min_x = 2147483647;
  e.min_y = 2147483647;
  e.max_x = -2147483647;
  e.max_y = -2147483647;
  e.count = 0u;
  e.gx_sum = 0;
  e.gy_sum = 0;
  e.pxgx_plus_pygy_sum = 0;
  extents[bi] = e;
  if (bi < kPrivateExtentsBlobs) {
    for (uint c = 1u; c < kExtentsCopies; ++c) extents[ExtentsSlot(bi, c, pc.max_raw_blobs)] = e;
  }
}

// hash_group.comp for boundary point i.
void StageHash(uint i) {
  uvec2 k = keys[i];

  uint h = k.x * 0x9E3779B1u + k.y * 0x85EBCA6Bu;
  h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
  h &= pc.table_mask;

  for (uint probe = 0u; probe < pc.max_probes; ++probe) {
    uint owner = hash_owner[h];
    if (owner == 0u) {
      uint prev = atomicCompSwap(hash_owner[h], 0u, i + 1u);
      if (prev == 0u) {
        uint dense = atomicAdd(raw_blob_counter, 1u);
        slot_dense[h] = dense + 1u;
        if (dense < pc.max_raw_blobs) InitExtents(dense);
      }
      owner = (prev == 0u) ? (i + 1u) : prev;
    }
    uint o = owner - 1u;
    if (keys[o] == k) {
      point_slot[i] = h;
      return;
    }
    h = (h + 1u) & pc.table_mask;
  }
  atomicAdd(drop_counter, 1u);
  point_slot[i] = 0xFFFFFFFFu;
}

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
  atomicAdd(extents_u64[s * 4u + 2u], (uint64_t(uint(a.sum)) << 32) | uint64_t(a.count));
  atomicAdd(extents_u64[s * 4u + 3u], (uint64_t(uint(a.gy)) << 32) | uint64_t(uint(a.gx)));
#else
  atomicAdd(extents[s].count, a.count);
  if (a.gx != 0) atomicAdd(extents[s].gx_sum, a.gx);
  if (a.gy != 0) atomicAdd(extents[s].gy_sum, a.gy);
  atomicAdd(extents[s].pxgx_plus_pygy_sum, a.sum);
#endif
}

// reduce_extents_hash_x4_body.glsl for the four points from `first`, accumulating into `copy`.
void StageReduce(uint first, uint n, uint copy) {
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

// Border polarity test (MinMaxExtents::dot() < 0) without float64; see select_blobs.comp.
bool HasReversedBorder(MinMaxExtentsGpu e) {
  int sum2 = e.pxgx_plus_pygy_sum * 2 -
             (e.min_x + e.max_x) * e.gx_sum -
             (e.min_y + e.max_y) * e.gy_sum;
  float two_dot = float(sum2) - 0.10236 * float(e.gx_sum) + 0.057162 * float(e.gy_sum);
  return two_dot < 0.0;
}

// select_blobs.comp for raw blob i.
void StageSelect(uint i) {
  remap[i] = 0xFFFFFFFFu;

  MinMaxExtentsGpu e = extents[i];
  if (i < kPrivateExtentsBlobs) {
    for (uint c = 1u; c < kExtentsCopies; ++c) {
      MinMaxExtentsGpu o = extents[ExtentsSlot(i, c, pc.max_raw_blobs)];
      e.min_x = min(e.min_x, o.min_x);
      e.min_y = min(e.min_y, o.min_y);
      e.max_x = max(e.max_x, o.max_x);
      e.max_y = max(e.max_y, o.max_y);
      e.count += o.count;
      e.gx_sum += o.gx_sum;
      e.gy_sum += o.gy_sum;
      e.pxgx_plus_pygy_sum += o.pxgx_plus_pygy_sum;
    }
  }
  if (pc.gx_gy_biased != 0u) {
    e.gx_sum -= int(e.count);
    e.gy_sum -= int(e.count);
  }
  if (e.count < pc.min_cluster_pixels) return;
  if (e.count > pc.max_cluster_pixels) return;

  uint bbox_w = uint(e.max_x - e.min_x);
  uint bbox_h = uint(e.max_y - e.min_y);
  if (bbox_w == 0u || bbox_h == 0u) return;

  if (pc.aspect_max > 0.0) {
    float long_side = float(max(bbox_w, bbox_h));
    float short_side = float(min(bbox_w, bbox_h));
    if (long_side > short_side * pc.aspect_max) return;
  }

  if (pc.fill_min > 0.0 || pc.fill_max > 0.0) {
    float perimeter = 2.0 * float(bbox_w + bbox_h);
    float fill = float(e.count) / perimeter;
    if (pc.fill_min > 0.0 && fill < pc.fill_min) return;
    if (pc.fill_max > 0.0 && fill > pc.fill_max) return;
  }

  bool quad_reversed_border = HasReversedBorder(e);
  if (pc.reversed_border == 0u && quad_reversed_border) return;
  if (pc.normal_border == 0u && !quad_reversed_border) return;

  uint pos = atomicAdd(selected_counter, 1u);
  if (pos >= pc.max_blobs) return;

  selected[pos] = e;
  remap[i] = pos;
}

// scan_blob_counts.comp, run by workgroup 0 alone.
void StageScan() {
  uint tid = gl_LocalInvocationID.x;
  uint n = min(selected_counter, pc.max_blobs);
  uint per = (pc.max_blobs + gl_WorkGroupSize.x - 1u) / gl_WorkGroupSize.x;
  uint begin = min(tid * per, pc.max_blobs);
  uint end = min(begin + per, pc.max_blobs);

  uint local_sum = 0u;
  for (uint i = begin; i < min(end, n); ++i) local_sum += selected[i].count;
  chunk_sums[tid] = local_sum;
  barrier();

  for (uint offset = 1u; offset < gl_WorkGroupSize.x; offset <<= 1u) {
    uint addend = (tid >= offset) ? chunk_sums[tid - offset] : 0u;
    barrier();
    chunk_sums[tid] += addend;
    barrier();
  }

  uint run = chunk_sums[tid] - local_sum;
  for (uint i = begin; i < end; ++i) {
    if (i < n) run += selected[i].count;
    blob_point_offsets[i] = run;
  }

  if (tid == 0u) {
    blob_args[pc.sort_args_slot * 3u + 0u] = n;
    blob_args[pc.sort_args_slot * 3u + 1u] = 1u;
    blob_args[pc.sort_args_slot * 3u + 2u] = 1u;
  }
}

// scatter_index_points.comp for boundary point i.
void StageScatter(uint i) {
  uint slot = point_slot[i];
  if (slot == 0xFFFFFFFFu) return;

  uint bi = slot_dense[slot];
  if (bi == 0u || bi > pc.max_raw_blobs) return;
  bi -= 1u;

  uint sel = remap[bi];
  if (sel == 0xFFFFFFFFu) return;

  QBPoint p = UnpackQBPoint(compacted[i]);

  MinMaxExtentsGpu e = selected[sel];
  float cx = (float(e.min_x) + float(e.max_x)) * 0.5 + 0.05118;
  float cy = (float(e.min_y) + float(e.max_y)) * 0.5 + -0.028581;

  float dx = float(p.x) - cx;
  float dy = float(p.y) - cy;
  float d = abs(dx) + abs(dy);
  float ratio = (d > 0.0) ? (dy / d) : 0.0;
  float pseudo_angle = (dx >= 0.0) ? ((dy >= 0.0) ? ratio : (4.0 + ratio)) : (2.0 - ratio);

  int theta_int = int(round(pseudo_angle * 2.5e5));
  uint theta_key = uint(max(theta_int, 0));

  uint rank = atomicAdd(blob_cursor[sel], 1u);
  if (rank >= e.count) return;
  uint pos = blob_point_offsets[sel] - e.count + rank;
  if (pos >= pc.ipoint_capacity) return;

  IPoint out_pt;
  out_pt.blob_index = sel;
  out_pt.xy = PackXY(p.x, p.y);
  out_pt.theta_key = theta_key;
  index_points[pos] = out_pt;
}

void main() {
  const uint threads = gl_WorkGroupSize.x;
  const uint wg = gl_WorkGroupID.x;
  const uint gid = wg * threads + gl_LocalInvocationID.x;
  const uint stride = pc.groups * threads;
  const uint n = min(count_buf, pc.count);

  for (uint i = gid; i < n; i += stride) StageHash(i);
  GridBarrier(0u);
  if (s_stop != 0u) return;

  for (uint first = gid * 4u; first < n; first += stride * 4u) {
    StageReduce(first, n, wg & (kExtentsCopies - 1u));
  }
  GridBarrier(1u);
  if (s_stop != 0u) return;

  const uint raw_blobs = min(raw_blob_counter, pc.max_raw_blobs);
  for (uint i = gid; i < raw_blobs; i += stride) StageSelect(i);
  GridBarrier(2u);
  if (s_stop != 0u) return;

  if (wg == 0u) StageScan();
  GridBarrier(3u);
  if (s_stop != 0u) return;

  for (uint i = gid; i < n; i += stride) StageScatter(i);
}
