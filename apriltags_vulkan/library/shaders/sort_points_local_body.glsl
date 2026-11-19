// Shared body for sort_points_local.comp / sort_points_local_u8.comp; the wrapper declares
// DecimatedImage and DECIMATED_AT(i).
//
// Sorts each selected blob's points by theta_key and writes their RawLineFitPoints in that order.
// Mode 0: one workgroup per blob; lists blobs above kLocalCap points for modes 1 and 2 and writes
// blobs above kLargeCap unsorted. Mode 1: sorts one kLocalCap run of a listed blob. Mode 2: places
// each point of a listed blob by ranking it against the blob's other runs.
layout(local_size_x_id = 0, local_size_x = 256) in;
// Shared-memory slots per workgroup (one word per point), independent of the workgroup size; see
// local_sort_split_ in GpuDetector.cpp.
layout(constant_id = 3) const uint kLocalCap = 1024;
// A power of two: runs of kChunk points are ranked, then merged pairwise by binary search.
layout(constant_id = 4) const uint kChunk = 16;
layout(constant_id = 5) const uint kMode = 0;
// Largest blob that is sorted; larger ones are written unsorted (kLargeSortCap, GpuDetector.cpp).
layout(constant_id = 6) const uint kLargeCap = 1024;
// Blobs of at most this many points are sorted by ranking every key against all of them, read 4 at a
// time; 0 disables.
layout(constant_id = 7) const uint kRankAll = 0;

layout(std430, binding = 0) readonly buffer Selected { MinMaxExtentsGpu selected[]; };
layout(std430, binding = 1) readonly buffer BlobPointOffsets { uint blob_point_offsets[]; };
layout(std430, binding = 2) readonly buffer Src { IPoint src[]; };
layout(std430, binding = 4) writeonly buffer Output { RawLineFitPoint output_points[]; };
// Incremented once per blob that exceeds kLargeCap.
layout(std430, binding = 5) buffer OversizedBlobs { uint oversized_blobs; };
// Point count read from this buffer when count_from_buffer is set, otherwise from the push
// constant.
layout(std430, binding = 6) readonly buffer CountBuf { uint count_buf; };
// A blob of more than kLocalCap points and the first workgroup of its share of modes 1 and 2.
struct LargeBlob {
  uint blob;
  uint first_run_group;
  uint first_merge_group;
};
// Blobs of more than kLocalCap points: [0..2] mode 1's dispatch, [3..5] mode 2's, [6] the count,
// then the blobs. Zeroed each frame. Read-only in the modes 1 and 2 build
// (sort_points_large*.comp), which the validation layer's shader-access analysis relies on.
#ifdef SORT_LARGE_ONLY
layout(std430, binding = 7) readonly buffer LargeBlobs {
#else
layout(std430, binding = 7) buffer LargeBlobs {
#endif
  uint large_args[8];
  LargeBlob large_blobs[];
};
// Mode 1's sorted runs, indexed like Output: packed keys whose low bits index into the run.
layout(std430, binding = 8) buffer SortScratch { uint scratch_keys[]; };

layout(push_constant) uniform PushConstants {
  uint num_selected_blobs;
  uint count_from_buffer;
  // Clamps the device-side count, which can exceed max_blobs.
  uint max_blobs;
  int decimated_width;
  int decimated_height;
} pc;

// TransformLineFitPoint's gradient-weight computation; emits only (x2, y2, W, blob_index),
// see RawLineFitPoint in common.glsl.
RawLineFitPoint ComputeLineFitPoint(IPoint p) {
  int ix2 = int(UnpackX(p.xy)) + 1;
  int iy2 = int(UnpackY(p.xy)) + 1;
  int ix = ix2 / 2;
  int iy = iy2 / 2;

  int W = 1;
  if (ix > 0 && ix + 1 < pc.decimated_width && iy > 0 && iy + 1 < pc.decimated_height) {
    int grad_x = int(DECIMATED_AT(iy * pc.decimated_width + ix + 1)) -
                 int(DECIMATED_AT(iy * pc.decimated_width + ix - 1));
    int grad_y = int(DECIMATED_AT((iy + 1) * pc.decimated_width + ix)) -
                 int(DECIMATED_AT((iy - 1) * pc.decimated_width + ix));
    W = int(sqrt(float(grad_x * grad_x + grad_y * grad_y))) + 1;
  }

  RawLineFitPoint out_pt;
  out_pt.xy2 = PackLineFitXY2(ix2, iy2);
  out_pt.w_blob = PackLineFitWBlob(W, p.blob_index);
  return out_pt;
}

// Packs theta_key (20 bits) in the high bits and the point's index within the sorted range
// (12 bits) in the low bits, so comparing packed words sorts by key. kLocalCap is at most 1 << 12.
const uint kLocalIndexBits = 12u;
const uint kLocalIndexMask = (1u << kLocalIndexBits) - 1u;
const uint kMaxThetaKey = (1u << (32u - kLocalIndexBits)) - 1u;

shared uint s_packed[kLocalCap];
// Ping-pong partner of s_packed for the rank-merge sort.
shared uint s_merge[kLocalCap];
// The keys of a blob of at most kRankAll points, 4 to an entry, padded with 0xFFFFFFFF.
shared uvec4 s_rank[kRankAll != 0u ? kRankAll / 4u : 1u];

uint SortedAt(bool from_merge, uint i) { return from_merge ? s_merge[i] : s_packed[i]; }

// The packed key of the point `index` places after `first` in the blob.
uint PackedKey(uint base, uint first, uint index) {
  return (min(src[base + first + index].theta_key, kMaxThetaKey) << kLocalIndexBits) | index;
}

// Places the packed key at position `pos` of the range starting at `first`: mode 0 (whose range is
// the whole blob) writes the point's RawLineFitPoint, mode 1 the key itself.
void Emit(uint base, uint first, uint pos, uint packed) {
  if (kMode == 0u) {
    output_points[base + pos] = ComputeLineFitPoint(src[base + (packed & kLocalIndexMask)]);
  } else {
    scratch_keys[base + first + pos] = packed;
  }
}

// Sorts the blob's points [first, first + count) in shared memory (count <= kLocalCap) and emits
// them at positions first .. first + count - 1. Uniform across the workgroup.
void SortRange(uint base, uint first, uint count) {
  uint tid = gl_LocalInvocationID.x;
  uint threads = gl_WorkGroupSize.x;

  if (kRankAll != 0u && count <= kRankAll) {
    uint padded = (count + 3u) & ~3u;
    for (uint idx = tid; idx < padded; idx += threads) {
      s_rank[idx >> 2u][idx & 3u] = (idx < count) ? PackedKey(base, first, idx) : 0xFFFFFFFFu;
    }
    memoryBarrierShared();
    barrier();
    for (uint idx = tid; idx < count; idx += threads) {
      uint v = s_rank[idx >> 2u][idx & 3u];
      uvec4 ranks = uvec4(0u);
      for (uint j = 0u; j < padded / 4u; ++j) ranks += uvec4(lessThan(s_rank[j], uvec4(v)));
      Emit(base, first, ranks.x + ranks.y + ranks.z + ranks.w, v);
    }
    return;
  }

  for (uint idx = tid; idx < count; idx += threads) {
    s_packed[idx] = PackedKey(base, first, idx);
  }
  memoryBarrierShared();
  barrier();

  if (count <= kChunk) {
    for (uint idx = tid; idx < count; idx += threads) {
      uint v = s_packed[idx];
      uint rank = 0u;
      for (uint j = 0u; j < count; ++j) rank += uint(s_packed[j] < v);
      Emit(base, first, rank, v);
    }
    return;
  }

  // Sorts each run of kChunk points by ranking, into s_merge.
  for (uint idx = tid; idx < count; idx += threads) {
    uint run_start = idx & ~(kChunk - 1u);
    uint run_end = min(run_start + kChunk, count);
    uint v = s_packed[idx];
    uint rank = 0u;
    for (uint j = run_start; j < run_end; ++j) rank += uint(s_packed[j] < v);
    s_merge[run_start + rank] = v;
  }
  memoryBarrierShared();
  barrier();

  // Merges pairs of runs, doubling the run length each round; the last round emits.
  bool in_merge = true;
  for (uint run = kChunk; run < count; run <<= 1u) {
    const bool last = (run << 1u) >= count;
    for (uint idx = tid; idx < count; idx += threads) {
      uint v = SortedAt(in_merge, idx);
      uint pair_start = idx & ~((run << 1u) - 1u);
      bool left = (idx - pair_start) < run;
      uint own_start = left ? pair_start : pair_start + run;
      uint other_start = left ? pair_start + run : pair_start;
      // The right run of the last pair can be short or empty; the left run is always full.
      uint other_len = other_start < count ? min(run, count - other_start) : 0u;
      uint lo = 0u;
      uint hi = other_len;
      while (lo < hi) {
        uint mid = (lo + hi) >> 1u;
        if (SortedAt(in_merge, other_start + mid) < v) {
          lo = mid + 1u;
        } else {
          hi = mid;
        }
      }
      uint pos = pair_start + (idx - own_start) + lo;
      if (last) {
        Emit(base, first, pos, v);
      } else if (in_merge) {
        s_packed[pos] = v;
      } else {
        s_merge[pos] = v;
      }
    }
    if (!last) {
      memoryBarrierShared();
      barrier();
      in_merge = !in_merge;
    }
  }
}

const uint kNone = 0xFFFFFFFFu;
shared uint s_large_slot;

// The listed blob whose share of mode 1 (`runs`) or mode 2 holds workgroup `group`, or kNone.
// Every invocation calls it.
uint FindLargeBlob(uint group, bool runs) {
  uint tid = gl_LocalInvocationID.x;
  uint threads = gl_WorkGroupSize.x;
  if (tid == 0u) s_large_slot = kNone;
  barrier();
  uint n = large_args[6];
  for (uint s = tid; s < n; s += threads) {
    LargeBlob e = large_blobs[s];
    uint count = selected[e.blob].count;
    uint first = runs ? e.first_run_group : e.first_merge_group;
    uint groups = runs ? (count + kLocalCap - 1u) / kLocalCap : (count + threads - 1u) / threads;
    if (group >= first && group < first + groups) s_large_slot = s;
  }
  barrier();
  return s_large_slot;
}

void main() {
  uint tid = gl_LocalInvocationID.x;
  uint threads = gl_WorkGroupSize.x;

  if (kMode == 0u) {
    uint blob = gl_WorkGroupID.x;
    if (blob >= (pc.count_from_buffer != 0u ? min(count_buf, pc.max_blobs)
                                            : pc.num_selected_blobs)) return;
    uint count = selected[blob].count;
    uint base = blob_point_offsets[blob] - count;

    if (count > kLargeCap) {
      // One bump per blob.
      if (tid == 0u) atomicAdd(oversized_blobs, 1u);
      for (uint idx = tid; idx < count; idx += threads) {
        output_points[base + idx] = ComputeLineFitPoint(src[base + idx]);
      }
      return;
    }
    if (count > kLocalCap) {
      // Listed for modes 1 and 2 with its own share of their workgroups; the y and z group counts
      // are 1 whenever x is non-zero.
#ifndef SORT_LARGE_ONLY
      if (tid == 0u) {
        LargeBlob e;
        e.blob = blob;
        e.first_run_group = atomicAdd(large_args[0], (count + kLocalCap - 1u) / kLocalCap);
        e.first_merge_group = atomicAdd(large_args[3], (count + threads - 1u) / threads);
        large_blobs[atomicAdd(large_args[6], 1u)] = e;
        large_args[1] = 1u;
        large_args[2] = 1u;
        large_args[4] = 1u;
        large_args[5] = 1u;
      }
#endif
      return;
    }
    SortRange(base, 0u, count);
    return;
  }

  if (kMode == 1u) {
    uint slot = FindLargeBlob(gl_WorkGroupID.x, true);
    if (slot == kNone) return;
    LargeBlob e = large_blobs[slot];
    uint count = selected[e.blob].count;
    uint base = blob_point_offsets[e.blob] - count;
    uint first = (gl_WorkGroupID.x - e.first_run_group) * kLocalCap;
    SortRange(base, first, min(kLocalCap, count - first));
    return;
  }

  // Mode 2: a point's position is its rank in its own run plus, in every other run, the number of
  // smaller keys; earlier runs hold lower indices, so their equal keys also come first.
  uint slot = FindLargeBlob(gl_WorkGroupID.x, false);
  if (slot == kNone) return;
  LargeBlob e = large_blobs[slot];
  uint count = selected[e.blob].count;
  uint base = blob_point_offsets[e.blob] - count;
  uint idx = (gl_WorkGroupID.x - e.first_merge_group) * threads + tid;
  if (idx >= count) return;

  uint run = idx / kLocalCap;
  uint v = scratch_keys[base + idx];
  uint key = v >> kLocalIndexBits;
  uint pos = idx - run * kLocalCap;
  uint runs = (count + kLocalCap - 1u) / kLocalCap;
  for (uint r = 0u; r < runs; ++r) {
    if (r == run) continue;
    uint start = r * kLocalCap;
    uint lo = 0u;
    uint hi = min(kLocalCap, count - start);
    while (lo < hi) {
      uint mid = (lo + hi) >> 1u;
      uint k = scratch_keys[base + start + mid] >> kLocalIndexBits;
      if (r < run ? k <= key : k < key) {
        lo = mid + 1u;
      } else {
        hi = mid;
      }
    }
    pos += lo;
  }
  output_points[base + pos] =
      ComputeLineFitPoint(src[base + run * kLocalCap + (v & kLocalIndexMask)]);
}
