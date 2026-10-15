// Shared body for sort_points_local.comp / sort_points_local_u8.comp; the wrapper declares
// DecimatedImage and DECIMATED_AT(i).
//
// One workgroup per selected blob: sorts the blob's points by theta_key within its contiguous
// range using a sorting network over kLocalCap virtual slots (each thread owns a strided subset),
// then writes each point's RawLineFitPoint in sorted order. A blob with more than kLocalCap
// points is written unsorted and counted in OversizedBlobs.
layout(local_size_x_id = 0, local_size_x = 256) in;
// Per-blob slot capacity (one shared word per point), independent of the workgroup size; see
// local_sort_virtual_cap_ in GpuDetector.cpp.
layout(constant_id = 3) const uint kLocalCap = 1024;
// Non-zero (a power of two) selects the rank-merge sort: runs of kChunk points are ranked, then
// merged pairwise by binary search. 0 selects the sorting network.
layout(constant_id = 4) const uint kChunk = 0;

layout(std430, binding = 0) readonly buffer Selected { MinMaxExtentsGpu selected[]; };
layout(std430, binding = 1) readonly buffer BlobPointOffsets { uint blob_point_offsets[]; };
layout(std430, binding = 2) readonly buffer Src { IPoint src[]; };
layout(std430, binding = 4) writeonly buffer Output { RawLineFitPoint output_points[]; };
// Incremented once per blob that exceeds kLocalCap.
layout(std430, binding = 5) buffer OversizedBlobs { uint oversized_blobs; };
// Point count read from this buffer when count_from_buffer is set, otherwise from the push
// constant.
layout(std430, binding = 6) readonly buffer CountBuf { uint count_buf; };

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

// Packs theta_key (20 bits) in the high bits and the local index (12 bits) in the low bits, so
// comparing packed words sorts by key. kLocalCap must not exceed 1 << kLocalIndexBits.
const uint kLocalIndexBits = 12u;
const uint kMaxThetaKey = (1u << (32u - kLocalIndexBits)) - 1u;

shared uint s_packed[kLocalCap];
// Ping-pong partner of s_packed for the rank-merge sort.
shared uint s_merge[kChunk != 0u ? kLocalCap : 1u];

uint SortedAt(bool from_merge, uint i) { return from_merge ? s_merge[i] : s_packed[i]; }

void main() {
  uint blob = gl_WorkGroupID.x;
  if (blob >= (pc.count_from_buffer != 0u ? min(count_buf, pc.max_blobs)
                                          : pc.num_selected_blobs)) return;

  uint tid = gl_LocalInvocationID.x;
  uint threads = gl_WorkGroupSize.x;
  uint count = selected[blob].count;
  uint base = blob_point_offsets[blob] - count;

  if (count > kLocalCap) {
    // One bump per blob.
    if (tid == 0u) atomicAdd(oversized_blobs, 1u);
    for (uint idx = tid; idx < count; idx += threads) {
      output_points[base + idx] = ComputeLineFitPoint(src[base + idx]);
    }
    return;
  }

  if (kChunk != 0u) {
    for (uint idx = tid; idx < count; idx += threads) {
      s_packed[idx] = (min(src[base + idx].theta_key, kMaxThetaKey) << kLocalIndexBits) | idx;
    }
    memoryBarrierShared();
    barrier();

    if (count <= kChunk) {
      for (uint idx = tid; idx < count; idx += threads) {
        uint v = s_packed[idx];
        uint rank = 0u;
        for (uint j = 0u; j < count; ++j) rank += uint(s_packed[j] < v);
        output_points[base + rank] = ComputeLineFitPoint(src[base + idx]);
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

    // Merges pairs of runs, doubling the run length each round; the last round writes the output.
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
          output_points[base + pos] =
              ComputeLineFitPoint(src[base + (v & ((1u << kLocalIndexBits) - 1u))]);
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
    return;
  }

  // Sizes the network to this blob; count is uniform across the workgroup, so cap is too.
  uint cap = 1u;
  uint log2_cap = 0u;
  while (cap < count) {
    cap <<= 1u;
    log2_cap += 1u;
  }

  for (uint idx = tid; idx < cap; idx += threads) {
    uint key = (idx < count) ? min(src[base + idx].theta_key, kMaxThetaKey) : kMaxThetaKey;
    s_packed[idx] = (key << kLocalIndexBits) | idx;
  }
  memoryBarrierShared();
  barrier();

  // Batcher odd-even mergesort over the padded cap slots. The (p, q, r, d) schedule depends only
  // on cap. Comparators are enumerated directly (n_active per round) using shifts and masks,
  // without divide or modulo.
  if (log2_cap >= 1u) {
    uint p = 1u << (log2_cap - 1u);
    uint log2p = log2_cap - 1u;
    while (p >= 1u) {
      uint q = 1u << (log2_cap - 1u);
      uint r = 0u;
      uint d = p;
      while (d >= 1u) {
        uint two_p = p << 1u;
        uint region_below = cap - d;
        uint full_periods = region_below >> (log2p + 1u);
        uint remainder = region_below & (two_p - 1u);
        int partial_signed = int(remainder) - int(r);
        uint partial = uint(clamp(partial_signed, 0, int(p)));
        uint n_active = full_periods * p + partial;
        for (uint c = tid; c < n_active; c += threads) {
          uint idx = ((c >> log2p) << (log2p + 1u)) | r | (c & (p - 1u));
          uint partner = idx + d;
          uint a = s_packed[idx];
          uint b = s_packed[partner];
          if (a > b) {
            s_packed[idx] = b;
            s_packed[partner] = a;
          }
        }
        memoryBarrierShared();
        barrier();
        if (d == q) {
          d = 0u;
        } else {
          d = q - p;
          q >>= 1u;
          r = p;
        }
      }
      p >>= 1u;
      log2p -= 1u;
    }
  }

  for (uint idx = tid; idx < count; idx += threads) {
    uint local_idx = s_packed[idx] & ((1u << kLocalIndexBits) - 1u);
    output_points[base + idx] = ComputeLineFitPoint(src[base + local_idx]);
  }
}
