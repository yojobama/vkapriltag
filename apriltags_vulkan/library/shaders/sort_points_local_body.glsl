// Shared body for sort_points_local.comp / sort_points_local_u8.comp; the
// wrapper declares DecimatedImage and a DECIMATED_AT(i) accessor before
// including this.
//
// Sorts each selected blob's points by theta_key (angle around the centroid),
// one workgroup per blob, within its contiguous range [base, base + count). The
// sorting network runs over kLocalCap virtual slots strided across the
// workgroup's threads.
//
// A blob with more than kLocalCap points is emitted unsorted and counted in
// OversizedBlobs (DetectProfile::oversized_sort_blobs).
//
// Each output RawLineFitPoint is computed directly from the point that lands at
// that sorted position.
layout(local_size_x_id = 0, local_size_x = 256) in;
// Virtual per-blob capacity, derived from the shared-memory budget (one word
// per point); see local_sort_virtual_cap_ in GpuDetector.cpp.
layout(constant_id = 3) const uint kLocalCap = 1024;

layout(std430, binding = 0) readonly buffer Selected { MinMaxExtentsGpu selected[]; };
layout(std430, binding = 1) readonly buffer BlobPointOffsets { uint blob_point_offsets[]; };
layout(std430, binding = 2) readonly buffer Src { IPoint src[]; };
layout(std430, binding = 4) writeonly buffer Output { RawLineFitPoint output_points[]; };
// Incremented once per blob that exceeds kLocalCap.
layout(std430, binding = 5) buffer OversizedBlobs { uint oversized_blobs; };
// Point count read from this buffer when count_from_buffer is set, else from
// the push constant.
layout(std430, binding = 6) readonly buffer CountBuf { uint count_buf; };

layout(push_constant) uniform PushConstants {
  uint num_selected_blobs;
  uint count_from_buffer;
  // select_blobs.comp's counter can exceed max_blobs, so the device-side bound
  // clamps to it.
  uint max_blobs;
  int decimated_width;
  int decimated_height;
} pc;

// TransformLineFitPoint's gradient-weight computation; emits x2, y2, W and
// blob_index (see RawLineFitPoint in common.glsl).
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

// Key (high 20 bits) and local index (low 12) share one word, so comparing
// packed values sorts by key. kLocalCap is at most 4096.
const uint kLocalIndexBits = 12u;
const uint kMaxThetaKey = (1u << (32u - kLocalIndexBits)) - 1u;

shared uint s_packed[kLocalCap];

void main() {
  uint blob = gl_WorkGroupID.x;
  if (blob >= (pc.count_from_buffer != 0u ? min(count_buf, pc.max_blobs)
                                          : pc.num_selected_blobs)) return;

  uint tid = gl_LocalInvocationID.x;
  uint threads = gl_WorkGroupSize.x;
  uint count = selected[blob].count;
  uint base = blob_point_offsets[blob] - count;

  if (count > kLocalCap) {
    // One bump per blob, not per point.
    if (tid == 0u) atomicAdd(oversized_blobs, 1u);
    for (uint idx = tid; idx < count; idx += threads) {
      output_points[base + idx] = ComputeLineFitPoint(src[base + idx]);
    }
    return;
  }

  // Size the network to this blob. count is workgroup-uniform, so every barrier
  // is reached by all invocations.
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

  // Batcher's odd-even mergesort over the padded `cap` slots: a fixed comparator
  // schedule depending only on cap, in log2(cap)*(log2(cap)+1)/2 barrier rounds.
  // q >= p always, so q - p never wraps.
  if (log2_cap >= 1u) {
    uint p = 1u << (log2_cap - 1u);
    while (p >= 1u) {
      uint q = 1u << (log2_cap - 1u);
      uint r = 0u;
      uint d = p;
      while (d >= 1u) {
        for (uint idx = tid; idx + d < cap; idx += threads) {
          if ((idx & p) == r) {
            uint partner = idx + d;
            uint a = s_packed[idx];
            uint b = s_packed[partner];
            if (a > b) {
              s_packed[idx] = b;
              s_packed[partner] = a;
            }
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
    }
  }

  for (uint idx = tid; idx < count; idx += threads) {
    uint local_idx = s_packed[idx] & ((1u << kLocalIndexBits) - 1u);
    output_points[base + idx] = ComputeLineFitPoint(src[base + local_idx]);
  }
}
