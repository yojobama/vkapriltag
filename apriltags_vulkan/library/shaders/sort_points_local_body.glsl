// Shared body for sort_points_local.comp / sort_points_local_u8.comp. The
// two differ only in DecimatedImage's element type (uint vs uint8_t, see
// decimate_u8.comp's comment on why that variant exists) - the wrapper
// #including this file declares DecimatedImage itself and a DECIMATED_AT(i)
// accessor macro before this point, so everything below is written against
// the macro and never mentions the buffer's actual type.
//
// Sorts each selected blob's points into ascending angular order around its
// centroid, one workgroup per blob. rewrite_index_points.comp already packed
// every blob's points into one contiguous range [base, base+count) of the
// input array, so this needs no composite key at all - only theta_key, and
// only within the blob's own range. That replaces what used to be a single
// flat radix/bitonic sort across every selected blob's points combined
// (which needed a (blob_index, theta) composite key purely to keep
// different blobs' points from interleaving).
//
// The bitonic network operates over kLocalCap virtual slots, independent of
// gl_WorkGroupSize.x: each thread owns kLocalCap/gl_WorkGroupSize.x slots in
// a strided pattern (idx = tid, tid+threads, tid+2*threads, ...), so the
// per-blob capacity can be sized from the shared-memory budget alone rather
// than being capped at the workgroup's thread count. This matters because
// real blobs (e.g. a large tag's own border) can comfortably exceed the
// device's max workgroup invocation count while still fitting in shared
// memory.
//
// Blobs whose point count exceeds kLocalCap fall back to processing points
// in their original (unsorted) order: correctness (no lost points, no
// crash) is preserved, but that blob's points won't be angularly resorted.
// Selected blobs have already passed the shape/size quad filters in
// select_blobs.comp, so real tag candidates are expected to stay under this
// cap in practice; it exists as a defensive bound for pathological inputs,
// in the same spirit as the capacity clamps elsewhere in this pipeline.
//
// Fused with the line-fit moment computation (formerly a separate
// compute_line_fit_points.comp dispatch reading this shader's sorted
// output): once a blob's points are in final order, each thread already
// knows exactly which source point lands at its output position, so it can
// sample the decimated image and write the RawLineFitPoint itself instead
// of writing an intermediate sorted IPoint for a second dispatch to re-read.
// That retires a whole 2-3 MB buffer (index_points_sorted_buf_) and its
// read/write traffic.
layout(local_size_x_id = 0, local_size_x = 256) in;
// Virtual per-blob element capacity, decoupled from the workgroup's thread
// count and derived from the device's shared memory budget (1 word/point -
// key and local index packed together, see kLocalIndexBits below) - see
// local_sort_virtual_cap_ in GpuDetector.cpp.
layout(constant_id = 3) const uint kLocalCap = 1024;

layout(std430, binding = 0) readonly buffer Selected { MinMaxExtentsGpu selected[]; };
layout(std430, binding = 1) readonly buffer BlobPointOffsets { uint blob_point_offsets[]; };
layout(std430, binding = 2) readonly buffer Src { IPoint src[]; };
layout(std430, binding = 4) writeonly buffer Output { RawLineFitPoint output_points[]; };

layout(push_constant) uniform PushConstants {
  uint num_selected_blobs;
  int decimated_width;
  int decimated_height;
} pc;

// TransformLineFitPoint's gradient-weight computation (formerly
// compute_line_fit_points.comp's entire body). Mx/My/Mxx/Mxy/Myy are all
// exact functions of (x2, y2, W), so only those three plus blob_index are
// emitted - see RawLineFitPoint's own comment in common.glsl.
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
  out_pt.x2 = ix2;
  out_pt.y2 = iy2;
  out_pt.W = W;
  out_pt.blob_index = p.blob_index;
  return out_pt;
}

// theta_key (scatter_index_points.comp) is scaled to fit 20 bits and
// kLocalCap is capped at 4096 (12 bits), so key and local index share one
// word - key in the high 20 bits, index in the low 12 - instead of two
// separate arrays. Comparing packed values directly still sorts by key
// first (it occupies the high bits), with the index as an incidental,
// harmless tiebreaker; this halves the network's shared-memory footprint
// and its per-compare traffic (one load/store pair instead of two).
//
// 12 rather than 11 bits because a blob's perimeter, in decimated points,
// scales as 1/decimation: the ~1200-point border of a 1080p tag at
// decimation 2 becomes ~2400 at decimation 1, which overflowed the old
// 2048-slot ceiling and silently took the unsorted fallback below - see the
// host's local_sort_virtual_cap_ derivation, which only requests the larger
// ceiling for the decimations that actually need it.
const uint kLocalIndexBits = 12u;
const uint kMaxThetaKey = (1u << (32u - kLocalIndexBits)) - 1u;

shared uint s_packed[kLocalCap];

void main() {
  uint blob = gl_WorkGroupID.x;
  if (blob >= pc.num_selected_blobs) return;

  uint tid = gl_LocalInvocationID.x;
  uint threads = gl_WorkGroupSize.x;
  uint count = selected[blob].count;
  uint base = blob_point_offsets[blob] - count;

  if (count > kLocalCap) {
    for (uint idx = tid; idx < count; idx += threads) {
      output_points[base + idx] = ComputeLineFitPoint(src[base + idx]);
    }
    return;
  }

  // Size the network to THIS blob, not to kLocalCap. `count` is uniform across
  // the workgroup, so `cap` is too, and every barrier below is still reached
  // by every invocation.
  uint cap = 1u;
  while (cap < count) cap <<= 1u;

  for (uint idx = tid; idx < cap; idx += threads) {
    uint key = (idx < count) ? min(src[base + idx].theta_key, kMaxThetaKey) : kMaxThetaKey;
    s_packed[idx] = (key << kLocalIndexBits) | idx;
  }
  memoryBarrierShared();
  barrier();

  for (uint k = 2u; k <= cap; k <<= 1u) {
    for (uint j = k >> 1u; j > 0u; j >>= 1u) {
      for (uint idx = tid; idx < cap; idx += threads) {
        uint partner = idx ^ j;
        if (partner > idx) {
          bool ascending = ((idx & k) == 0u);
          uint a = s_packed[idx];
          uint b = s_packed[partner];
          bool swap = ascending ? (b < a) : (a < b);
          if (swap) {
            s_packed[idx] = b;
            s_packed[partner] = a;
          }
        }
      }
      memoryBarrierShared();
      barrier();
    }
  }

  for (uint idx = tid; idx < count; idx += threads) {
    uint local_idx = s_packed[idx] & ((1u << kLocalIndexBits) - 1u);
    output_points[base + idx] = ComputeLineFitPoint(src[base + local_idx]);
  }
}
