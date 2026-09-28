// Shared definitions for the AprilTag Vulkan compute shaders. Core GLSL/SPIR-V
// only: no vendor extensions, no shaderInt64 / shaderFloat64.
#ifndef APRILTAG_COMMON_GLSL
#define APRILTAG_COMMON_GLSL

// Packs a pixel coordinate pair into 14 bits each (max 16383 per axis); shared
// by QBPoint and IPoint.
uint PackXY(uint x, uint y) {
  return (x & 0x3FFFu) | ((y & 0x3FFFu) << 14u);
}
uint UnpackX(uint xy) { return xy & 0x3FFFu; }
uint UnpackY(uint xy) { return (xy >> 14u) & 0x3FFFu; }

// --- The per-pixel word label_pixels.comp writes into parent[] ---
//
//   bits [0:29]  label: 0 when the pixel has no usable blob (ambiguous, or its
//                blob is below min_cluster_pixels), else 1 + the union-find
//                root's decimated pixel index.
//   bits [30:31] code: threshold value, 0 = black (0), 1 = ambiguous (127),
//                2 = white (255).
//
// The 30 label bits cover the largest decimated grid the host accepts (< 2^26
// pixels).
//
// The code mapping is monotonic: (vN > v0) is (cN > c0), v0 + vN == 255 is
// c0 + cN == 2 when c0 != 1, and equality tests are preserved.
const uint kPixelLabelBits = 30u;
const uint kPixelLabelMask = (1u << kPixelLabelBits) - 1u;

// 0/127/255 -> 0/1/2, branch-free and monotonic.
uint PackThreshCode(uint v) { return (v + 1u) >> 7u; }
uint PixelLabel(uint word) { return word & kPixelLabelMask; }
uint PixelThreshCode(uint word) { return word >> kPixelLabelBits; }

// A boundary candidate point (frc971::apriltag::QuadBoundaryPoint) packed into
// one uint32: x, y via PackXY, and gx, gy in {-1, 0, 1} stored as +1 in 2 bits
// each.
// Layout: bits [0:13] x, [14:27] y, [28:29] gx+1, [30:31] gy+1.
struct QBPoint {
  uint x;
  uint y;
  int gx;
  int gy;
};

uint PackQBPoint(uint x, uint y, int gx, int gy) {
  return PackXY(x, y) | (uint(gx + 1) << 28u) | (uint(gy + 1) << 30u);
}

QBPoint UnpackQBPoint(uint packed) {
  QBPoint p;
  p.x = UnpackX(packed);
  p.y = UnpackY(packed);
  p.gx = int((packed >> 28u) & 0x3u) - 1;
  p.gy = int((packed >> 30u) & 0x3u) - 1;
  return p;
}

// Min/max extents and summary statistics of one (rep0, rep1) blob pair
// (frc971::apriltag::MinMaxExtents without starting_offset, rep0, rep1).
//
// The accumulator is replicated kExtentsCopies ways, each workgroup writing its
// own copy; merge_extents.comp folds the copies into copy 0, the canonical
// array read by downstream shaders. Only blob indices below
// kPrivateExtentsBlobs are replicated; higher ones use the shared slot.
//
//   copy 0, any bi      -> bi
//   copy c>0, bi < P    -> max_raw_blobs + (c - 1) * P + bi
//   copy c>0, bi >= P   -> bi
const uint kExtentsCopies = 8u;
const uint kPrivateExtentsBlobs = 4096u;

uint ExtentsSlot(uint bi, uint copy, uint max_raw_blobs) {
  if (copy == 0u || bi >= kPrivateExtentsBlobs) return bi;
  return max_raw_blobs + (copy - 1u) * kPrivateExtentsBlobs + bi;
}

// Field order matters: `count` and `pxgx_plus_pygy_sum` are adjacent at byte
// offset 16, forming one aligned 64-bit word (u64 index slot * 4 + 2) used by
// reduce_extents_hash_atomic64.comp.
struct MinMaxExtentsGpu {
  int min_x;
  int min_y;
  int max_x;
  int max_y;
  uint count;
  int pxgx_plus_pygy_sum;
  int gx_sum;
  int gy_sum;
};

// A selected point with its compact blob index and an angle sort key around the
// blob centroid (frc971::apriltag::IndexPoint without gx/gy and padding). x/y
// use QBPoint's packed encoding.
struct IPoint {
  uint blob_index;
  uint xy;
  uint theta_key;
};

// Per-point line-fit record packed into two words. The host reconstructs the
// moments from x2, y2, W (see RawLineFitPoint in Types.h). Bounds:
//
//   x2, y2      <= 16383 (14 bits each); x2 = 2*ix + 1 on the decimated grid.
//   W           <= 361 (10 bits allocated); W = int(sqrt(gx*gx + gy*gy)) + 1
//               with |gx|, |gy| <= 255.
//   blob_index  < 65536 (22 bits allocated).
struct RawLineFitPoint {
  uint xy2;     // x2 in bits [0:13], y2 in bits [14:27]
  uint w_blob;  // W in bits [0:9], blob_index in bits [10:31]
};

uint PackLineFitXY2(int x2, int y2) {
  return (uint(x2) & 0x3FFFu) | ((uint(y2) & 0x3FFFu) << 14u);
}
uint PackLineFitWBlob(int W, uint blob_index) {
  return (uint(W) & 0x3FFu) | (blob_index << 10u);
}

#endif // APRILTAG_COMMON_GLSL
