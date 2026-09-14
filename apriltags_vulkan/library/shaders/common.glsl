// Shared definitions included by all AprilTag Vulkan compute shaders.
// Deliberately uses only core GLSL/SPIR-V features available on any
// Vulkan 1.2 conformant device (no vendor extensions, no shaderInt64 /
// shaderFloat64, which are optional device features).
#ifndef APRILTAG_COMMON_GLSL
#define APRILTAG_COMMON_GLSL

// Packs a full-resolution pixel coordinate pair into 14 bits each (max value
// 16383 per axis), shared by QBPoint and IPoint - both carry this same
// "un-decimated boundary coordinate" domain (see blob_diff.comp's own
// comment on that convention). 14 bits per axis bounds the supported image
// width/height at 16383, two orders of magnitude past any realistic sensor;
// GpuDetector's constructor asserts config_.width/height stay under it.
uint PackXY(uint x, uint y) {
  return (x & 0x3FFFu) | ((y & 0x3FFFu) << 14u);
}
uint UnpackX(uint xy) { return xy & 0x3FFFu; }
uint UnpackY(uint xy) { return (xy >> 14u) & 0x3FFFu; }

// --- The per-pixel word label_pixels.comp writes into parent[] ---
//
// label_pixels.comp rewrites parent[] in place after the union find has
// converged (see its comment). It packs two things into that one word:
//
//   bits [0:29]  label: 0 when the pixel has no usable blob (ambiguous, or
//                its blob is below min_cluster_pixels), else 1 + the
//                union-find root's decimated pixel index.
//   bits [30:31] code:  the three-valued threshold at this pixel, as
//                0 = black (0), 1 = ambiguous (127), 2 = white (255).
//
// blob_diff.comp then needs only SIX loads per interior pixel, from one
// array, where it previously took twelve from two - it read both
// thresholded[n] and parent[n] at the pixel and each of its five
// neighbours. Folding the threshold into the word label_pixels already
// writes costs that shader one extra sequential read and saves blob_diff
// six random-ish streams.
//
// LABEL BIT BUDGET. GpuDetector's constructor rejects any configuration
// with 2*(width/decimation) > 16383, so the decimated grid is at most
// 8191x8191 = 67,092,481 pixels, which is under 2^26 (67,108,864). So
// label needs at most 26 bits in the worst configuration the constructor
// accepts - 1080p at decimation 1 is 21 bits, at decimation 2 is 20 - and
// 30 are allocated. The host asserts this too; see CreateBuffers.
//
// THE CODE MAPPING IS MONOTONIC, which is what makes blob_diff's rewritten
// tests identity-preserving rather than merely equivalent:
//   * the boundary test v0 + vN == 255 becomes c0 + cN == 2. With c0 in
//     {0,2} (c0 == 1 returns early), c0 + cN == 2 forces cN = 2 - c0, so
//     the (1,1) collision is unreachable.
//   * the gradient sign (vN > v0) becomes (cN > c0) by monotonicity - and
//     blob_diff computes some of those signs unconditionally, consuming
//     them only where the matching want* holds, so monotonicity means they
//     are identical bit patterns even where unused.
//   * the SW dedup test is pure equality, preserved by any injective map.
const uint kPixelLabelBits = 30u;
const uint kPixelLabelMask = (1u << kPixelLabelBits) - 1u;

// 0/127/255 -> 0/1/2, branch-free and monotonic.
uint PackThreshCode(uint v) { return (v + 1u) >> 7u; }
uint PixelLabel(uint word) { return word & kPixelLabelMask; }
uint PixelThreshCode(uint word) { return word >> kPixelLabelBits; }

// A "boundary candidate point" - produced once per pixel-pair that straddles
// a black/white threshold boundary. Mirrors frc971::apriltag::
// QuadBoundaryPoint, packed into a single uint32 rather than a multi-field
// struct: x, y (packed via PackXY above) and gx, gy (each in {-1, 0, 1}, so
// 2 bits apiece as a "+1" unsigned offset) are the only fields anything ever
// reads back out of a compacted point. The original struct also carried
// rep0/rep1 (the raw union-find roots) and a validity flag; neither had any
// reader once the (rep0, rep1) grouping key moved into its own tight arrays
// (see blob_diff.comp's append()) and the shaders that used to test validity
// only ever ran over exactly the boundary points that were, in fact, valid.
//
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

// Min/max extents + summary statistics of one unique (rep0,rep1) blob pair.
// Mirrors frc971::apriltag::MinMaxExtents, minus two fields nothing reads:
// starting_offset (there is no sorted array left to offset into - see
// reduce_extents_hash.comp) and rep0/rep1 (the raw union-find roots; no
// consumer on the CPU side ever read them back out of MinMaxExtentsGpu).
struct MinMaxExtentsGpu {
  int min_x;
  int min_y;
  int max_x;
  int max_y;
  uint count;
  int gx_sum;
  int gy_sum;
  int pxgx_plus_pygy_sum;
};

// A point that survived blob selection, with its compact blob index and a
// sort key giving its angle around the blob centroid. Mirrors
// frc971::apriltag::IndexPoint, minus gx/gy (written by scatter_index_points.
// comp, never read by anything downstream - compute_line_fit_points.comp
// recomputes its gradient weight straight from the decimated image) and
// padding. x/y share QBPoint's packed encoding.
struct IPoint {
  uint blob_index;
  uint xy;
  uint theta_key;
};

// Per point moments used for line fitting.  Mxx/Mxy/Myy are stored as exact
// 64 bit products (via imulExtended, core SPIR-V) split into hi/lo 32 bit
// halves because GLSL core has no int64 type.  Mirrors
// frc971::apriltag::LineFitPoint.
struct RawLineFitPoint {
  int x2;           // doubled x coordinate (was implicit in Mx = W * x2)
  int y2;           // doubled y coordinate
  int W;            // gradient-magnitude weight
  uint blob_index;
};

#endif // APRILTAG_COMMON_GLSL
