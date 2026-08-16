// Shared definitions included by all AprilTag Vulkan compute shaders.
// Deliberately uses only core GLSL/SPIR-V features available on any
// Vulkan 1.2 conformant device (no vendor extensions, no shaderInt64 /
// shaderFloat64, which are optional device features).
#ifndef APRILTAG_COMMON_GLSL
#define APRILTAG_COMMON_GLSL

// A "boundary candidate point" - produced once per pixel-pair that straddles
// a black/white threshold boundary.  Mirrors frc971::apriltag::QuadBoundaryPoint
// but stored as plain scalar fields instead of a hand packed 64 bit key,
// since GLSL core has no native 64 bit integer type.
struct QBPoint {
  uint rep0;   // union-find root of the "black" side blob
  uint rep1;   // union-find root of the "white" side blob
  uint x;      // pixel x in full (non-decimated) coordinates
  uint y;      // pixel y in full (non-decimated) coordinates
  int gx;      // gradient x (-1, 0, or 1)
  int gy;      // gradient y (-1, 0, or 1)
  uint valid;  // 0 = empty/invalid entry, 1 = valid
  uint pad0;
};

// Min/max extents + summary statistics of one unique (rep0,rep1) blob pair.
// Mirrors frc971::apriltag::MinMaxExtents.
struct MinMaxExtentsGpu {
  int min_x;
  int min_y;
  int max_x;
  int max_y;
  uint starting_offset;
  uint count;
  int gx_sum;
  int gy_sum;
  int pxgx_plus_pygy_sum;
  uint rep0;
  uint rep1;
  uint pad0;
};

// A point that survived blob selection, with its compact blob index and a
// sort key giving its angle around the blob centroid.  Mirrors
// frc971::apriltag::IndexPoint.
struct IPoint {
  uint blob_index;
  uint x;
  uint y;
  int gx;
  int gy;
  uint theta_key;
  uint pad0;
  uint pad1;
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
