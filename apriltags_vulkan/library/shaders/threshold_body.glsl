// Shared body for threshold.comp / threshold_u8.comp; the wrapper declares DecimatedImage,
// Thresholded, DECIMATED_AT(i) and STORE_THRESHOLDED(i, v).
//
// Final adaptive threshold: writes 127 (ambiguous), 0 (black) or 255 (white) per decimated pixel
// (InternalThreshold). Each thread covers kRun consecutive pixels of one row. Dispatched 2D.
layout(local_size_x_id = 0, local_size_x = 16, local_size_y_id = 1, local_size_y = 16) in;
layout(constant_id = 3) const uint kRun = 4;

layout(std430, binding = 1) readonly buffer FilteredMinMax { uint minmax_filtered[]; };

layout(push_constant) uniform PushConstants {
  uint decimated_width;
  uint decimated_height;
  uint min_white_black_diff;
  uint block_width;
} pc;

void main() {
  uint x0 = gl_GlobalInvocationID.x * kRun;
  uint y = gl_GlobalInvocationID.y;
  if (x0 >= pc.decimated_width || y >= pc.decimated_height) return;
  uint x_end = min(x0 + kRun, pc.decimated_width);
  uint row = y * pc.decimated_width;
  // block_width is the host's ceil-rounded block grid (decimated_width need not be a multiple of 4).
  uint block_row = (y / 4u) * pc.block_width;

  uint block = 0xffffffffu;
  uint mn = 0u, mx = 0u;
  for (uint x = x0; x < x_end; ++x) {
    if (x / 4u != block) {
      block = x / 4u;
      uint packed = minmax_filtered[block + block_row];
      mn = packed & 0xffu;
      mx = (packed >> 8u) & 0xffu;
    }
    uint result;
    if (mx - mn < pc.min_white_black_diff) {
      result = 127u;
    } else {
      uint thresh = mn + (mx - mn) / 2u;
      result = (uint(DECIMATED_AT(row + x)) > thresh) ? 255u : 0u;
    }
    STORE_THRESHOLDED(row + x, result);
  }
}
