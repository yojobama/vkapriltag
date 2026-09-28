// Shared body for label_pixels.comp / label_pixels_u8.comp; the wrapper declares Thresholded and
// THRESHOLDED_AT(i).
//
// Rewrites each pixel's parent[] entry in place as a packed word holding its blob label (0 when
// the blob is below min_blob_pixels or the pixel is ambiguous) and its threshold code. See
// common.glsl's PixelLabel/PixelThreshCode for the layout.

layout(local_size_x_id = 0, local_size_x = 256) in;

layout(std430, binding = 0) buffer Parent { uint parent[]; };
layout(std430, binding = 1) readonly buffer BlobSize { uint blob_size[]; };
// Read only when honour_changed_flag is set.
layout(std430, binding = 3) readonly buffer Changed { uint changed_flag; };

layout(push_constant) uniform PushConstants {
  uint count;
  // Must equal uf_final.comp's min_blob_pixels.
  uint min_blob_pixels;
  // When set and changed_flag is nonzero, returns without modifying parent[].
  uint honour_changed_flag;
} pc;

void main() {
  uint i = gl_GlobalInvocationID.x;
  if (i >= pc.count) return;
  if (pc.honour_changed_flag != 0u && changed_flag != 0u) return;
  uint code = PackThreshCode(uint(THRESHOLDED_AT(i)));
  // Ambiguous (code 1) pixels always get label 0 without reading blob_size.
  uint label = 0u;
  if (code != 1u) {
    uint r = parent[i];
    // Labels are root + 1 so that 0 means no usable blob.
    label = (blob_size[r] >= pc.min_blob_pixels) ? (r + 1u) : 0u;
  }
  parent[i] = label | (code << kPixelLabelBits);
}
