// Shared body for label_pixels.comp / label_pixels_u8.comp; the wrapper
// declares Thresholded and a THRESHOLDED_AT(i) accessor before including this.
//
// Rewrites parent[i] in place as a packed word: the blob label (0 if the blob
// is smaller than min_blob_pixels, else root + 1) and the threshold code
// (layout in common.glsl). Each invocation touches only its own entry.

layout(local_size_x_id = 0, local_size_x = 256) in;

layout(std430, binding = 0) buffer Parent { uint parent[]; };
layout(std430, binding = 1) readonly buffer BlobSize { uint blob_size[]; };
// Read only when honour_changed_flag is set.
layout(std430, binding = 3) readonly buffer Changed { uint changed_flag; };

layout(push_constant) uniform PushConstants {
  uint count;
  // Must equal uf_final.comp's min_blob_pixels.
  uint min_blob_pixels;
  // Non-zero: return without touching parent[] if changed_flag is set
  // (labelling not converged). The write below destroys the union-find
  // pointers.
  uint honour_changed_flag;
} pc;

void main() {
  uint i = gl_GlobalInvocationID.x;
  if (i >= pc.count) return;
  if (pc.honour_changed_flag != 0u && changed_flag != 0u) return;
  uint r = parent[i];
  // Label 0 means no usable blob, hence the +1 bias.
  uint label = (blob_size[r] >= pc.min_blob_pixels) ? (r + 1u) : 0u;
  uint code = PackThreshCode(uint(THRESHOLDED_AT(i)));
  parent[i] = label | (code << kPixelLabelBits);
}
