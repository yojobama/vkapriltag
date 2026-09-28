// Shared body for uf_final.comp / uf_final_u8.comp; the wrapper declares Thresholded and
// THRESHOLDED_AT(i).
//
// Saturating per-blob pixel counter indexed by root label. Runs after the union find has converged
// and blob_size has been zeroed. A pixel stops incrementing once its blob has reached
// min_blob_pixels, and ambiguous (127) pixels are skipped, so blob_size[r] is only reliable
// as a test against min_blob_pixels.
layout(local_size_x_id = 0, local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Parent { uint parent[]; };
layout(std430, binding = 1) buffer BlobSize { uint blob_size[]; };

layout(push_constant) uniform PushConstants {
  uint width;
  uint height;
  // Must equal label_pixels.comp's min_blob_pixels.
  uint min_blob_pixels;
} pc;

void main() {
  uint i = gl_GlobalInvocationID.x;
  uint total = pc.width * pc.height;
  if (i >= total) return;
  if (PackThreshCode(uint(THRESHOLDED_AT(i))) == 1u) return;
  uint r = parent[i];
  if (blob_size[r] >= pc.min_blob_pixels) return;
  atomicAdd(blob_size[r], 1u);
}
