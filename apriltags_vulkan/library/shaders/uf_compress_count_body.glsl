// Shared body for uf_compress_count.comp / uf_compress_count_u8.comp; the wrapper declares
// Thresholded and THRESHOLDED_AT(i).
//
// uf_compress.comp plus uf_final.comp's saturating per-blob pixel count, using the root already in
// a register. With honour_changed_flag set and changed_flag zero the walk is skipped.
layout(local_size_x_id = 0, local_size_x = 256) in;

layout(std430, binding = 0) buffer Parent { uint parent[]; };
layout(std430, binding = 1) readonly buffer Changed { uint changed_flag; };
layout(std430, binding = 3) buffer BlobSize { uint blob_size[]; };

layout(push_constant) uniform PushConstants {
  uint width;
  uint height;
  uint honour_changed_flag;
  // Must equal blob_diff.comp's min_blob_pixels.
  uint min_blob_pixels;
} pc;

void main() {
  uint i = gl_GlobalInvocationID.x;
  uint total = pc.width * pc.height;
  if (i >= total) return;

  uint p = parent[i];
  uint n = i;
  if (!(pc.honour_changed_flag != 0u && changed_flag == 0u)) {
    const uint original = p;
    while (p != n) {
      n = p;
      p = parent[n];
    }
    // Each thread owns parent[i], so `original` is still its current value.
    if (n != original) parent[i] = n;
  } else {
    n = p;
  }

  // Ambiguous (127) pixels are skipped; counting stops once a blob reaches min_blob_pixels.
  if (PackThreshCode(uint(THRESHOLDED_AT(i))) == 1u) return;
  if (blob_size[n] >= pc.min_blob_pixels) return;
  atomicAdd(blob_size[n], 1u);
}
