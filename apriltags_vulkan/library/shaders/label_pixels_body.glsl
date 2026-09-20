// Shared body for label_pixels.comp / label_pixels_u8.comp. The two differ
// only in Thresholded's element type (uint vs uint8_t, the
// storageBuffer8BitAccess axis) - the wrapper #including this declares
// Thresholded's binding and a THRESHOLDED_AT(i) accessor macro before this
// point, exactly as blob_diff_body.glsl is parametrized.
//
// Folds three things into a single spatially-local per-pixel word, written
// back into `parent[]` itself: which blob this pixel belongs to, whether
// that blob is big enough to matter, and the pixel's three-valued threshold.
// See common.glsl's PixelLabel/PixelThreshCode for the layout and the bit
// budget, and for why the code mapping's monotonicity makes blob_diff's
// rewritten tests bit-identical rather than merely equivalent.
//
// Rewriting parent[] in place is safe because each invocation reads and
// writes only its OWN entry - no thread reads another thread's parent[]
// entry here - so there is no separate buffer to allocate, zero, bind or
// read. Next frame's uf_init overwrites every entry unconditionally, so the
// repurposed encoding cannot leak across frames.
//
// Why this exists: blob_diff.comp used to look up blob_size[parent[n]] for
// the pixel and each of its five neighbours - six RANDOM gathers into a 2 MB
// array per interior pixel - plus two more per emitted point to translate
// roots into dense ids. That became ONE random gather here. This pass then
// also absorbed the threshold read, taking blob_diff from twelve loads per
// interior pixel across two arrays down to six from one; the cost here is
// one extra streaming read of thresholded[i], at the same index this
// invocation already owns.

layout(local_size_x_id = 0, local_size_x = 256) in;

layout(std430, binding = 0) buffer Parent { uint parent[]; };
layout(std430, binding = 1) readonly buffer BlobSize { uint blob_size[]; };
// Read only when honour_changed_flag is set - see the guard below and
// GpuDetector's finish_frame comment on the fused fast path.
layout(std430, binding = 3) readonly buffer Changed { uint changed_flag; };

layout(push_constant) uniform PushConstants {
  uint count;
  // Must equal uf_final.comp's min_blob_pixels - that shader saturates its
  // counter at this same floor, and the two agreeing is what makes the
  // comparison below exact. See uf_final.comp's proof.
  uint min_blob_pixels;
  // Set only by the fused fast path's SPECULATIVE first attempt, where
  // labelling might not actually have converged yet - see GpuDetector's
  // finish_frame. When set and changed_flag is nonzero, this invocation
  // returns without touching parent[] at all.
  //
  // This guard exists because this shader's own write below is
  // DESTRUCTIVE: it overwrites parent[] with a packed (label, threshold
  // code) word, permanently discarding the raw union-find parent pointer
  // that lived there. If that happened before labelling had genuinely
  // converged, a retry's uf_merge/uf_compress passes would resume
  // find()-walking a buffer that no longer holds valid parent pointers -
  // in the worst case a cycle, which hangs the GPU (observed as
  // VK_ERROR_DEVICE_LOST while developing this). Skipping here instead
  // leaves parent[] exactly as the labelling chunk left it: incomplete,
  // but a valid union-find structure a retry can safely keep converging.
  uint honour_changed_flag;
} pc;

void main() {
  uint i = gl_GlobalInvocationID.x;
  if (i >= pc.count) return;
  if (pc.honour_changed_flag != 0u && changed_flag != 0u) return;
  uint r = parent[i];
  // The "+1" biasing is what lets label 0 mean "no usable blob", collapsing
  // blob_diff's two separate rejections (ambiguous pixel, blob too small)
  // into one comparison.
  uint label = (blob_size[r] >= pc.min_blob_pixels) ? (r + 1u) : 0u;
  uint code = PackThreshCode(uint(THRESHOLDED_AT(i)));
  parent[i] = label | (code << kPixelLabelBits);
}
