// Shared body for uf_final.comp / uf_final_u8.comp. The two differ only in
// Thresholded's element type (uint vs uint8_t) - the wrapper #including this
// declares Thresholded's binding and a THRESHOLDED_AT(i) accessor macro
// before this point, exactly as label_pixels_body.glsl is parametrized.
//
// Saturating per-blob pixel counter, indexed by root label. Must run after
// the union find has fully converged (last uf_compress pass) and after
// blob_size has been zero-initialized.
//
// This was a faithful histogram - one unconditional atomicAdd per pixel,
// 518400 of them at 1080p - and the contention was worst exactly where it
// hurt most: every pixel of a large tag border or background region targets
// the same blob_size[] slot. But blob_size[] has exactly ONE reader,
// label_pixels.comp, and it only ever asks `blob_size[r] >= min_blob_pixels`.
// The count itself is never consumed. So once a blob has reached the floor,
// every further increment is provably unobservable, and skipping it after a
// plain non-atomic read turns the contended case into a cache-hot load.
//
// AMBIGUOUS (127, code 1) PIXELS SKIP EVEN THAT LOAD. 127 pixels never
// merge with anything (uf_init.comp/uf_merge_body.glsl), so pixel i is
// always its own unique root here - no other pixel's find() can ever reach
// index i as a root when THRESHOLDED_AT(i) == 127, since a 127 index is
// never an operand to any doUnion call (the merge shader's own guard
// requires the calling thread's own pixel to be non-127 before it unions
// anything, and it only ever unions with a same-valued, hence also
// non-127, neighbour). So blob_size[i] is touched by exactly one atomicAdd
// (this same pixel's own), making it always exactly 0 or 1 - and both
// compare false against min_blob_pixels, which GpuDetector's config
// handling floors at 2. Reading thresholded[i] first and skipping the
// parent[]/blob_size[] pair entirely for code 1 is therefore exact, not
// approximate, and this is a streaming read of a buffer this pipeline
// already keeps resident (the same one label_pixels.comp reads), not a
// second gather.
//
// THE THRESHOLDED PREDICATE (for non-ambiguous pixels) STAYS EXACTLY
// CORRECT - not merely conservative. Let S(r) be the true pixel count of
// root r and M = min_blob_pixels. blob_size[r] only ever increments, at
// most S(r) times, from 0. Then:
//
//   S(r) >= M  =>  final >= M:  either nothing skipped, so final = S(r) >= M;
//     or some thread skipped, which means it read a value >= M, and the
//     counter never decreases, so final >= M.
//   S(r) <  M  =>  final <  M:  the counter never exceeds S(r) < M, so no
//     load can ever return >= M, so no thread skips, so final = S(r).
//
// The STORED VALUE becomes an underestimate for blobs above the floor. That
// is fine because nothing reads it - but it does mean this array is now a
// saturating counter, NOT a histogram: do not add a second reader that wants
// the true size without reverting this.
//
// The plain read races with other invocations' atomicAdd, which Vulkan's
// memory model calls a data race. Deliberately NOT marked coherent/volatile,
// and that is a correctness argument rather than an oversight: a stale load
// can only return some previously-written value, and every written value is
// <= S(r). So a stale read is always an UNDERESTIMATE, which costs an
// unnecessary atomicAdd and never a wrong answer - both directions of the
// proof above go through unchanged. Marking the buffer coherent would force
// the load past L1 on every pixel, defeating the entire point, to buy a
// freshness this shader does not need. (The codebase already relies on a
// plain non-atomic read of a concurrently-atomically-updated location:
// hash_group.comp reads hash_owner[h] that way and uses it as an index.)
layout(local_size_x_id = 0, local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Parent { uint parent[]; };
layout(std430, binding = 1) buffer BlobSize { uint blob_size[]; };

layout(push_constant) uniform PushConstants {
  uint width;
  uint height;
  // Must equal label_pixels.comp's min_blob_pixels. If the two ever differ,
  // this shader saturates at a different threshold than the one actually
  // tested, and the predicate stops being exact.
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
