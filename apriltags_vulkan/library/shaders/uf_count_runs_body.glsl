// Shared body for uf_count_runs.comp / uf_count_runs_u8.comp; the wrapper declares Thresholded and
// THRESHOLDED_AT(i), and THRESHOLDED_WORD(w) when the codes can be read 4 to a word.
//
// After uf_init_scan.comp and uf_merge.comp. The first pixel of every piece (a run, cut where
// uf_init_scan's segments end) points itself at its root and adds the piece's capped length (left
// in blob_size by uf_init_scan) to the root's count unless the count has reached min_blob_pixels; a
// root's entry already holds its own piece's share, so blob_size[root] reaches min_blob_pixels
// exactly when the blob has that many pixels. The last pixel of each segment, which the next
// segment's first piece points at, is pointed at its root too. Every pointer the merge can leave in
// parent[] then leads to one of these pixels, so any pixel's root is parent[parent[i]]
// (blob_diff.comp, kTwoHop). Each thread covers 4 pixels of a row; dispatched 2D with a
// one-row-tall workgroup.
layout(local_size_x_id = 0, local_size_x = 256) in;

layout(std430, binding = 0) buffer Parent { uint parent[]; };
layout(std430, binding = 2) buffer BlobSize { uint blob_size[]; };

layout(push_constant) uniform PushConstants {
  uint width;
  uint height;
  // Must equal blob_diff.comp's min_blob_pixels.
  uint min_blob_pixels;
  // uf_init_scan's segment length (a power of two).
  uint segment;
} pc;

void main() {
  uint x0 = gl_GlobalInvocationID.x * 4u;
  uint y = gl_GlobalInvocationID.y;
  if (x0 >= pc.width || y >= pc.height) return;
  uint row = y * pc.width;
  uint x_end = min(x0 + 4u, pc.width);
#ifdef THRESHOLDED_WORD
  bool whole = (pc.width & 3u) == 0u;
  uint word = whole ? THRESHOLDED_WORD((row + x0) >> 2u) : 0u;
#endif
  uint prev = (x0 > 0u) ? uint(THRESHOLDED_AT(row + x0 - 1u)) : 127u;
  for (uint k = 0u; k < 4u; ++k) {
    uint x = x0 + k;
    if (x >= x_end) break;
#ifdef THRESHOLDED_WORD
    uint v = whole ? ((word >> (8u * k)) & 0xFFu) : uint(THRESHOLDED_AT(row + x));
#else
    uint v = uint(THRESHOLDED_AT(row + x));
#endif
    bool joins = x > 0u && prev == v;
    prev = v;
    if (v == 127u) continue;
    uint seg_pos = x & (pc.segment - 1u);
    bool piece_first = !joins || seg_pos == 0u;
    if (!piece_first && seg_pos != pc.segment - 1u) continue;

    uint i = row + x;
    uint n = parent[i];
    if (n == i) continue;
    uint p = parent[n];
    while (p != n) {
      n = p;
      p = parent[n];
    }
    parent[i] = n;
    // A piece's first pixel is not a root here, so no other invocation writes its entry.
    if (piece_first && blob_size[n] < pc.min_blob_pixels) atomicAdd(blob_size[n], blob_size[i]);
  }
}
