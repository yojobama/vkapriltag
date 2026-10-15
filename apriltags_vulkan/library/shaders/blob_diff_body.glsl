// Shared body for blob_diff.comp / blob_diff_u8.comp; the wrapper declares Thresholded and
// THRESHOLDED_AT(i).
//
// Emits up to 4 boundary points per interior pixel (E, SE, S, SW) with one counter increment.
// Threshold codes are read first; parent[] and blob_size[] only for pixels next to the other colour.
//
// QBPoint.x/.y hold the un-decimated coordinate (base*2 + offset), not the
// decimated pixel index.
// Dispatched 2D: (ox, oy) is the interior coordinate.
layout(local_size_x_id = 0, local_size_x = 16, local_size_y_id = 1, local_size_y = 16) in;

layout(std430, binding = 0) readonly buffer Parent { uint parent[]; };
layout(std430, binding = 1) writeonly buffer Compacted { uint compacted[]; };
layout(std430, binding = 2) buffer Counter { uint counter; };
layout(std430, binding = 3) writeonly buffer Keys { uvec2 keys[]; };
// Read only when honour_changed_flag is set.
layout(std430, binding = 4) readonly buffer Changed { uint changed_flag; };
layout(std430, binding = 5) readonly buffer BlobSize { uint blob_size[]; };

layout(push_constant) uniform PushConstants {
  uint width;
  uint height;
  uint capacity;
  // Non-zero: return early if labelling has not converged (parent[] then holds
  // unfinished union-find pointers).
  uint honour_changed_flag;
  // Must equal uf_final.comp's min_blob_pixels.
  uint min_blob_pixels;
} pc;

uint CodeAt(uint i) { return PackThreshCode(uint(THRESHOLDED_AT(i))); }

// Whether pixel i belongs to a blob large enough to matter; `root` is its blob's root label.
bool Usable(uint i, out uint root) {
  root = parent[i];
  return blob_size[root] >= pc.min_blob_pixels;
}

// Stores one boundary point and its (rep0, rep1) grouping key at `pos`, dropping it when the
// output is full. rep_a/rep_b are raw union-find roots.
void put(uint pos, uint rep_a, uint rep_b, uint px, uint py, int gx, int gy) {
  if (pos >= pc.capacity) return;

  compacted[pos] = PackQBPoint(px, py, gx, gy);

// Read by hash_group.comp; one interleaved uvec2 gives single 8-byte accesses.
  keys[pos] = uvec2(min(rep_a, rep_b), max(rep_a, rep_b));
}

void main() {
  if (pc.honour_changed_flag != 0u && changed_flag != 0u) return;
  uint iw = pc.width - 2u;
  uint ih = pc.height - 2u;
  uint ox = gl_GlobalInvocationID.x;
  uint oy = gl_GlobalInvocationID.y;
  if (ox >= iw || oy >= ih) return;

  uint x = ox + 1u;
  uint y = oy + 1u;

  uint idx = x + y * pc.width;
  uint c0 = CodeAt(idx);
  // Ambiguous pixel: contributes nothing.
  if (c0 == 1u) return;

// Neighbour samples; the interior-only dispatch keeps all five in range.
  uint idxE = idx + 1u;
  uint idxSE = idx + pc.width + 1u;
  uint idxS = idx + pc.width;
  uint idxSW = idx + pc.width - 1u;
  uint idxW = idx - 1u;

  uint cE = CodeAt(idxE);
  uint cSE = CodeAt(idxSE);
  uint cS = CodeAt(idxS);
  uint cSW = CodeAt(idxSW);

  // Connections E, SE, S, SW: a candidate when the pair straddles black/white. With
  // c0 in {0, 2}, c0 + cN == 2 implies cN == 2 - c0.
  bool candE = (c0 + cE == 2u);
  bool candSE = (c0 + cSE == 2u);
  bool candS = (c0 + cS == 2u);
  bool candSW = (c0 + cSW == 2u);
  if (!(candE || candSE || candS || candSW)) return;

  // Blob too small: contributes nothing.
  uint rep0;
  if (!Usable(idx, rep0)) return;

  uint rE = 0u, rSE = 0u, rS = 0u, rSW = 0u;
  bool wantE = candE && Usable(idxE, rE);
  bool wantSE = candSE && Usable(idxSE, rSE);
  bool wantS = candS && Usable(idxS, rS);
  bool wantSW = false;
  if (candSW && Usable(idxSW, rSW)) {
    // Skip the SW connection when W and S are labelled, unambiguous and differ:
    // W's own SE connection already covers that edge.
    bool sw_is_duplicate = false;
    if (x != 1u) {
      uint cW = CodeAt(idxW);
      if (cW != 1u && cS != 1u && cS != cW) {
        uint rW;
        if (Usable(idxW, rW)) {
          // wantS is Usable(S) whenever S is a candidate; otherwise read it here.
          sw_is_duplicate = candS ? wantS : Usable(idxS, rS);
        }
      }
    }
    wantSW = !sw_is_duplicate;
  }

  // Gradient signs from code comparison (codes are monotonic in threshold
  // value); only consumed where the matching want* holds.
  int gSE = (cSE > c0) ? 1 : -1;
  int gSWx = (cSW > c0) ? -1 : 1;
  int gSWy = (cSW > c0) ? 1 : -1;

  // One counter increment per pixel, however many of its four connections are valid.
  uint count = uint(wantE) + uint(wantSE) + uint(wantS) + uint(wantSW);
  if (count == 0u) return;
  uint pos = atomicAdd(counter, count);
  if (wantE) put(pos++, rep0, rE, x * 2u + 1u, y * 2u, (cE > c0) ? 1 : -1, 0);
  if (wantSE) put(pos++, rep0, rSE, x * 2u + 1u, y * 2u + 1u, gSE, gSE);
  if (wantS) put(pos++, rep0, rS, x * 2u, y * 2u + 1u, 0, (cS > c0) ? 1 : -1);
  if (wantSW) put(pos, rep0, rSW, x * 2u - 1u, y * 2u + 1u, gSWx, gSWy);
}
