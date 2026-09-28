// Shared body for blob_diff.comp / blob_diff_subgroup.comp; the variants differ
// only in whether append() aggregates its counter increment across the subgroup.
//
// Emits up to 4 QuadBoundaryPoint candidates per interior pixel (E, SE, S, SW
// connections) and appends the valid ones to the compacted output. Append
// order is nondeterministic and nothing depends on it.
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

layout(push_constant) uniform PushConstants {
  uint width;
  uint height;
  uint capacity;
  // Non-zero: return early if labelling has not converged (parent[] then holds
  // raw union-find pointers rather than packed label_pixels words).
  uint honour_changed_flag;
} pc;

// Appends one boundary point and its (rep0, rep1) grouping key. rep_a/rep_b
// are raw union-find roots. want_append is a value so that every thread makes
// all four calls.
void append(bool want_append, uint rep_a, uint rep_b, uint px, uint py, int gx, int gy) {
  if (!want_append) return;
  uint pos = atomicAdd(counter, 1u);
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
  uint w0 = parent[idx];
  uint c0 = PixelThreshCode(w0);
  uint l0 = PixelLabel(w0);

// Ambiguous pixel, or blob too small: contributes nothing.
  if (c0 == 1u || l0 == 0u) return;

// Neighbour samples; the interior-only dispatch keeps all five in range.
  uint idxE = idx + 1u;
  uint idxSE = idx + pc.width + 1u;
  uint idxS = idx + pc.width;
  uint idxSW = idx + pc.width - 1u;
  uint idxW = idx - 1u;

  uint wE = parent[idxE];
  uint wSE = parent[idxSE];
  uint wS = parent[idxS];
  uint wSW = parent[idxSW];
  uint wW = parent[idxW];

  uint cE = PixelThreshCode(wE);
  uint cSE = PixelThreshCode(wSE);
  uint cS = PixelThreshCode(wS);
  uint cSW = PixelThreshCode(wSW);
  uint cW = PixelThreshCode(wW);

  uint lE = PixelLabel(wE);
  uint lSE = PixelLabel(wSE);
  uint lS = PixelLabel(wS);
  uint lSW = PixelLabel(wSW);
  uint lW = PixelLabel(wW);

  uint rep0 = l0 - 1u;

  // Skip the SW connection when W and S are labelled, unambiguous and differ:
  // W's own SE connection already covers that edge.
  bool sw_is_duplicate = cW != 1u && cS != 1u && cS != cW && x != 1u &&
                         lW != 0u && lS != 0u;

  // Connections E, SE, S, SW: emit when the pair straddles black/white. With
  // c0 in {0, 2}, c0 + cN == 2 implies cN == 2 - c0.
  bool wantE = (c0 + cE == 2u) && lE != 0u;
  bool wantSE = (c0 + cSE == 2u) && lSE != 0u;
  bool wantS = (c0 + cS == 2u) && lS != 0u;
  bool wantSW = !sw_is_duplicate && (c0 + cSW == 2u) && lSW != 0u;

  // Gradient signs from code comparison (codes are monotonic in threshold
  // value); only consumed where the matching want* holds.
  int gSE = (cSE > c0) ? 1 : -1;
  int gSWx = (cSW > c0) ? -1 : 1;
  int gSWy = (cSW > c0) ? 1 : -1;

  append(wantE, rep0, lE - 1u, x * 2u + 1u, y * 2u, (cE > c0) ? 1 : -1, 0);
  append(wantSE, rep0, lSE - 1u, x * 2u + 1u, y * 2u + 1u, gSE, gSE);
  append(wantS, rep0, lS - 1u, x * 2u, y * 2u + 1u, 0, (cS > c0) ? 1 : -1);
  append(wantSW, rep0, lSW - 1u, x * 2u - 1u, y * 2u + 1u, gSWx, gSWy);
}
