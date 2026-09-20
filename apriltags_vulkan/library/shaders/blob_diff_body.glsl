// Shared body for blob_diff.comp / blob_diff_subgroup.comp. The two differ
// only in whether append() aggregates its counter increment across the
// subgroup (the subgroup-ballot axis), so this file is parametrized by that
// instead of duplicating the whole shader. The wrapper #including this
// declares the subgroup extensions (if aggregating) before this point.
//
// There used to be four variants: the storageBuffer8BitAccess axis crossed
// with the ballot axis, because this shader read `thresholded` directly and
// so had to know its element width. It no longer reads it at all -
// label_pixels.comp folds the three-valued threshold into the spare high
// bits of the same parent[] word it already rewrites (see common.glsl's
// PixelLabel/PixelThreshCode) - so the 8-bit axis moved there and the four
// variants here collapsed back to two.
//
// Computes up to 4 QuadBoundaryPoint candidates per interior pixel, one per
// diamond-shaped neighbor connection (E, SE, S, SW), and appends the valid
// ones straight into the compacted output.
//
// This used to write a DENSE 4-plane array of (width-2)*(height-2) QBPoints
// each - 66 MB at 1080p, overwhelmingly empty entries - which a separate
// compact_qbp.comp pass then re-read in its entirety to pick out the valid
// ones. That cost ~132 MB of memory traffic per frame plus a 2M-thread
// dispatch, to move a few hundred thousand real points. Appending directly
// with the same atomic counter compaction already used removes the dense
// array, its traffic, and that whole second pass.
//
// Every read here is now spatially local, and there are six of them rather
// than twelve. The blob identity, the "big enough to matter" test AND the
// pixel's threshold value all come from one parent[] word (repurposed in
// place by label_pixels.comp - see its comment): first that replaced six
// random gathers into blob_size[] per interior pixel plus two more per
// emitted point into root_dense_id[], then it absorbed the parallel
// thresholded[] stencil as well.
//
// Ordering: the append order is nondeterministic (it depends on atomic
// arrival order). Nothing depends on it - the points are immediately grouped
// by (rep0, rep1) with a hash table, and MinMaxExtentsGpu::starting_offset is
// not consumed by the CPU tail.
//
// IMPORTANT: QBPoint.x/.y store the *un-decimated* coordinate (matching
// CUDA's QuadBoundaryPoint::x()/y(): base_x()*2+dx(), base_y()*2+dy()), not
// the raw decimated pixel index - every downstream consumer (extents
// min/max + centroid via MinMaxExtentsGpu::cx()/cy()'s 0.05118/-0.028581
// sub-pixel offsets, line-fit moments, theta) assumes this doubled,
// half-integer-accurate convention.

// Dispatched 2D so the interior (ox, oy) coordinate comes straight from
// gl_GlobalInvocationID.xy instead of a runtime `%`/`/` by the interior
// width (see decimate.comp's comment) - the hottest of the four converted
// shaders, at up to 4 appends per interior pixel.
layout(local_size_x_id = 0, local_size_x = 16, local_size_y_id = 1, local_size_y = 16) in;

layout(std430, binding = 0) readonly buffer Parent { uint parent[]; };
layout(std430, binding = 1) writeonly buffer Compacted { uint compacted[]; };
layout(std430, binding = 2) buffer Counter { uint counter; };
layout(std430, binding = 3) writeonly buffer Keys { uvec2 keys[]; };
// Read only when honour_changed_flag is set - see the guard in main() and
// label_pixels_body.glsl's own copy of this mechanism, which this mirrors.
layout(std430, binding = 4) readonly buffer Changed { uint changed_flag; };

layout(push_constant) uniform PushConstants {
  uint width;
  uint height;
  uint capacity;
  // Set only by the fused fast path's speculative first attempt. This
  // shader is not itself destructive, but when labelling has not actually
  // converged, parent[] here still holds RAW union-find pointers rather
  // than label_pixels.comp's packed (label, code) word - label_pixels is
  // gated the same way and will not have run yet - so PixelLabel/
  // PixelThreshCode below would extract meaningless bits from an arbitrary
  // pixel index. Bounded and clamped everywhere downstream, so this would
  // not crash, only waste a full speculative pass appending garbage
  // points; skipping it here just avoids paying for that on the rare
  // frame that needs a retry.
  uint honour_changed_flag;
} pc;

// Appends one boundary point together with its (rep0, rep1) grouping key.
//
// rep_a/rep_b are raw union-find roots (decimated pixel indices). They are no
// longer translated to dense ids: the grouping is a hash table now, so the key
// only has to be distinct per blob, not narrow. They are also no longer part
// of QBPoint itself (see common.glsl) - nothing downstream ever read them out
// of the compacted point, only out of these key arrays.
//
// want_append is passed as a VALUE rather than used to guard the call site:
// main() calls append() exactly 4 times for every surviving thread,
// unconditionally, handing each direction's test result in as data.
//
// That shape was originally load-bearing for a reason that no longer
// applies - the subgroup-aggregated variant below executed subgroup ops
// inside this function, and subgroup ops reached only through divergent
// control flow are a documented gray area on SPIR-V 1.3-era drivers (the
// divergent-call version measurably dropped points on the desktop test GPU;
// see OPTIMIZATION_NOTES.md). With that variant retired there is no longer
// a correctness argument for it, only a stylistic one: the four directions
// read as four uniform calls. Guarding at the call site instead would be
// equally correct now.
//
// There used to be a subgroup-aggregated variant of the append below - the
// textbook warp-aggregated atomic increment, collapsing one atomicAdd per
// lane into one per subgroup. It is gone: measured on an RX 9060 XT it made
// the `boundary` span 24% SLOWER than the plain atomic (0.0437 vs 0.0330 ms,
// three sessions), and integrated parts never took it at all. Aggregating a
// bare counter is the case a modern atomic unit already handles well, so the
// ballot sequence bought nothing and cost its own issue slots. The variant
// that survives elsewhere in the pipeline, reduce_extents_hash_subgroup,
// aggregates per-point VALUES by key rather than a counter, which is a
// different and still-worthwhile trade. See GpuDetector::CreatePipelines().
void append(bool want_append, uint rep_a, uint rep_b, uint px, uint py, int gx, int gy) {
  if (!want_append) return;
  uint pos = atomicAdd(counter, 1u);
  if (pos >= pc.capacity) return;

  compacted[pos] = PackQBPoint(px, py, gx, gy);

  // Re-read by hash_group.comp when it compares a probed slot's claimant
  // against this point. One interleaved uvec2 rather than two parallel uint
  // arrays, so this is a single 8-byte store and hash_group's probe
  // comparison is a single 8-byte load - see hash_group.comp's Keys comment.
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

  // Ambiguous pixel, or a blob too small to matter: contributes nothing.
  //
  // The c0 == 1u half is in fact already implied by l0 == 0u whenever
  // min_cluster_pixels >= 2 (the default is 24): uf_init only joins
  // left-neighbours with v != 127 and uf_merge only unions down-edges with
  // v != 127, so nothing ever points at an ambiguous pixel and it never
  // points elsewhere - its component is exactly itself, size 1. Kept
  // explicit anyway so this stays an identity-preserving rewrite even at
  // min_cluster_pixels == 1.
  if (c0 == 1u || l0 == 0u) return;

  // Neighbor samples. The dispatch covers interior pixels only
  // (x in [1, width-1), y in [1, height-1)), so all five are in range.
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

  // Dedup check: if the West and South neighbors are both unambiguous and
  // differ from one another, the SW diagonal connection (3) would duplicate
  // the topological edge already captured by the West pixel's own SE (=our
  // South) diagonal connection, so skip emitting it. Folded into wantSW as
  // a value, not an early return before append() calls 1-3 have all run -
  // see append()'s own comment on why every call site is unconditional.
  // (Equality on codes rather than on 0/127/255 values - preserved by any
  // injective mapping, see common.glsl.)
  bool sw_is_duplicate = cW != 1u && cS != 1u && cS != cW && x != 1u &&
                         lW != 0u && lS != 0u;

  // Connections 0 (E), 1 (SE), 2 (S), 3 (SW): emit a point if the two pixels
  // straddle a black/white boundary. All four append() calls are reached by
  // every thread that got this far, unconditionally - only want* varies.
  //
  // `v0 + vN == 255u` becomes `c0 + cN == 2u`: c0 is in {0, 2} here (the
  // c0 == 1 case returned above), so c0 + cN == 2 forces cN = 2 - c0 and the
  // (1, 1) collision that would otherwise alias an ambiguous pair is
  // unreachable. See common.glsl.
  bool wantE = (c0 + cE == 2u) && lE != 0u;
  bool wantSE = (c0 + cSE == 2u) && lSE != 0u;
  bool wantS = (c0 + cS == 2u) && lS != 0u;
  bool wantSW = !sw_is_duplicate && (c0 + cSW == 2u) && lSW != 0u;

  // Gradient signs: (vN > v0) becomes (cN > c0) because the code mapping is
  // monotonic. These three are computed unconditionally and consumed only
  // where the matching want* holds, and monotonicity makes them identical
  // bit patterns even where unused.
  int gSE = (cSE > c0) ? 1 : -1;
  int gSWx = (cSW > c0) ? -1 : 1;
  int gSWy = (cSW > c0) ? 1 : -1;

  append(wantE, rep0, lE - 1u, x * 2u + 1u, y * 2u, (cE > c0) ? 1 : -1, 0);
  append(wantSE, rep0, lSE - 1u, x * 2u + 1u, y * 2u + 1u, gSE, gSE);
  append(wantS, rep0, lS - 1u, x * 2u, y * 2u + 1u, 0, (cS > c0) ? 1 : -1);
  append(wantSW, rep0, lSW - 1u, x * 2u - 1u, y * 2u + 1u, gSWx, gSWy);
}
