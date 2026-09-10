// Shared body for blob_diff.comp / blob_diff_u8.comp / blob_diff_subgroup.
// comp / blob_diff_u8_subgroup.comp. The four differ only in Thresholded's
// element type (uint vs uint8_t, A8's storageBuffer8BitAccess axis) and
// whether append() aggregates its counter increment across the subgroup
// (M2's subgroup-ballot axis) - orthogonal choices, so this file is
// parametrized by both instead of duplicating the whole shader four times.
// The wrapper #including this declares Thresholded's binding, a
// THRESHOLDED_AT(i) accessor macro, and (if aggregating) the subgroup
// extensions, before this point.
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
// Every read here is now spatially local. The blob identity and the
// "big enough to matter" test both come from parent[] (repurposed in place
// by label_pixels.comp - see its comment), which replaced six random gathers
// into blob_size[] per interior pixel plus two more per emitted point into
// root_dense_id[].
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

layout(std430, binding = 1) readonly buffer Parent { uint parent[]; };
layout(std430, binding = 2) writeonly buffer Compacted { uint compacted[]; };
layout(std430, binding = 3) buffer Counter { uint counter; };
layout(std430, binding = 4) writeonly buffer KeysHi { uint keys_hi[]; };
layout(std430, binding = 5) writeonly buffer KeysLo { uint keys_lo[]; };

layout(push_constant) uniform PushConstants {
  uint width;
  uint height;
  uint capacity;
} pc;

// Appends one boundary point together with its (rep0, rep1) grouping key.
//
// rep_a/rep_b are raw union-find roots (decimated pixel indices). They are no
// longer translated to dense ids: the grouping is a hash table now, so the key
// only has to be distinct per blob, not narrow. They are also no longer part
// of QBPoint itself (see common.glsl) - nothing downstream ever read them out
// of the compacted point, only out of these key arrays.
//
// want_append is a plain VALUE, not a condition guarding whether append() is
// called at all - main() calls append() exactly 4 times for every surviving
// thread, unconditionally, passing each direction's test result as data.
// This is deliberate: subgroup operations executed from inside a function
// that is itself only ENTERED under divergent control flow are a documented
// gray area on some SPIR-V 1.3-era compilers/drivers (full "maximal
// reconvergence" guarantees are a later addition) - calling append()
// uniformly and gating the actual write on a boolean parameter instead
// keeps every subgroup op reachable through the exact same, non-divergent
// control-flow path on every invocation, which is unambiguously well-defined
// on every target. (This was empirically load-bearing, not just caution:
// the divergent-call version measurably dropped points on the desktop test
// GPU - see OPTIMIZATION_NOTES.md.)
//
// AGGREGATE_APPEND_COUNTER (subgroup variants only): every lane with
// want_append true wants to append exactly one point, to the SAME counter
// address regardless of key - a plain "warp-aggregated atomic increment"
// (ballot -> one atomicAdd(counter, popcount) from the elected lane ->
// broadcast the base back to every lane -> each lane's own slot is
// base + its exclusive rank in the ballot), the textbook GPU compaction
// idiom. This needs only ballot + broadcast + elect, all guaranteed
// wherever SUBGROUP_FEATURE_BALLOT_BIT is reported in COMPUTE - no
// partition/match-any extension.
void append(bool want_append, uint rep_a, uint rep_b, uint px, uint py, int gx, int gy) {
#ifdef AGGREGATE_APPEND_COUNTER
  uvec4 ballot = subgroupBallot(want_append);
  uint count = subgroupBallotBitCount(ballot);
  uint rank = subgroupBallotExclusiveBitCount(ballot);
  uint base = 0u;
  if (subgroupElect()) {
    base = atomicAdd(counter, count);
  }
  base = subgroupBroadcastFirst(base);
  uint pos = base + rank;
#else
  if (!want_append) return;
  uint pos = atomicAdd(counter, 1u);
#endif
  if (!want_append || pos >= pc.capacity) return;

  compacted[pos] = PackQBPoint(px, py, gx, gy);

  // Re-read by hash_group.comp when it compares a probed slot's claimant
  // against this point.
  keys_hi[pos] = min(rep_a, rep_b);
  keys_lo[pos] = max(rep_a, rep_b);
}

void main() {
  uint iw = pc.width - 2u;
  uint ih = pc.height - 2u;
  uint ox = gl_GlobalInvocationID.x;
  uint oy = gl_GlobalInvocationID.y;
  if (ox >= iw || oy >= ih) return;

  uint x = ox + 1u;
  uint y = oy + 1u;

  uint idx = x + y * pc.width;
  uint v0 = uint(THRESHOLDED_AT(idx));
  uint l0 = parent[idx];

  // Ambiguous pixel, or a blob too small to matter: contributes nothing.
  if (v0 == 127u || l0 == 0u) return;

  // Neighbor samples. The dispatch covers interior pixels only
  // (x in [1, width-1), y in [1, height-1)), so all five are in range.
  uint idxE = idx + 1u;
  uint idxSE = idx + pc.width + 1u;
  uint idxS = idx + pc.width;
  uint idxSW = idx + pc.width - 1u;
  uint idxW = idx - 1u;

  uint vE = uint(THRESHOLDED_AT(idxE));
  uint vSE = uint(THRESHOLDED_AT(idxSE));
  uint vS = uint(THRESHOLDED_AT(idxS));
  uint vSW = uint(THRESHOLDED_AT(idxSW));
  uint vW = uint(THRESHOLDED_AT(idxW));

  uint lE = parent[idxE];
  uint lSE = parent[idxSE];
  uint lS = parent[idxS];
  uint lSW = parent[idxSW];
  uint lW = parent[idxW];

  uint rep0 = l0 - 1u;

  // Dedup check: if the West and South neighbors are both unambiguous and
  // differ from one another, the SW diagonal connection (3) would duplicate
  // the topological edge already captured by the West pixel's own SE (=our
  // South) diagonal connection, so skip emitting it. Folded into wantSW as
  // a value, not an early return before append() calls 1-3 have all run -
  // see append()'s own comment on why every call site is unconditional.
  bool sw_is_duplicate = vW != 127u && vS != 127u && vS != vW && x != 1u &&
                         lW != 0u && lS != 0u;

  // Connections 0 (E), 1 (SE), 2 (S), 3 (SW): emit a point if the two pixels
  // straddle a black/white boundary. All four append() calls are reached by
  // every thread that got this far, unconditionally - only want* varies.
  bool wantE = (v0 + vE == 255u) && lE != 0u;
  bool wantSE = (v0 + vSE == 255u) && lSE != 0u;
  bool wantS = (v0 + vS == 255u) && lS != 0u;
  bool wantSW = !sw_is_duplicate && (v0 + vSW == 255u) && lSW != 0u;

  int gSE = (vSE > v0) ? 1 : -1;
  int gSWx = (vSW > v0) ? -1 : 1;
  int gSWy = (vSW > v0) ? 1 : -1;

  append(wantE, rep0, lE - 1u, x * 2u + 1u, y * 2u, (vE > v0) ? 1 : -1, 0);
  append(wantSE, rep0, lSE - 1u, x * 2u + 1u, y * 2u + 1u, gSE, gSE);
  append(wantS, rep0, lS - 1u, x * 2u, y * 2u + 1u, 0, (vS > v0) ? 1 : -1);
  append(wantSW, rep0, lSW - 1u, x * 2u - 1u, y * 2u + 1u, gSWx, gSWy);
}
