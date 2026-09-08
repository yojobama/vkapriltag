# Taking the pose-estimation speedup upstream

A plan for contributing this repo's `PoseEstimator` work back to
[AprilRobotics/apriltag](https://github.com/AprilRobotics/apriltag)
(`apriltag_pose.c`), written against upstream v3.4.5.

## Short answer: yes, most of it transfers

The 50–60x is not an artifact of this being a C++ reimplementation, and it is
not algorithmic cleverness. It comes from removing `matd_op()` — a **runtime
string-expression interpreter** — from the solver's inner loop. Per call it
scans the expression, `malloc`s an argument array plus a `2*exprlen` garbage
array, recursively parses it character by character allocating intermediates,
copies the result, then frees everything. `orthogonal_iteration` issues ~16 of
those per iteration across 100 iterations (two 50-step solves), so roughly
**1600 interpreted expression evaluations per pose**. The actual arithmetic is
~100k FLOPs.

That is equally removable in C99 behind a completely unchanged public API. Two
of the four things this repo did, however, do **not** transfer as-is.

| Change made here | Transfers upstream? | Notes |
| --- | --- | --- |
| Fixed-size stack arithmetic replacing `matd_op` | **Yes, fully** | Pure C99; no API change required |
| Horner instead of `polyval`'s per-term `pow()` | **Yes, trivially** | Self-contained, ~3 lines |
| Hoisting `(F_j − I)`, computing `R·p_j` once instead of three times | **Yes** | Pure restructuring, no numerical change |
| Own 3x3 Jacobi SVD instead of `matd_svd` | Yes, but only as part of removing allocation | `matd_svd` is correct; it just allocates |
| `WorkerPool` threading across detections | **No** | Upstream never calls pose itself — the caller does, per detection. Threading belongs in the caller, not the library |
| Convergence early exit **as the default** | **Not as a default** | Changes numerical output; must be opt-in (Stage 4) |

## Constraints upstream has that this port did not

1. **C99, C-only project.** `LANGUAGES C`, `C_STANDARD 99`. This repo's
   implementation is C++. Nothing in the solver needs C++ — the math is plain
   `double` arrays and loops — but the port must drop `std::` and the worker
   pool.

2. **`apriltag_pose_t` holds `matd_t*`.** `R` and `t` are `matd_t*` in a
   public header. That cannot change without breaking every downstream
   consumer, so this is strictly an **internal** rewrite: stack arrays inside,
   allocate exactly the output matrices at the end. Per-pose allocation drops
   from ~1600 to 4 (`matd_create` allocates header and data separately, so two
   per output matrix).

3. **Six internals are non-static but appear in no header.**
   `matd_to_double`, `calculate_F`, `orthogonal_iteration`, `polyval`,
   `solve_poly_approx`, `fix_pose_ambiguities` are all exported symbols of a
   `SOVERSION 3` shared library while being absent from `apriltag_pose.h`.
   Nothing else in the library calls them (verified), but downstream code can
   declare such symbols itself — **this very repo does exactly that** for
   `quad_decode_index` and `reconcile_detections`. So silently removing them is
   a real ABI break, not a hypothetical one. See Stage 3.

4. **There is no pose test at all.** `test/test_detection.c` does not mention
   pose, and the Python bindings (`apriltag_pywrap.c`) do not expose it either.
   There is currently *nothing* protecting this file from a regression. This is
   the single most important fact in this plan, and it is why Stage 0 exists.

## The plan

Four stages, each independently reviewable and independently mergeable,
smallest and safest first. Do not combine them into one pull request: this is
a numerically delicate file with no existing tests, and a ~900-line rewrite
landing in one commit is a hard review to accept.

### Stage 0 — build the regression net first, changing no solver code

Nothing currently proves a pose change is safe. Establish golden values from
the **current** implementation, so every later stage is measured against
pre-change behaviour rather than against an assumption.

Add `test/test_pose.c`, wired into `test/CMakeLists.txt` beside the existing
detection test:

- Sweep synthetic poses (distance, tilt, in-plane rotation, off-axis
  translation), project the four tag corners through given intrinsics, build
  `H` via `homography_compute`, and exercise **all three** public entry points
  — `estimate_pose_for_tag_homography`, `estimate_tag_pose_orthogonal_iteration`
  (both solutions, both errors), and `estimate_tag_pose`. Testing each stage
  separately is what makes a later divergence localizable instead of merely
  visible.
- Assert against committed golden values **and** against the synthetic ground
  truth. Golden-only would pass if a change preserved behaviour while both
  were wrong; ground-truth-only would not catch a small regression inside the
  existing error budget.
- Include deliberately degenerate geometry: fronto-parallel (`tilt = 0`),
  `t` nearly parallel to `e_x` (the Gram-Schmidt degeneracy in
  `fix_pose_ambiguities`), extreme range, tag only a few pixels across.
- Record which ambiguity branch each case selects, and treat a flip as
  acceptable **only** when the two errors are a numerical tie.

One trap worth inheriting rather than rediscovering: **do not measure rotation
error with `acos((trace − 1) / 2)`.** `acos` has an infinite derivative at 1,
so for two nearly-identical rotations a single-ULP trace error becomes ~2.4e-6
degrees of phantom difference — a hard noise floor that hides real
regressions. It was caught here only because a solver appeared to differ from
*itself*. Use the axis-angle form instead:

```c
// D = A' * B; angle = atan2(|skew(D)| / 2, (trace(D) - 1) / 2)
```

**Deliverable:** PR 1, a pure test addition. Valuable on its own merits
regardless of whether anything below is ever merged.

### Stage 1 — cheap, local, obviously-safe wins

Keeps `matd_t` throughout, so the diff stays small and the review easy:

- **Horner `polyval`.** Currently `p[i]*pow(x, i)` per term — five libm `pow`
  calls per evaluation, inside a 100-iteration Newton loop, inside a 4-level
  recursion in `solve_poly_approx`.
- **Hoist `(F_j − I)`** out of the iteration loop; upstream re-derives it
  inside `matd_op` every step even though it is loop-invariant.
- **Compute `R·p_j` once per point per iteration.** Upstream's expression
  structure recomputes it three times: in the `M2` loop, the `q` loop, and the
  error loop.

**Expected gain: not yet measured — measure it with Stage 0's harness before
claiming a figure.** Reasoning suggests single-digit percent for Horner alone
(the quartic solve runs once per pose against 100 solver iterations), with the
hoisting worth more. Anyone landing this should report the measurement rather
than repeat this estimate.

### Stage 2 — the actual win: stack arrays behind the unchanged API

- Add `static` helpers over `double[3][3]` / `double[3]`: multiply,
  transpose-multiply, mat-vec, determinant, adjugate inverse, cross,
  normalize.
- Rewrite `orthogonal_iteration` and `fix_pose_ambiguities` internally against
  those. Public signatures and semantics unchanged.
- The only remaining allocation is the output `matd_t` per solution.

**The one genuine trap.** The closest-rotation step needs a *real* SVD, not a
polar-decomposition shortcut. `M3 = Σ_j (q_j − q_mean) · p_res_jᵀ` is **always
rank-deficient**, because the four tag corners are coplanar, so every `p_res_j`
has a zero z component and `M3`'s third column is identically zero. A Newton
polar iteration needs `M⁻¹` and collapses to a zero pose on exactly this
input — confirmed the hard way in a prototype here before it was fixed. Use
symmetric cyclic Jacobi on `MᵀM` for `V` and the squared singular values, take
`U`'s columns as `M·v_i / σ_i`, and recover the column belonging to the zero
singular value as the cross product of the other two.

Behaviour that must be preserved exactly, all of it load-bearing:

- The `det(R) < 0` fixup in `orthogonal_iteration` (negate `R`'s third column)
  — and **no** such fixup in `homography_to_pose`'s polar step, which upstream
  deliberately omits.
- `homography_to_pose`'s single-precision `sqrtf` for the scale factor, if bit
  parity is wanted. Using `sqrt` instead perturbs the seed by ~1e-7 relative;
  since orthogonal iteration is a *local* optimizer, that can very occasionally
  steer it into the other basin of the planar-pose ambiguity. Measured here as
  the dominant source of divergence from upstream. Keep `sqrtf` for exact
  parity, or change it deliberately and document it.
- The `hypotenuse < 1e-100` guard, `MAX_ROOT = 1000`, the fixed 100 Newton
  iterations with no convergence assertion, the
  `|t_cur − t_initial| > 0.1` distinctness test, and
  `n_minima > 1` → no second solution.
- A quirk to port **by intent, not by index arithmetic**:
  `fix_pose_ambiguities` builds `R_t` by reading `MATD_EL(R_t_1, 0, 1)` and
  `(0, 2)` from a **3x1** column vector — outside its declared column range,
  but landing on the right flat offsets in row-major storage. The effect is
  "the vector's three components as a matrix row"; that is what to write.

**Expected gain: the bulk of the 50–60x** (see Measured outcome below).

### Stage 3 — decide what happens to the six exported internals

Maintainers' call; both options are defensible:

- **(a) Keep them as thin `matd_t` wrappers** delegating to the new `static`
  internals, possibly marked deprecated. Zero ABI break, small ongoing
  maintenance cost. **Recommended.**
- **(b) Mark them `static` and bump `SOVERSION`.** Cleaner, and arguably what
  they should always have been, but it breaks downstream code that declared
  them itself — which, per constraint 3, demonstrably happens.

### Stage 4 — optional: convergence early exit, opt-in only

Upstream's `orthogonal_iteration` runs a fixed `n_steps` and computes a
per-step `error` that it assigns to `prev_error` and **never compares**. So
there is a free win available — but taking it changes numerical output, so it
cannot become the default behaviour of `estimate_tag_pose`.

- **Test the pose, not the error.** Near a minimum the error is quadratically
  flat in the pose, so an error-delta threshold of 1e-12 can be satisfied while
  the pose is still ~1e-6 from converged. Compare the relative change in `t`
  and the largest elementwise change in `R` against the previous step instead.
- Route it through a new entry point or an explicit tolerance parameter
  defaulting to "off", so `estimate_tag_pose` stays bit-compatible. There is
  precedent for exposing iteration control: `estimate_tag_pose_orthogonal_iteration`
  already takes `nIters`.
- Measured here over 210 synthetic poses, wall clock (not iteration count,
  which overstates the benefit by about a fifth because the seed, the
  per-point `F` precompute and the quartic solve are fixed overhead):

| tol | rotation vs fixed | ground-truth accuracy | wall clock saved |
| --- | --- | --- | --- |
| off | 0 | rot ≤ 0.006098°, dt ≤ 2.598e-06 | — |
| 1e-10 | 1.7e-07° | identical | 27–30% |
| **1e-08** | 2.8e-05° | **identical** | **52–54%** |
| 1e-06 | 2.2e-03° | *degrades* (0.006578°, 4.752e-06) | 78% |

  `1e-8` is the loosest tolerance leaving ground-truth accuracy identical to
  running all 50 iterations. `1e-6` is where real accuracy starts to go.
- **Caveat to state in the PR:** the saving is scene-dependent. 34 of those
  210 cases still hit the 50-iteration cap at `1e-8`, and so does a real
  1080p detection that happens to be near-fronto-parallel — precisely the
  ill-conditioned geometry where the iteration keeps twitching below the
  tolerance. Those cases save nothing.

## Measured outcome to expect

From this repo's implementation of the same algorithm, verified against
upstream at all four stages (agreement ≤ 5.2e-07° rotation and ≤ 3.8e-10
relative translation over 420 synthetic cases, with ground-truth accuracy
*identical* to upstream's):

| Per tag | upstream `estimate_tag_pose` | rewritten | speedup |
| --- | --- | --- | --- |
| RK3588 (Cortex-A76/A55) | 0.916 ms | 0.016 ms | **58x** |
| Ryzen 5 5600X | 0.616 ms | 0.013 ms | **48x** |

For a user tracking four tags per frame on an RK3588-class CPU that is roughly
**3.7 ms → 0.06 ms per frame**, which on embedded hardware is often the
difference between pose being affordable and not.

Note the upstream figure varies with the input: a detection whose ambiguity
search finds no second minimum runs one 50-step solve (~0.9 ms), while one that
finds a second runs two (~1.3 ms).

## Risks, and how each is retired

| Risk | Mitigation |
| --- | --- |
| Numerically delicate code with no existing tests | Stage 0 lands first, always |
| Rank-deficient `M3` breaking the rotation step | Explicit fronto-parallel (`tilt = 0`) test case; documented in Stage 2 |
| Ambiguity branch flipping on a near-tie | Assert the branch choice; permit flips only on exact numerical ties |
| Silent ABI break for downstream users | Stage 3 is a deliberate, separate decision |
| Reviewer burden | Four small PRs, cheapest and safest first |
| Licensing | Upstream is BSD 2-Clause; contribution-friendly |

## What not to propose

- **Threading inside libapriltag's pose.** Upstream never calls pose itself —
  the caller invokes it per detection — so parallelism belongs in the caller.
  (This repo threads it because *it* owns the loop over detections.)
- **A GPU implementation.** Assessed and rejected here: ~1 µs of arithmetic
  behind ~0.46 ms of queue-submission overhead, with 50 strictly sequential
  iterations and only 4 points to spread across a subgroup. It loses to a tight
  CPU implementation by 6–9x and only overtakes a threaded one past roughly 38
  simultaneous tags.
- **`-ffast-math`.** It would invalidate the numerical parity the tests assert.

## Reference implementation

`apriltags_vulkan/library/src/PoseEstimator.cpp` and
`apriltags_vulkan/tools/validate_pose/validate_pose.cpp` in this repo are a
working version of Stages 1, 2 and 4 plus the Stage 0 harness, in C++. They are
the place to look for the exact arithmetic, the SVD, and the verification
structure — but they are not a drop-in patch: the C99 port, the `matd_t`
boundary and the ABI decision are genuinely new work.
