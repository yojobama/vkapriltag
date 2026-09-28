// Shared body for uf_merge.comp / uf_merge_u8.comp; the wrapper declares Thresholded and
// THRESHOLDED_AT(i). Parent is uint32 in both.
//
// One hooking pass of parallel union-find over 4-connected same-valued neighbours. Only the down
// edge is considered, as uf_init.comp has already joined horizontal runs. Ambiguous (127)
// pixels never merge. A union is performed only at the leftmost column of each run overlap
// (x == 0, or the left or below-left pixel differs). Run repeatedly, alternating with
// uf_compress.comp, until changed_flag stays 0; it is set only when a union joined two distinct
// components. Dispatched 2D with a one-row-tall workgroup so x is available without a divide.
layout(local_size_x_id = 0, local_size_x = 256) in;
// 0 = naive find(); 1 = path splitting (nodes on the walk are repointed to their grandparent
// with plain stores). Set by GpuDetector::CreatePipelines; APRILTAG_VK_FIND_MODE overrides it.
layout(constant_id = 3) const uint kFindMode = 0u;

layout(std430, binding = 0) buffer Parent { uint parent[]; };
layout(std430, binding = 2) buffer Changed { uint changed_flag; };

layout(push_constant) uniform PushConstants {
  uint width;
  uint height;
} pc;

shared uint wg_changed;

// kFindMode == 0: read-only walk to the root.
uint findNaive(uint n) {
  uint p = parent[n];
  while (p != n) {
    n = p;
    p = parent[n];
  }
  return n;
}

// kFindMode == 1: path splitting using plain non-atomic stores; a lost race only leaves a longer
// path.
uint findSplit(uint n) {
  uint p = parent[n];
  if (p != n) {
    uint prev = n;
    uint next = parent[p];
    while (p > next) {
      parent[prev] = next;
      prev = p;
      p = next;
      next = parent[p];
    }
    n = p;
  }
  return n;
}

uint find(uint n) {
  return (kFindMode != 0u) ? findSplit(n) : findNaive(n);
}

// Returns true if the two elements were in distinct components (i.e. this
// call performed real work and the labelling has not converged yet).
bool doUnion(uint a, uint b) {
  bool merged = false;
  bool done = false;
  while (!done) {
    a = find(a);
    b = find(b);
    if (a < b) {
      uint old = atomicMin(parent[b], a);
      done = (old == b);
      merged = true;
      b = old;
    } else if (b < a) {
      uint old = atomicMin(parent[a], b);
      done = (old == a);
      merged = true;
      a = old;
    } else {
      done = true;
    }
  }
  return merged;
}

// How the per-pixel "merged" result becomes the global changed_flag write.
//   0 = shared-memory aggregation: one atomicOr per workgroup.
//   1 = read-guarded global atomicOr per pixel, with no shared memory or barriers.
// Set by GpuDetector::CreatePipelines from ctx_.caps().unified_memory.
layout(constant_id = 4) const uint kMergeFlagMode = 0u;

void main() {
  uint x = gl_GlobalInvocationID.x;
  uint y = gl_GlobalInvocationID.y;

  if (kMergeFlagMode == 0u) {
    // Every invocation must reach both barriers, so the tests below select work rather than
    // returning early.
    if (gl_LocalInvocationID.x == 0u) wg_changed = 0u;
    memoryBarrierShared();
    barrier();
  }

  bool merged = false;
  if (x < pc.width && y + 1u < pc.height) {
    uint i = y * pc.width + x;
    uint v = THRESHOLDED_AT(i);
    if (v != 127u && THRESHOLDED_AT(i + pc.width) == v) {
      // Leftmost column of this run overlap?
      bool overlap_start = (x == 0u) || (THRESHOLDED_AT(i - 1u) != v) ||
                           (THRESHOLDED_AT(i - 1u + pc.width) != v);
      if (overlap_start && doUnion(i, i + pc.width)) merged = true;
    }
  }

  if (kMergeFlagMode == 0u) {
    if (merged) atomicOr(wg_changed, 1u);
    memoryBarrierShared();
    barrier();

    if (gl_LocalInvocationID.x == 0u && wg_changed != 0u) {
      atomicOr(changed_flag, 1u);
    }
  } else if (merged && changed_flag == 0u) {
    atomicOr(changed_flag, 1u);
  }
}
