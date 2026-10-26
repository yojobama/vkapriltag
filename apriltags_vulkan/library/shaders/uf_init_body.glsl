// Shared body for uf_init.comp / uf_init_u8.comp; the wrapper declares Thresholded and
// THRESHOLDED_AT(i).
//
// Initialises the union-find parent array. Each thread covers kRun consecutive pixels of one row:
// a pixel that joins its left neighbour (same non-ambiguous value) points at the start of its run
// within the chunk, or at the chunk's left neighbour when the run began before the chunk. Every run
// is rooted at its leftmost index; ambiguous pixels (127) stay their own root.
layout(local_size_x_id = 0, local_size_x = 16, local_size_y_id = 1, local_size_y = 16) in;
layout(constant_id = 3) const uint kRun = 4;

layout(std430, binding = 0) writeonly buffer Parent { uint parent[]; };

layout(push_constant) uniform PushConstants {
  uint width;
  uint height;
} pc;

void main() {
  uint x0 = gl_GlobalInvocationID.x * kRun;
  uint y = gl_GlobalInvocationID.y;
  if (x0 >= pc.width || y >= pc.height) return;
  uint row = y * pc.width;
  uint x_end = min(x0 + kRun, pc.width);

  uint prev = (x0 > 0u) ? uint(THRESHOLDED_AT(row + x0 - 1u)) : 0u;
  uint run_start = row + x0 - 1u;
  for (uint x = x0; x < x_end; ++x) {
    uint i = row + x;
    uint v = uint(THRESHOLDED_AT(i));
    bool join_left = (v != 127u) && (x > 0u) && (prev == v);
    if (!join_left) run_start = i;
    parent[i] = run_start;
    prev = v;
  }
}
