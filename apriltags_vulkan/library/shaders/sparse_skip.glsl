// Flat-region skipping for the labelling passes. threshold_block_u8.comp marks each 64 x 4 pixel cell
// that holds a block with contrast; a workgroup whose pixels all lie in unmarked cells is all
// ambiguous (127) and has nothing to label. The includer defines ACTIVE_CELLS_BINDING.
layout(std430, binding = ACTIVE_CELLS_BINDING) readonly buffer ActiveCells { uint active_cells[]; };

// True when every cell over pixel columns [x_first, x_last] and rows [y_first, y_last] is unmarked.
bool CellsFlat(uint width, uint x_first, uint x_last, uint y_first, uint y_last) {
  uint cells_x = (((width + 3u) / 4u) + 15u) / 16u;
  uint c_last = min(x_last / 64u, cells_x - 1u);
  for (uint r = y_first / 4u; r <= y_last / 4u; ++r) {
    for (uint c = x_first / 64u; c <= c_last; ++c) {
      if (active_cells[r * cells_x + c] != 0u) return false;
    }
  }
  return true;
}
