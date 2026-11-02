// The cells listed by threshold_block_u8.comp: cell_list[0] is the count, the cells follow. A cell is
// 64 x 4 decimated pixels, numbered row by row. The includer defines CELL_LIST_BINDING.
layout(std430, binding = CELL_LIST_BINDING) readonly buffer CellList { uint cell_list[]; };

// Pixel column of the cell's left edge, and row `sub` of the cell, for workgroup `group` (four per cell).
void CellOfGroup(uint group, uint width, out uint x_first, out uint y) {
  uint cells_x = (((width + 3u) / 4u) + 15u) / 16u;
  uint cell = cell_list[1u + group / 4u];
  x_first = (cell % cells_x) * 64u;
  y = (cell / cells_x) * 4u + (group & 3u);
}
