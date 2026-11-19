#pragma once

#include <cstdint>
#include <span>
#include <string_view>

namespace apriltag_vulkan {

// A tag family: its code words and the cells that carry them. The tables are static; a Family is a
// small value, so a copy with another `max_hamming` is cheap.
struct Family {
  const char *name;
  uint32_t ncodes;
  const uint64_t *codes;
  int width_at_border;
  int total_width;
  bool reversed_border;
  uint32_t nbits;
  const uint32_t *bit_x;  // nbits entries: the cell of code bit k, from the corner of the border square
  const uint32_t *bit_y;
  uint32_t h;  // minimum Hamming distance between any two codes
  int max_hamming = 2;  // wrong bits the decode corrects, 0 to 3

  // The built-in family called `name` (tag36h11, tag25h9, tag16h5, tagCircle21h7, tagCircle49h12,
  // tagStandard41h12, tagStandard52h13 or tagCustom48h12), or null.
  static const Family *ByName(std::string_view name);
  static std::span<const Family *const> All();
};

}  // namespace apriltag_vulkan
