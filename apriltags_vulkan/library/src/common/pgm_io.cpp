#include "vkapriltag/common/pgm_io.h"

#include <cctype>
#include <fstream>

namespace apriltag_vulkan {
namespace {

// Skips whitespace and '#'-prefixed comments, per the NetPBM plain-header
// grammar (used for the P5 magic/width/height/maxval fields, before the
// raw binary pixel data begins).
void SkipWhitespaceAndComments(std::istream &in) {
  for (;;) {
    int c = in.peek();
    if (c == '#') {
      std::string line;
      std::getline(in, line);
      continue;
    }
    if (c != EOF && std::isspace(c)) {
      in.get();
      continue;
    }
    break;
  }
}

bool ReadHeaderToken(std::istream &in, int *value) {
  SkipWhitespaceAndComments(in);
  return static_cast<bool>(in >> *value);
}

}  // namespace

bool LoadGrayPgm(const std::string &path, std::vector<uint8_t> *out_pixels, uint32_t *out_width,
                 uint32_t *out_height) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;

  char magic[2] = {0, 0};
  f.get(magic[0]);
  f.get(magic[1]);
  if (magic[0] != 'P' || magic[1] != '5') return false;

  int width = 0, height = 0, maxval = 0;
  if (!ReadHeaderToken(f, &width)) return false;
  if (!ReadHeaderToken(f, &height)) return false;
  if (!ReadHeaderToken(f, &maxval)) return false;
  if (width <= 0 || height <= 0 || maxval <= 0 || maxval > 255) return false;

  // Exactly one whitespace character separates maxval from the raw pixel
  // data; ReadHeaderToken's istream >> already consumed it.
  f.get();  // consume the single mandatory separator after maxval.

  std::vector<uint8_t> pixels(static_cast<size_t>(width) * height);
  f.read(reinterpret_cast<char *>(pixels.data()), pixels.size());
  if (!f) return false;

  *out_pixels = std::move(pixels);
  *out_width = static_cast<uint32_t>(width);
  *out_height = static_cast<uint32_t>(height);
  return true;
}

}  // namespace apriltag_vulkan
