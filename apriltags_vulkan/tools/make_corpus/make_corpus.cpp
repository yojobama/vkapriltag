// Phase 0 corpus generation: rescales a tracked source image into a spread
// of resolutions, giving the validate tool a set of images with different
// effective tag pixel sizes to sweep the item-2 geometric-prefilter
// thresholds (min_tag_pixels, aspect_max, fill bounds) against.
//
// Caveat, stated plainly: rescaling the whole frame is a proxy for "the tag
// is farther away", not an identical stand-in for it - background clutter,
// noise and blur all shrink proportionally too, which a real long-range shot
// would not do identically. It is good enough to validate that a
// resolution-relative threshold behaves sanely across scales and to catch
// gross regressions, but it is not a substitute for real multi-distance
// captures before trusting the thresholds in the field.
#define NOMINMAX
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace {

// The directory-mode loader in validate_against_libapriltag_opencv.cpp skips
// images whose dimensions aren't even (the GPU pipeline's decimation halves
// each dimension), so every generated image must round to that.
int RoundToEven(double value) {
  int rounded = static_cast<int>(std::lround(value / 2.0)) * 2;
  return std::max(2, rounded);
}

}  // namespace

int main(int argc, char **argv) {
  std::string input_path;
  std::string out_dir;
  // Downscale only (<= 1.0). Upscaling synthesizes pixels the source image
  // never had - it manufactures a smoother/differently-interpolated edge
  // profile rather than the resolution loss a real distant/small tag would
  // actually show, so it is not a valid stand-in for "the tag is farther
  // away" and must not be used for validating the resolution-relative
  // thresholds (item 2) or corner-fit changes (item 3).
  std::vector<double> scales = {0.25, 0.375, 0.5, 0.75, 1.0};

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&](const char *name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (arg == "--input") {
      input_path = next("--input");
    } else if (arg == "--out-dir") {
      out_dir = next("--out-dir");
    } else if (arg == "--scales") {
      std::string csv = next("--scales");
      scales.clear();
      std::stringstream ss(csv);
      std::string tok;
      while (std::getline(ss, tok, ',')) scales.push_back(std::stod(tok));
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return 1;
    }
  }

  if (input_path.empty() || out_dir.empty()) {
    std::cerr << "Usage: make_corpus --input <image> --out-dir <dir> "
                 "[--scales 0.25,0.375,0.5,0.75,1.0]"
             << std::endl;
    return 1;
  }

  cv::Mat src = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
  if (src.empty()) {
    std::cerr << "Failed to load input image: " << input_path << std::endl;
    return 1;
  }
  std::cout << "Loaded " << input_path << " (" << src.cols << "x" << src.rows << ")" << std::endl;

  std::error_code ec;
  std::filesystem::create_directories(out_dir, ec);

  for (double scale : scales) {
    int w = RoundToEven(src.cols * scale);
    int h = RoundToEven(src.rows * scale);

    cv::Mat resized;
    // INTER_AREA is the right choice when shrinking (proper decimation,
    // avoids aliasing); INTER_LINEAR is fine for the occasional upscale.
    const int interp = (scale < 1.0) ? cv::INTER_AREA : cv::INTER_LINEAR;
    cv::resize(src, resized, cv::Size(w, h), 0, 0, interp);

    std::ostringstream name;
    name << out_dir << "/scale_" << scale << "_" << w << "x" << h << ".png";
    const std::string path = name.str();
    if (!cv::imwrite(path, resized)) {
      std::cerr << "Failed to write " << path << std::endl;
      return 1;
    }
    std::cout << "Wrote " << path << std::endl;
  }

  return 0;
}
