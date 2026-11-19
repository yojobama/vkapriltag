// Detects the tags of one grey image with an installed vkapriltag and prints "id <n>" per detection.
// usage: consumer <image.png|image.pgm> [family]
#include <cstdio>
#include <cstdlib>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_PNM
#include "stb_image.h"

#include "vkapriltag/Detector.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    std::printf("usage: consumer <image> [family]\n");
    return 2;
  }
  int width = 0, height = 0, channels = 0;
  stbi_uc *grey = stbi_load(argv[1], &width, &height, &channels, 1);
  if (grey == nullptr) {
    std::printf("cannot load %s\n", argv[1]);
    return 2;
  }
  try {
    apriltag_vulkan::DetectorOptions options;
    options.width = static_cast<uint32_t>(width);
    options.height = static_cast<uint32_t>(height);
    options.families = {argc > 2 ? argv[2] : "tag36h11"};
    apriltag_vulkan::Detector detector(options);
    const apriltag_vulkan::Result &result = detector.Detect(grey);
    for (const apriltag_vulkan::Detection &d : result.detections) std::printf("id %d hamming %d\n", d.id, d.hamming);
    std::printf("detections %zu\n", result.detections.size());
  } catch (const apriltag_vulkan::Error &e) {
    std::printf("error: %s\n", e.what());
    return 1;
  }
  stbi_image_free(grey);
  return 0;
}
