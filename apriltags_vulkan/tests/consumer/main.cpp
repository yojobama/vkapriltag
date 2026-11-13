// Detects the tags of one grey image with an installed vkapriltag and prints "id <n>" per detection.
// usage: consumer <image.png|image.pgm> [family]
#include <cstdio>
#include <cstdlib>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_PNM
#include "stb_image.h"

#include "vkapriltag/Family.h"
#include "vkapriltag/TagDecoder.h"
#include "vkapriltag/gpu/GpuDetector.h"
#include "vkapriltag/gpu/QuadDecode.h"
#include "vkapriltag/vk/Context.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    std::printf("usage: consumer <image> [family]\n");
    return 2;
  }
  const apriltag_vulkan::Family *tf = apriltag_vulkan::Family::ByName(argc > 2 ? argv[2] : "tag36h11");
  int width = 0, height = 0, channels = 0;
  stbi_uc *grey = stbi_load(argv[1], &width, &height, &channels, 1);
  if (tf == nullptr || grey == nullptr) {
    std::printf("cannot load %s or find the family\n", argv[1]);
    return 2;
  }
  try {
    apriltag_vulkan::vk::Context ctx;
    apriltag_vulkan::DetectorConfig config;
    config.width = static_cast<uint32_t>(width);
    config.height = static_cast<uint32_t>(height);
    config.tag_width = static_cast<uint32_t>(tf->width_at_border);
    config.reversed_border = tf->reversed_border;
    config.normal_border = !tf->reversed_border;
    apriltag_vulkan::GpuDetector detector(ctx, config);
    apriltag_vulkan::QuadDecode quad_decode(config);
    apriltag_vulkan::TagDecoderConfig decoder_config;
    decoder_config.families = {tf};
    decoder_config.refine_edges = true;
    apriltag_vulkan::TagDecoder tag_decoder(decoder_config, config.decimation);

    detector.Detect(grey);
    std::vector<apriltag_vulkan::DetectedQuad> quads = quad_decode.Decode(detector.last_line_fit_points);
    const apriltag_vulkan::DetectionList &detections = tag_decoder.Decode(quads, grey, width, height, config.reversed_border);
    for (const apriltag_vulkan::Detection &d : detections) std::printf("id %d hamming %d\n", d.id, d.hamming);
    std::printf("detections %zu\n", detections.size());
  } catch (const std::exception &e) {
    std::printf("error: %s\n", e.what());
    return 1;
  }
  stbi_image_free(grey);
  return 0;
}
