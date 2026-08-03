// Standalone validation tool: runs this project's full Vulkan+CPU detection
// pipeline (GpuDetector -> QuadDecode -> TagDecoder) on a static grayscale
// PGM image and compares the resulting decoded tag IDs against the fetched
// `apriltag` C library's own, unmodified, reference CPU detector
// (apriltag_detector_detect()) run on the exact same image. This is the
// "verify against the official libapriltag outputs" check - not a manual/
// eyeballed comparison.
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "vkapriltag/TagDecoder.h"
#include "vkapriltag/apriltag_family.h"
#include "vkapriltag/common/pgm_io.h"
#include "vkapriltag/gpu/GpuDetector.h"
#include "vkapriltag/gpu/QuadDecode.h"
#include "vkapriltag/vk/Context.h"
#include <chrono>

extern "C" {
#include "apriltag.h"
}

namespace {

std::vector<int> SortedIds(const zarray_t *detections) {
  std::vector<int> ids;
  for (int i = 0; i < zarray_size(detections); ++i) {
    apriltag_detection_t *det;
    zarray_get(const_cast<zarray_t *>(detections), i, &det);
    ids.push_back(det->id);
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

void PrintIds(const char *label, const std::vector<int> &ids) {
  std::cout << label << ": [";
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i) std::cout << ", ";
    std::cout << ids[i];
  }
  std::cout << "]" << std::endl;
}

}  // namespace

int main(int argc, char **argv) {
  std::string pgm_path;
  std::string family_name = "tag36h11";
  // A single Detect() call is dominated by cold-start costs (first-touch page
  // faults across every buffer, pipeline warm-up), which say nothing about
  // per-frame throughput. Repeat and report the best/median to measure the
  // steady state a live camera feed would actually see.
  int iterations = 1;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&](const char *name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (arg == "--pgm") {
      pgm_path = next("--pgm");
    } else if (arg == "--family") {
      family_name = next("--family");
    } else if (arg == "--iterations") {
      iterations = std::max(1, std::stoi(next("--iterations")));
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return 1;
    }
  }

  if (pgm_path.empty()) {
    std::cerr << "Usage: apriltag_vulkan_validate --pgm <file.pgm> [--family tag36h11] "
                 "[--iterations N]"
             << std::endl;
    return 1;
  }

  std::vector<uint8_t> gray;
  uint32_t width = 0, height = 0;
  if (!apriltag_vulkan::LoadGrayPgm(pgm_path, &gray, &width, &height)) {
    std::cerr << "Failed to load PGM: " << pgm_path << std::endl;
    return 1;
  }
  std::cout << "Loaded " << pgm_path << " (" << width << "x" << height << ")" << std::endl;

  apriltag_family_t *tf = nullptr;
  if (!setup_tag_family(&tf, family_name.c_str())) {
    return 1;
  }

  int result = 1;
  try {
    // --- Our pipeline: Vulkan GPU detector + CPU QuadDecode + TagDecoder ---
    apriltag_detector_t *td_ours = apriltag_detector_create();
    apriltag_detector_add_family(td_ours, tf);
    td_ours->refine_edges = false;  // RefineEdges is not ported - see README.md.

    apriltag_vulkan::vk::Context ctx;
    apriltag_vulkan::DetectorConfig config;
    config.width = width;
    config.height = height;
    config.tag_width = static_cast<uint32_t>(tf->width_at_border);
    config.reversed_border = tf->reversed_border;
    config.normal_border = !tf->reversed_border;

    apriltag_vulkan::GpuDetector detector(ctx, config);
    apriltag_vulkan::QuadDecode quad_decode(config);
    apriltag_vulkan::TagDecoder tag_decoder(td_ours);

    // Warm-up plus timed repeats. Every iteration processes the same pixels, so
    // the detection result is identical; only the timing differs.
    std::vector<double> gpu_totals;
    gpu_totals.reserve(static_cast<size_t>(iterations));
    for (int it = 0; it < iterations; ++it) {
      detector.Detect(gray.data());
      gpu_totals.push_back(detector.last_profile().total_ms);
    }

    const auto t0 = std::chrono::steady_clock::now();
    detector.Detect(gray.data());
    const auto &profile = detector.last_profile();

    const auto t_quad0 = std::chrono::steady_clock::now();
    std::vector<apriltag_vulkan::DetectedQuad> quads =
        quad_decode.Decode(detector.last_selected_extents, detector.last_line_fit_points);
    const auto t_quad1 = std::chrono::steady_clock::now();

    const auto t_dec0 = std::chrono::steady_clock::now();
    zarray_t *ours = tag_decoder.Decode(quads, gray.data(), width, height, config.reversed_border);
    const auto t_dec1 = std::chrono::steady_clock::now();

    const auto t1 = std::chrono::steady_clock::now();
    std::cout << quads.size() << " candidate quad(s) from the Vulkan pipeline." << std::endl;
    std::cout << detector.DescribeSizing() << std::endl;
    std::cout << "GPU profile: total=" << profile.total_ms << " ms (upload=" << profile.upload_ms
              << ", threshold+label=" << profile.threshold_label_ms
              << ", boundary=" << profile.boundary_ms << ", sort+group=" << profile.sort_group_ms
              << ", linefit=" << profile.linefit_ms << ", readback=" << profile.readback_ms
              << " ms)" << std::endl;
    std::cout << "  work: boundary_points=" << profile.boundary_points
              << ", qbp_sort_n=" << profile.qbp_sort_n
              << ", ipoint_sort_n=" << profile.ipoint_sort_n
              << ", uf_iterations=" << profile.uf_iterations
              << (profile.uf_converged ? "" : " (HIT LIMIT)")
              << ", submits=" << profile.submits << ", blobs=" << profile.selected_blobs
              << ", points=" << profile.points << std::endl;
    std::cout << "  bytes: upload=" << profile.upload_bytes
              << ", readback=" << profile.readback_bytes << std::endl;
    if (gpu_totals.size() > 1) {
      std::vector<double> sorted = gpu_totals;
      std::sort(sorted.begin(), sorted.end());
      std::cout << "  over " << sorted.size() << " iterations: first=" << gpu_totals.front()
                << " ms, best=" << sorted.front() << " ms, median="
                << sorted[sorted.size() / 2] << " ms, worst=" << sorted.back() << " ms"
                << std::endl;
    }
    std::cout << "CPU profile: quad_decode="
          << std::chrono::duration<double, std::milli>(t_quad1 - t_quad0).count()
          << " ms, tag_decode="
          << std::chrono::duration<double, std::milli>(t_dec1 - t_dec0).count()
          << " ms, pipeline_total="
          << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms" << std::endl;
    std::cout << "--- Our detections ---" << std::endl;
    print_detections(ours);
    std::vector<int> our_ids = SortedIds(ours);

    apriltag_detector_destroy(td_ours);

    // --- Reference: the fetched apriltag library's own, unmodified,
    // full CPU detection pipeline (threshold -> quad detection -> decode),
    // run independently on the exact same pixels. ---
    apriltag_detector_t *td_ref = apriltag_detector_create();
    apriltag_detector_add_family(td_ref, tf);
    td_ref->quad_decimate = 2.0;
    td_ref->nthreads = 1;

    image_u8_t im{
        .width = static_cast<int32_t>(width),
        .height = static_cast<int32_t>(height),
        .stride = static_cast<int32_t>(width),
        .buf = gray.data(),
    };
    const auto t2 = std::chrono::steady_clock::now();
    zarray_t *ref = apriltag_detector_detect(td_ref, &im);
    const auto t3 = std::chrono::steady_clock::now();
    std::cout << "Reference detection time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
              << " ms" << std::endl;
    std::cout << "--- Reference (official libapriltag) detections ---" << std::endl;
    print_detections(ref);
    std::vector<int> ref_ids = SortedIds(ref);

    PrintIds("Our tag IDs", our_ids);
    PrintIds("Reference tag IDs", ref_ids);

    if (our_ids == ref_ids) {
      std::cout << "MATCH: decoded tag ID set agrees with the official libapriltag detector."
               << std::endl;
      result = 0;
    } else {
      std::cout << "MISMATCH: decoded tag ID set differs from the official libapriltag detector."
               << std::endl;
      result = 1;
    }

    apriltag_detections_destroy(ref);
    apriltag_detector_destroy(td_ref);
  } catch (const std::exception &e) {
    std::cerr << "Fatal error: " << e.what() << std::endl;
    result = 1;
  }

  teardown_tag_family(&tf, family_name.c_str());
  return result;
}
