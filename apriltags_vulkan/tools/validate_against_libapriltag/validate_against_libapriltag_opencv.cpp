// Standalone validation tool: runs this project's full Vulkan+CPU detection
// pipeline (GpuDetector -> QuadDecode -> TagDecoder) on a static grayscale
// image and compares the resulting decoded tag IDs AND corner positions
// against the fetched `apriltag` C library's own, unmodified, reference CPU
// detector (apriltag_detector_detect()) run on the exact same image. This is
// the "verify against the official libapriltag outputs" check - not a
// manual/eyeballed comparison.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include "vkapriltag/FramePipeline.h"
#include "vkapriltag/TagDecoder.h"
#include "vkapriltag/apriltag_family.h"
#include "vkapriltag/gpu/GpuDetector.h"
#include "vkapriltag/gpu/QuadDecode.h"
#include "vkapriltag/vk/Context.h"
#include "validate_common.h"
#include <chrono>

#include <opencv2/opencv.hpp>

extern "C" {
#include "apriltag.h"
}

using apriltag_vulkan::validate::AppendCsvRow;
using apriltag_vulkan::validate::CompareCorners;
using apriltag_vulkan::validate::ComputeStats;
using apriltag_vulkan::validate::ExtractDetections;
using apriltag_vulkan::validate::ImageMetrics;
using apriltag_vulkan::validate::PrintGpuStageBreakdown;
using apriltag_vulkan::validate::PrintIds;
using apriltag_vulkan::validate::SortedIds;
using apriltag_vulkan::validate::Stats;

namespace {

void PrintStats(const char *label, const Stats &s, int iterations) {
  std::cout << "  " << label << ": best=" << s.best << " ms";
  if (iterations > 1) std::cout << ", median=" << s.median << " ms, worst=" << s.worst << " ms";
  std::cout << std::endl;
}

}  // namespace

int main(int argc, char **argv) {
  std::string load_path;
  std::string family_name = "tag36h11";
  std::string csv_path;
  // A single Detect() call is dominated by cold-start costs (first-touch page
  // faults across every buffer, pipeline warm-up), which say nothing about
  // per-frame throughput. Repeat and report the best/median to measure the
  // steady state a live camera feed would actually see. The WHOLE pipeline
  // (GPU + quad_decode + tag_decode) is timed per iteration, not just
  // Detect(), since CPU-tail changes are invisible to a GPU-only timer.
  int iterations = 1;
  // Matches DetectorConfig::decimation's default. Kept in sync with
  // td_ref->quad_decimate below so "verified against unmodified upstream
  // apriltag" stays true at every tested decimation factor, not just 2.
  uint32_t decimation = 2;
  // 0 = leave DetectorConfig's default. Exposed because a frame that
  // overflows max_blobs detects a different set of tags every run, so raising
  // it until selected_blob_drops reads zero is the way to confirm that is what
  // a nondeterministic result is caused by.
  uint32_t max_blobs = 0;
  // Run the frame-pipelined path (FramePipeline) instead of the serial
  // Detect -> QuadDecode -> TagDecoder chain, so the two can be A/B'd from one
  // binary. Throughput-only: per-iteration timings below stop being a
  // breakdown, since the GPU pass and the CPU tail deliberately overlap.
  bool pipelined = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&](const char *name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (arg == "--data") {
      load_path = next("--data");
    } else if (arg == "--family") {
      family_name = next("--family");
    } else if (arg == "--iterations") {
      iterations = std::max(1, std::stoi(next("--iterations")));
    } else if (arg == "--csv") {
      csv_path = next("--csv");
    } else if (arg == "--decimation") {
      decimation = static_cast<uint32_t>(std::max(1, std::stoi(next("--decimation"))));
    } else if (arg == "--max-blobs") {
      max_blobs = static_cast<uint32_t>(std::max(1, std::stoi(next("--max-blobs"))));
    } else if (arg == "--pipelined") {
      pipelined = true;
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return 1;
    }
  }

  if (load_path.empty()) {
    std::cerr << "Usage: apriltag_vulkan_validate --data <FileName> [--family tag36h11] "
                 "[--iterations N] [--csv path.csv] [--decimation N]"
             << std::endl;
    return 1;
  }

  std::filesystem::path path = load_path;
  std::error_code errorCode;
  std::vector<std::string> files;
  int match = 0, mismatch = 0;

  if (std::filesystem::is_directory(path, errorCode)) {
    std::cout << "Testing all images in directory: " << load_path << std::endl;

    for (const auto& entry : std::filesystem::directory_iterator(path)) {
      // Skips subdirectories if you only want regular files
      if (std::filesystem::is_regular_file(entry.path())) {
	cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Failed to load Image: " << entry.path().string() << std::endl;
            continue;
        }
        else if (img.cols % 2 != 0 || img.rows % 2 != 0) {
            std::cerr << "Skipping Image (dimensions not even): " << entry.path().string() << std::endl;
            continue;
        }
        files.push_back(entry.path().string());
      }
    }
  } else if (std::filesystem::is_regular_file(path, errorCode)) {
	files.push_back(path.string());
  } else {
    std::cerr << "Error: " << load_path << " is not a valid file or directory." << std::endl;
    return 1;
  }

  std::cout << "------------------------------------------------------------" << std::endl;
  for (const std::string& file : files) {
      cv::Mat image = cv::imread(file, cv::IMREAD_GRAYSCALE);
      if (image.empty()) {
          std::cerr << "Failed to load Image: " << file << std::endl;
          return 1;
      }
      std::cout << "Loaded " << file << " (" << image.cols << "x" << image.rows << ")" << std::endl;

      uint32_t width = image.cols, height = image.rows;

      // initialize the vkapritlag stuff
      apriltag_vulkan::vk::Context ctx;
      apriltag_family_t* tf = nullptr;
      if (!setup_tag_family(&tf, family_name.c_str())) {
          return 1;
      }
      apriltag_detector_t* td_ours = apriltag_detector_create();
      apriltag_detector_add_family(td_ours, tf);
      // td_ref below is never told otherwise, so it runs in upstream's own
      // default config (refine_edges = true, apriltag.c) - matching that
      // here is what this tool's own header comment promises: verification
      // against the actual official outputs, not a handicapped comparison.
      td_ours->refine_edges = true;
      apriltag_vulkan::DetectorConfig config;
      config.width = width;
      config.height = height;
      config.decimation = decimation;
      if (max_blobs > 0) config.max_blobs = max_blobs;
      config.tag_width = static_cast<uint32_t>(tf->width_at_border);
      config.reversed_border = tf->reversed_border;
      config.normal_border = !tf->reversed_border;

      apriltag_vulkan::GpuDetector detector(ctx, config);
      apriltag_vulkan::QuadDecode quad_decode(config);
      apriltag_vulkan::TagDecoder tag_decoder(td_ours, decimation);

      ImageMetrics metrics;
      metrics.file = file;
      metrics.width = width;
      metrics.height = height;
      metrics.iterations = iterations;

      std::vector<double> gpu_totals, quad_decode_totals, tag_decode_totals, pipeline_totals;
      gpu_totals.reserve(static_cast<size_t>(iterations));
      quad_decode_totals.reserve(static_cast<size_t>(iterations));
      tag_decode_totals.reserve(static_cast<size_t>(iterations));
      pipeline_totals.reserve(static_cast<size_t>(iterations));

      zarray_t* ours = nullptr;
      apriltag_vulkan::GpuDetector::DetectProfile profile;
      std::vector<apriltag_vulkan::DetectedQuad> quads;

      if (pipelined) {
        // Pushing the same buffer every iteration is safe here only because
        // both consumers read it: the GPU pass copies it into its staging
        // buffer and TagDecoder samples it. A live camera must alternate two
        // buffers - see FramePipeline::Push.
        apriltag_vulkan::FramePipeline pipe(detector, quad_decode, tag_decoder);
        for (int it = 0; it < iterations; ++it) {
          const auto t0 = std::chrono::steady_clock::now();
          zarray_t *done = pipe.Push(image.data, width, height, config.reversed_border);
          const auto t1 = std::chrono::steady_clock::now();
          // Wall time per pushed frame - the throughput number. It is NOT
          // comparable to the serial path's per-stage breakdown, which is why
          // the three stage vectors stay empty in this mode.
          pipeline_totals.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
          if (done != nullptr) {
            ours = done;
            profile = pipe.last_profile();
            gpu_totals.push_back(profile.total_ms);
          }
        }
        if (zarray_t *tail = pipe.Flush()) {
          ours = tail;
          profile = pipe.last_profile();
          gpu_totals.push_back(profile.total_ms);
        }
        quads = pipe.last_quads();
      } else {
      for (int it = 0; it < iterations; ++it) {
          const auto t0 = std::chrono::steady_clock::now();
          detector.Detect(image.data);
          profile = detector.last_profile();
          gpu_totals.push_back(profile.total_ms);

          const auto t_quad0 = std::chrono::steady_clock::now();
          quads = quad_decode.Decode(detector.last_line_fit_points);
          const auto t_quad1 = std::chrono::steady_clock::now();
          quad_decode_totals.push_back(
              std::chrono::duration<double, std::milli>(t_quad1 - t_quad0).count());

          const auto t_dec0 = std::chrono::steady_clock::now();
          ours = tag_decoder.Decode(quads, image.data, width, height, config.reversed_border);
          const auto t_dec1 = std::chrono::steady_clock::now();
          tag_decode_totals.push_back(
              std::chrono::duration<double, std::milli>(t_dec1 - t_dec0).count());

          pipeline_totals.push_back(
              std::chrono::duration<double, std::milli>(t_dec1 - t0).count());
      }
      }

      metrics.gpu_total_ms = ComputeStats(gpu_totals);
      metrics.quad_decode_ms = ComputeStats(quad_decode_totals);
      metrics.tag_decode_ms = ComputeStats(tag_decode_totals);
      metrics.pipeline_ms = ComputeStats(pipeline_totals);
      metrics.candidate_quads = static_cast<uint32_t>(quads.size());
      metrics.boundary_points = profile.boundary_points;
      metrics.raw_blobs = profile.raw_blobs;
      metrics.selected_blobs = profile.selected_blobs;
      metrics.points = profile.points;
      metrics.uf_iterations = profile.uf_iterations;
      metrics.uf_converged = profile.uf_converged;
      {
        const apriltag_vulkan::QuadDecode::DpStats dp_stats = quad_decode.last_dp_stats();
        metrics.dp_attempts = dp_stats.attempts;
        metrics.dp_fallbacks = dp_stats.fallbacks;
      }

      std::cout << quads.size() << " candidate quad(s) from the Vulkan pipeline." << std::endl;
      std::cout << detector.DescribeSizing() << std::endl;
      std::cout << "GPU profile (last iteration): total=" << profile.total_ms
          << " ms (upload=" << profile.upload_ms
          << ", threshold+label=" << profile.threshold_label_ms
          << ", boundary=" << profile.boundary_ms << ", sort+group=" << profile.sort_group_ms
          << ", linefit=" << profile.linefit_ms << ", readback=" << profile.readback_ms
          << " ms)" << std::endl;
      std::cout << "  work: boundary_points=" << profile.boundary_points
          << ", raw_blobs=" << profile.raw_blobs
          << (profile.hash_probe_drops > 0
                  ? (" (" + std::to_string(profile.hash_probe_drops) + " HASH DROPS)")
                  : std::string())
          << ", uf_iterations=" << profile.uf_iterations
          << (profile.uf_converged ? "" : " (HIT LIMIT)")
          << (profile.selected_blob_drops > 0
                  ? (" (" + std::to_string(profile.selected_blob_drops) +
                     " BLOBS DROPPED - max_blobs exceeded, detections are NOT reproducible)")
                  : std::string())
          << ", submits=" << profile.submits << ", blobs=" << profile.selected_blobs
          << ", points=" << profile.points
          << (profile.oversized_sort_blobs > 0
                  ? (", oversized_unsorted=" +
                     std::to_string(profile.oversized_sort_blobs))
                  : std::string())
          << std::endl;
      std::cout << "  bytes: upload=" << profile.upload_bytes
          << ", readback=" << profile.readback_bytes << std::endl;
      const double gpu_span_total = PrintGpuStageBreakdown(profile);
      // Host-side cost of driving the GPU - see DetectProfile's cpu_*_ms
      // comment for what the residual below is (and, importantly, what it
      // is not).
      std::cout << "  Host-side submission cost (last iteration): begin="
          << profile.cpu_begin_ms << " ms, submit+wait=" << profile.cpu_submit_wait_ms
          << " ms, counter_reads=" << profile.cpu_counter_read_ms << " ms, over "
          << profile.submits << " submit(s)" << std::endl;
      if (profile.has_gpu_stage_breakdown) {
        // Deliberately NOT divided by the submit count: that reading
        // ("cost per round trip") was tested by removing a submission and
        // disproved - see DetectProfile's cpu_*_ms comment. This is GPU-side
        // time no span covers, dominated by inter-dispatch barriers.
        std::cout << "    unspanned GPU time (submit+wait minus spans) = "
            << (profile.cpu_submit_wait_ms - gpu_span_total) << " ms" << std::endl;
      }
      std::cout << "Whole-pipeline stage timings over " << iterations << " iteration(s):" << std::endl;
      PrintStats("GPU total", metrics.gpu_total_ms, iterations);
      PrintStats("quad_decode", metrics.quad_decode_ms, iterations);
      PrintStats("tag_decode", metrics.tag_decode_ms, iterations);
      PrintStats("pipeline_total", metrics.pipeline_ms, iterations);
      if (metrics.dp_attempts > 0) {
        std::cout << "DP corner seeding: " << metrics.dp_fallbacks << "/" << metrics.dp_attempts
            << " blobs fell back to the combinatorial search ("
            << (100.0 * metrics.dp_fallbacks / metrics.dp_attempts) << "%)" << std::endl;
      }

      std::cout << "--- Our detections ---" << std::endl;
      print_detections(ours);
      std::vector<apriltag_vulkan::validate::DetectionInfo> our_dets = ExtractDetections(ours);
      std::vector<int> our_ids = SortedIds(our_dets);

      apriltag_detector_t* td_ref = apriltag_detector_create();
      apriltag_detector_add_family(td_ref, tf);
      td_ref->quad_decimate = static_cast<float>(decimation);
      td_ref->nthreads = 1;

      image_u8_t im{
        .width = static_cast<int32_t>(width),
        .height = static_cast<int32_t>(height),
        .stride = static_cast<int32_t>(width),
        .buf = image.data,
      };
      const auto t2 = std::chrono::steady_clock::now();
      zarray_t* ref = apriltag_detector_detect(td_ref, &im);
      const auto t3 = std::chrono::steady_clock::now();
      std::cout << "Reference detection time: "
          << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
          << " ms" << std::endl;
      std::cout << "--- Reference (official libapriltag) detections ---" << std::endl;
      print_detections(ref);
      std::vector<apriltag_vulkan::validate::DetectionInfo> ref_dets = ExtractDetections(ref);
      std::vector<int> ref_ids = SortedIds(ref_dets);

      PrintIds("Our tag IDs", our_ids);
      PrintIds("Reference tag IDs", ref_ids);

      metrics.corners = CompareCorners(our_dets, ref_dets);
      if (metrics.corners.compared_tags > 0) {
        std::cout << "Corner RMS over " << metrics.corners.compared_tags
            << " tag(s) present in both: mean=" << metrics.corners.mean_rms
            << " px, max=" << metrics.corners.max_rms << " px" << std::endl;
      }

      metrics.ids_match = (our_ids == ref_ids);
      if (metrics.ids_match) {
          std::cout << "MATCH: decoded tag ID set agrees with the official libapriltag detector."
              << std::endl;
          match++;
      }
      else {
          std::cout << "MISMATCH: decoded tag ID set differs from the official libapriltag detector."
              << std::endl;
          mismatch++;
      }

      if (!csv_path.empty()) AppendCsvRow(csv_path, metrics);
	  std::cout << "------------------------------------------------------------" << std::endl;
    }

    std::cout << "Final Results: " << match << " matches, " << mismatch << " mismatches." << std::endl;

  // Exit non-zero on any mismatch, matching the PGM-only variant, so this can
  // gate a scripted run rather than only being read by a human. Zero images
  // validated is also a failure: a mistyped --data path or a directory whose
  // every image was skipped (unloadable, or odd dimensions) would otherwise
  // report success having checked nothing at all.
  if (mismatch > 0) return 1;
  if (match == 0) {
    std::cerr << "No images were validated - nothing to compare against libapriltag." << std::endl;
    return 1;
  }
  return 0;
}
