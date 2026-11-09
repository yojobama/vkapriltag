#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "vkapriltag/TagDecoder.h"
#include "vkapriltag/gpu/GpuDetector.h"
#include "vkapriltag/gpu/QuadDecode.h"

extern "C" {
#include "apriltag.h"
}

namespace apriltag_vulkan {

struct TrackingConfig {
  // Off: Detect() always runs the full GPU pipeline. Env override: APRILTAG_VK_TRACKING=0|1
  bool enabled = false;
  // At most this many consecutive frames skip the GPU; the next one runs the full detector, which is
  // also the only place new tags are found. Env override: APRILTAG_VK_TRACKING_KEYFRAME=<n>
  uint32_t keyframe_interval = 8;
  // A tracked tag must decode to its own ID with at least this decision margin, and at least
  // min_margin_ratio times its previous margin. Env override: APRILTAG_VK_TRACKING_MIN_MARGIN=<m>
  float min_margin = 10.0f;
  float min_margin_ratio = 0.3f;
  // A tracked tag's centre may move this fraction of its side from the prediction, and its side may
  // change by up to this ratio.
  float max_jump_frac = 0.5f;
  float max_scale_change = 1.4f;
  // When the predicted corners do not decode, search a region around the tag with libapriltag's quad
  // detector before giving up. Env override: APRILTAG_VK_TRACKING_SEARCH=0|1
  bool roi_search = true;
  // The search region's padding: at least roi_pad_min pixels and roi_pad_frac of the tag's side.
  float roi_pad_frac = 0.75f;
  float roi_pad_min = 16.0f;
};

// Runs the full detector on keyframes and, between them, re-detects the tags of the previous frame on
// the CPU alone, from corners predicted by constant velocity (refine, then decode); a tag that does not
// decode is searched for in a padded region, and if any tag is still missing the frame goes to the full
// detector. Detections have the same type as TagDecoder::Decode's.
class TrackingDetector {
 public:
  enum class Path { kFull, kTracked, kSearch };

  // `td` must already have its tag families added and refine_edges set; it is not owned. `config`
  // is for the full detector; `decimation` of the tracked frames' refinement matches it.
  TrackingDetector(vk::Context &ctx, apriltag_detector_t *td, const DetectorConfig &config,
                   const TrackingConfig &tracking = {});
  ~TrackingDetector();

  TrackingDetector(const TrackingDetector &) = delete;
  TrackingDetector &operator=(const TrackingDetector &) = delete;

  // Detects the tags of one grayscale frame (width*height bytes, as the config). Returns detections
  // owned by this object, valid until the next Detect().
  zarray_t *Detect(const uint8_t *gray_frame);

  // Forgets every track, so the next frame runs the full detector.
  void Reset();
  void SetEnabled(bool enabled);
  bool enabled() const { return tracking_.enabled; }

  Path last_path() const { return last_path_; }
  GpuDetector &gpu() { return *gpu_; }
  const DetectorConfig &config() const { return config_; }

  struct Stats {
    uint64_t frames_full = 0;
    uint64_t frames_tracked = 0;
    uint64_t frames_search = 0;
    uint64_t tracked_attempts_failed = 0;  // tracked frames that fell through to the full detector
  };
  const Stats &stats() const { return stats_; }

 private:
  struct Track {
    int id = 0;
    double p[4][2] = {};
    double v[4][2] = {};
    float margin = 0.0f;
  };

  zarray_t *DetectFull(const uint8_t *gray_frame);
  zarray_t *TryTrack(const uint8_t *gray_frame, bool *searched);
  void PredictQuad(const Track &t, DetectedQuad *q) const;
  bool Accept(const apriltag_detection_t *det, const Track &t, const DetectedQuad &from,
              bool strict) const;
  bool SearchQuad(const uint8_t *gray_frame, const Track &t, DetectedQuad *out);
  void UpdateTracks(zarray_t *dets);

  DetectorConfig config_;
  TrackingConfig tracking_;
  apriltag_detector_t *td_;  // not owned
  apriltag_detector_t *search_td_;
  std::unique_ptr<GpuDetector> gpu_;
  std::unique_ptr<QuadDecode> quad_decode_;
  std::unique_ptr<TagDecoder> full_decoder_;
  std::unique_ptr<TagDecoder> track_decoder_;
  std::vector<Track> tracks_;
  std::vector<uint8_t> crop_;
  uint32_t tracked_run_ = 0;
  Path last_path_ = Path::kFull;
  Stats stats_;
};

}  // namespace apriltag_vulkan
