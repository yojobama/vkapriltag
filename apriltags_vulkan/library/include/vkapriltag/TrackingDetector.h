#pragma once

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
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
  // The refined corners are fed back through refine and decode until they move less than converge_px
  // (at most max_iterations decodes); a tag that has not settled counts as lost.
  // Env overrides: APRILTAG_VK_TRACKING_CONVERGE=<px>, APRILTAG_VK_TRACKING_ITERATIONS=<n>
  // A first pass that moves a tag's corners more than max_first_shift_px gives up at once (the refine
  // is too slow to settle from there); the next 1, 2, 4... frames, up to keyframe_interval, then run full.
  float max_first_shift_px = 3.0f;
  float converge_px = 0.05f;
  // A track predicted to move more than this many pixels per frame (corner speed) is not tracked: on
  // blurred tags the refine settles up to 2 px off. 0 = no limit. Env override: APRILTAG_VK_TRACKING_MAX_SPEED=<px>
  float max_speed_px = 0.0f;
  uint32_t max_iterations = 3;
  // When the predicted corners do not decode, search a region around the tag with libapriltag's quad
  // detector before giving up. Env override: APRILTAG_VK_TRACKING_SEARCH=0|1
  bool roi_search = false;
  // The search region's padding: at least roi_pad_min pixels and roi_pad_frac of the tag's side.
  float roi_pad_frac = 0.75f;
  float roi_pad_min = 16.0f;
  // The region is decimated until the tag's side is about this many pixels, at most 4 times.
  float search_tag_px = 24.0f;
  // Widens the capture range of a track that failed its first refine: tries constant-acceleration and
  // zero-velocity starts, then refines on a box-decimated crop (by 2 to 8, from the speed) and finishes at full
  // resolution from that result. Env override: APRILTAG_VK_TRACKING_CAPTURE=0|1
  bool capture = false;
  // The decimated crop's refine must settle within this many crop pixels.
  float capture_converge_px = 0.1f;
  // The crop's padding around the tag, as a fraction of the tag's side (at least 16 pixels).
  float capture_pad_frac = 0.6f;
  // A track predicted to move more than this many pixels per frame starts from a decimated refine
  // even if its own prediction would decode (0 = only after a failure). Env override: APRILTAG_VK_TRACKING_CAPTURE_SPEED=<px>
  float capture_speed_px = 0.0f;
  // Asynchronous discovery (replaces the keyframes): the full detector runs on a background thread
  // while tracked frames return at the tracked cost; its tags join as tentative tracks that are
  // measured on the current frame before they are reported, and a lost track stays out of the output
  // until a background result finds it again. With no live track the frame runs the full detector
  // synchronously. Env override: APRILTAG_VK_TRACKING_ASYNC=0|1
  bool async = false;
  // The background detector starts at most every this many frames.
  // Env override: APRILTAG_VK_TRACKING_ASYNC_INTERVAL=<n>
  uint32_t async_interval = 1;
  // Threads of the background decoder (0 = all cores). Env override: APRILTAG_VK_TRACKING_ASYNC_THREADS=<n>
  uint32_t async_threads = 1;
  // Copies a frame for the background detector. Off: the caller keeps each frame valid until the next
  // Detect(), which waits for the background pass if it still reads it.
  // Env override: APRILTAG_VK_TRACKING_ASYNC_COPY=0|1
  bool async_copy = true;
  // A tracked tag that fails leaves the output until a background result finds it again; off: the
  // frame runs the full detector instead (only new tags then wait for the background).
  // Env override: APRILTAG_VK_TRACKING_ASYNC_DROP=0|1
  bool async_drop_lost = true;
};

// Runs the full detector on keyframes and, between them, re-detects the tags of the previous frame on
// the CPU alone, from corners predicted by constant velocity (refine, then decode); a tag that does not
// decode is searched for in a padded region, and if any tag is still missing the frame goes to the full
// detector. Detections have the same type as TagDecoder::Decode's. Not thread-safe: call Detect() from
// one thread (the asynchronous mode runs its own background thread).
class TrackingDetector {
 public:
  enum class Path { kFull, kTracked, kSearch, kCapture };

  // `td` must already have its tag families added and refine_edges set; it is not owned. `config`
  // is for the full detector; `decimation` of the tracked frames' refinement matches it.
  TrackingDetector(vk::Context &ctx, apriltag_detector_t *td, const DetectorConfig &config,
                   const TrackingConfig &tracking = {});
  ~TrackingDetector();

  TrackingDetector(const TrackingDetector &) = delete;
  TrackingDetector &operator=(const TrackingDetector &) = delete;

  // Detects the tags of one grayscale frame; the buffer must hold the configured width*height bytes (the
  // size is not checked). Returns detections owned by this object, valid until the next Detect().
  zarray_t *Detect(const uint8_t *gray_frame);

  // Forgets every track, so the next frame runs the full detector.
  void Reset();
  void SetEnabled(bool enabled);
  bool enabled() const { return tracking_.enabled; }
  // Switches the asynchronous mode; forgets every track.
  void SetAsync(bool async);
  bool async() const { return tracking_.async; }
  // The live configuration; call Reset() after changing it.
  TrackingConfig &tracking() { return tracking_; }
  // Blocks until the background detector is idle; its result is kept for the next Detect().
  void WaitBackground();

  Path last_path() const { return last_path_; }
  GpuDetector &gpu() { return *gpu_; }
  const DetectorConfig &config() const { return config_; }

  struct Stats {
    uint64_t frames_full = 0;
    uint64_t frames_tracked = 0;
    uint64_t frames_search = 0;
    uint64_t tracked_attempts_failed = 0;  // tracked frames that fell through to the full detector
    uint64_t frames_capture = 0;           // tracked frames that needed the wider capture
    uint64_t background_runs = 0;
    uint64_t tentative_accepted = 0;
    uint64_t tentative_rejected = 0;
    uint64_t tracks_dropped = 0;  // asynchronous mode: tracks that failed and left the output
    double copy_ms = 0.0;         // total time copying frames for the background detector
  };
  const Stats &stats() const { return stats_; }

 private:
  struct Track {
    int id = 0;
    double p[4][2] = {};
    double v[4][2] = {};  // per frame
    double a[4][2] = {};  // change of v per frame
    float margin = 0.0f;
    bool has_v = false;
    uint32_t span = 1;  // frames since p was measured
    bool tentative = false;  // from the background detector, not yet measured on a frame
  };

  static double Speed(const Track &t);
  zarray_t *DetectFull(const uint8_t *gray_frame);
  zarray_t *DetectAsync(const uint8_t *gray_frame);
  // Tracks every entry of tracks_ on the frame. Not partial: all or nullptr. Partial: the tracks that
  // succeed (nullptr if none).
  zarray_t *TrackFrame(const uint8_t *gray_frame, bool partial, Path *path);
  zarray_t *Solve(const uint8_t *gray_frame, std::vector<size_t> active, const std::vector<DetectedQuad> &start,
                  const std::vector<char> &strict, std::vector<DetectedQuad> *cur, std::vector<char> *ok);
  void PredictQuad(const Track &t, DetectedQuad *q) const;
  bool Accept(const apriltag_detection_t *det, const Track &t, const DetectedQuad &from,
              bool strict) const;
  bool SearchQuad(const uint8_t *gray_frame, const Track &t, DetectedQuad *out);
  bool CaptureStart(const uint8_t *gray_frame, const Track &t, DetectedQuad *out);
  bool CoarseStart(const uint8_t *gray_frame, const Track &t, int d, DetectedQuad *out);
  void EnsureCapture();
  void UpdateTracks(zarray_t *dets);
  void EnsureBackground();
  void StopBackground();
  void BackgroundMain();
  void StartBackground(const uint8_t *gray_frame);
  void CollectBackground();

  DetectorConfig config_;
  TrackingConfig tracking_;
  apriltag_detector_t *td_;  // not owned
  apriltag_detector_t *search_td_;
  std::unique_ptr<GpuDetector> gpu_;
  std::unique_ptr<QuadDecode> quad_decode_;
  std::unique_ptr<TagDecoder> full_decoder_;
  std::vector<Track> tracks_;
  std::vector<uint8_t> crop_;
  apriltag_detector_t *cap_td_ = nullptr;
  std::unique_ptr<TagDecoder> cap_decoder_;
  std::vector<uint8_t> cap_img_;
  uint64_t frame_index_ = 0;
  uint64_t last_full_index_ = 0;
  uint64_t last_bg_start_ = 0;

  std::unique_ptr<TagDecoder> bg_decoder_;
  std::thread bg_thread_;
  std::mutex bg_mutex_;
  std::condition_variable bg_job_cv_;
  std::condition_variable bg_idle_cv_;
  bool bg_job_ = false;      // a frame is waiting for the thread
  bool bg_running_ = false;  // from hand-off until the result is stored
  bool bg_stop_ = false;
  bool bg_borrowed_ = false;  // the job reads the caller's buffer
  const uint8_t *bg_frame_ = nullptr;
  std::vector<uint8_t> bg_copy_;
  uint64_t bg_frame_index_ = 0;
  bool bg_ready_ = false;
  uint64_t bg_result_index_ = 0;
  std::vector<Track> bg_result_;
  uint32_t tracked_run_ = 0;
  uint32_t skip_ = 0;
  uint32_t backoff_ = 1;
  Path last_path_ = Path::kFull;
  Stats stats_;
};

}  // namespace apriltag_vulkan
