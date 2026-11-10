#include "vkapriltag/TrackingDetector.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace apriltag_vulkan {
namespace {

double Side(const double p[4][2]) {
  double sum = 0.0;
  for (int k = 0; k < 4; ++k) {
    sum += std::hypot(p[k][0] - p[(k + 1) % 4][0], p[k][1] - p[(k + 1) % 4][1]);
  }
  return sum * 0.25;
}

void ApplyEnv(TrackingConfig *c) {
  if (const char *v = std::getenv("APRILTAG_VK_TRACKING")) c->enabled = v[0] != '0';
  if (const char *v = std::getenv("APRILTAG_VK_TRACKING_KEYFRAME")) {
    c->keyframe_interval = static_cast<uint32_t>(std::max(1, std::atoi(v)));
  }
  if (const char *v = std::getenv("APRILTAG_VK_TRACKING_MIN_MARGIN")) {
    c->min_margin = static_cast<float>(std::atof(v));
  }
  if (const char *v = std::getenv("APRILTAG_VK_TRACKING_CONVERGE")) c->converge_px = static_cast<float>(std::atof(v));
  if (const char *v = std::getenv("APRILTAG_VK_TRACKING_ITERATIONS")) {
    c->max_iterations = static_cast<uint32_t>(std::max(1, std::atoi(v)));
  }
  if (const char *v = std::getenv("APRILTAG_VK_TRACKING_SEARCH")) c->roi_search = v[0] != '0';
}

}  // namespace

TrackingDetector::TrackingDetector(vk::Context &ctx, apriltag_detector_t *td,
                                   const DetectorConfig &config, const TrackingConfig &tracking)
    : config_(config), tracking_(tracking), td_(td) {
  ApplyEnv(&tracking_);
  gpu_ = std::make_unique<GpuDetector>(ctx, config_);
  quad_decode_ = std::make_unique<QuadDecode>(config_);
  full_decoder_ = std::make_unique<TagDecoder>(td_, config_.decimation, config_.cpu_threads);
  search_td_ = apriltag_detector_create();
  for (int i = 0; i < zarray_size(td_->tag_families); ++i) {
    apriltag_family_t *family = nullptr;
    zarray_get(td_->tag_families, i, &family);
    apriltag_detector_add_family(search_td_, family);
  }
  search_td_->nthreads = 1;
  search_td_->quad_decimate = 1.0f;
  search_td_->refine_edges = false;
  search_td_->qtp.min_white_black_diff = static_cast<int>(config_.min_white_black_diff);
}

TrackingDetector::~TrackingDetector() {
  full_decoder_.reset();
  apriltag_detector_destroy(search_td_);
}

void TrackingDetector::Reset() {
  tracks_.clear();
  tracked_run_ = 0;
  skip_ = 0;
  backoff_ = 1;
}

void TrackingDetector::SetEnabled(bool enabled) {
  tracking_.enabled = enabled;
  Reset();
}

zarray_t *TrackingDetector::Detect(const uint8_t *gray_frame) {
  if (skip_ > 0) --skip_;
  if (tracking_.enabled && !tracks_.empty() && skip_ == 0 && tracked_run_ + 1 < tracking_.keyframe_interval) {
    bool searched = false;
    if (zarray_t *dets = TryTrack(gray_frame, &searched)) {
      ++tracked_run_;
      backoff_ = 1;
      last_path_ = searched ? Path::kSearch : Path::kTracked;
      ++(searched ? stats_.frames_search : stats_.frames_tracked);
      return dets;
    }
    ++stats_.tracked_attempts_failed;
    skip_ = backoff_ + 1;
    backoff_ = std::min(backoff_ * 2, tracking_.keyframe_interval);
  }
  zarray_t *dets = DetectFull(gray_frame);
  last_path_ = Path::kFull;
  ++stats_.frames_full;
  if (tracking_.enabled) {
    UpdateTracks(dets);
  } else {
    tracks_.clear();
  }
  tracked_run_ = 0;
  return dets;
}

zarray_t *TrackingDetector::DetectFull(const uint8_t *gray_frame) {
  gpu_->Detect(gray_frame);
  const std::vector<DetectedQuad> quads = quad_decode_->Decode(gpu_->last_line_fit_points);
  return full_decoder_->Decode(quads, gray_frame, config_.width, config_.height,
                               config_.reversed_border);
}

void TrackingDetector::UpdateTracks(zarray_t *dets) {
  std::vector<Track> next;
  for (int i = 0; i < zarray_size(dets); ++i) {
    apriltag_detection_t *det = nullptr;
    zarray_get(dets, i, &det);
    Track t;
    t.id = det->id;
    t.margin = det->decision_margin;
    for (int k = 0; k < 4; ++k) {
      t.p[k][0] = det->p[k][0];
      t.p[k][1] = det->p[k][1];
    }
    const auto old = std::find_if(tracks_.begin(), tracks_.end(),
                                  [&](const Track &o) { return o.id == t.id; });
    if (old != tracks_.end()) {
      for (int k = 0; k < 4; ++k) {
        t.v[k][0] = t.p[k][0] - old->p[k][0];
        t.v[k][1] = t.p[k][1] - old->p[k][1];
      }
    }
    next.push_back(t);
  }
  // The same ID twice cannot be told apart by ID, so such a scene is never tracked.
  for (size_t a = 0; a < next.size(); ++a) {
    for (size_t b = a + 1; b < next.size(); ++b) {
      if (next[a].id == next[b].id) next.clear();
    }
    if (next.empty()) break;
  }
  tracks_ = std::move(next);
}

void TrackingDetector::PredictQuad(const Track &t, DetectedQuad *q) const {
  for (int k = 0; k < 4; ++k) {
    q->p[3 - k][0] = t.p[k][0] + t.v[k][0];
    q->p[3 - k][1] = t.p[k][1] + t.v[k][1];
  }
}

bool TrackingDetector::Accept(const apriltag_detection_t *det, const Track &t,
                              const DetectedQuad &from, bool strict) const {
  if (det->id != t.id) return false;
  if (det->decision_margin < tracking_.min_margin ||
      det->decision_margin < tracking_.min_margin_ratio * t.margin) {
    return false;
  }
  const double side = Side(t.p);
  const double ratio = Side(det->p) / side;
  if (ratio > tracking_.max_scale_change || ratio * tracking_.max_scale_change < 1.0) return false;
  if (strict) {
    double cx = 0.0, cy = 0.0;
    for (int k = 0; k < 4; ++k) {
      cx += from.p[k][0] * 0.25;
      cy += from.p[k][1] * 0.25;
    }
    if (std::hypot(det->c[0] - cx, det->c[1] - cy) > tracking_.max_jump_frac * side) return false;
  }
  return true;
}

bool TrackingDetector::SearchQuad(const uint8_t *gray_frame, const Track &t, DetectedQuad *out) {
  const double w = config_.width, h = config_.height;
  double x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  DetectedQuad predicted;
  PredictQuad(t, &predicted);
  for (int k = 0; k < 4; ++k) {
    const double *both[2] = {t.p[k], predicted.p[k]};
    for (const double *p : both) {
      x0 = std::min(x0, p[0]);
      x1 = std::max(x1, p[0]);
      y0 = std::min(y0, p[1]);
      y1 = std::max(y1, p[1]);
    }
  }
  const double pad = std::max<double>(tracking_.roi_pad_min, tracking_.roi_pad_frac * Side(t.p));
  const int rx0 = std::max(0, static_cast<int>(std::floor(x0 - pad)));
  const int ry0 = std::max(0, static_cast<int>(std::floor(y0 - pad)));
  const int rx1 = std::min(static_cast<int>(w), static_cast<int>(std::ceil(x1 + pad)));
  const int ry1 = std::min(static_cast<int>(h), static_cast<int>(std::ceil(y1 + pad)));
  const int rw = rx1 - rx0, rh = ry1 - ry0;
  if (rw < 8 || rh < 8) return false;
  crop_.resize(static_cast<size_t>(rw) * rh);
  for (int y = 0; y < rh; ++y) {
    std::copy_n(gray_frame + static_cast<size_t>(ry0 + y) * config_.width + rx0, rw,
                crop_.begin() + static_cast<size_t>(y) * rw);
  }
  search_td_->quad_decimate = std::clamp(std::floor(static_cast<float>(Side(t.p)) / tracking_.search_tag_px), 1.0f, 4.0f);
  image_u8_t sub{rw, rh, rw, crop_.data()};
  zarray_t *found = apriltag_detector_detect(search_td_, &sub);
  bool ok = false;
  for (int i = 0; i < zarray_size(found) && !ok; ++i) {
    apriltag_detection_t *det = nullptr;
    zarray_get(found, i, &det);
    if (det->id != t.id) continue;
    for (int k = 0; k < 4; ++k) {
      out->p[3 - k][0] = det->p[k][0] + rx0;
      out->p[3 - k][1] = det->p[k][1] + ry0;
    }
    ok = true;
  }
  apriltag_detections_destroy(found);
  return ok;
}

zarray_t *TrackingDetector::TryTrack(const uint8_t *gray_frame, bool *searched) {
  const size_t n = tracks_.size();
  std::vector<DetectedQuad> quads(n);
  const double w = config_.width, h = config_.height;
  for (size_t i = 0; i < n; ++i) {
    PredictQuad(tracks_[i], &quads[i]);
    for (int k = 0; k < 4; ++k) {
      if (quads[i].p[k][0] < 2.0 || quads[i].p[k][1] < 2.0 || quads[i].p[k][0] > w - 3.0 ||
          quads[i].p[k][1] > h - 3.0) {
        return nullptr;
      }
    }
  }
  auto decode = [&](const std::vector<DetectedQuad> &q) {
    return full_decoder_->Decode(q, gray_frame, config_.width, config_.height,
                                  config_.reversed_border);
  };
  // The detection for track i among `dets`, or nullptr if it is missing or ambiguous.
  auto find = [&](zarray_t *dets, size_t i) -> apriltag_detection_t * {
    apriltag_detection_t *hit = nullptr;
    for (int d = 0; d < zarray_size(dets); ++d) {
      apriltag_detection_t *det = nullptr;
      zarray_get(dets, d, &det);
      if (det->id != tracks_[i].id) continue;
      if (hit) return nullptr;
      hit = det;
    }
    return hit;
  };

  // Decodes from `start`, feeding each tag's refined corners back in until they stop moving; ok[i] is
  // set when track i decodes as itself and has settled.
  auto solve = [&](const std::vector<DetectedQuad> &start, const std::vector<char> &strict,
                   std::vector<char> *ok) {
    std::vector<DetectedQuad> cur = start;
    zarray_t *dets = nullptr;
    for (uint32_t iter = 0; iter < tracking_.max_iterations; ++iter) {
      dets = decode(cur);
      ok->assign(n, 1);
      bool moving = false, lost = false;
      for (size_t i = 0; i < n; ++i) {
        const apriltag_detection_t *det = find(dets, i);
        if (!det || !Accept(det, tracks_[i], start[i], strict[i] != 0)) {
          (*ok)[i] = 0;
          lost = true;
          continue;
        }
        double shift = 0.0;
        for (int k = 0; k < 4; ++k) {
          shift = std::max(shift, std::hypot(det->p[k][0] - cur[i].p[3 - k][0],
                                             det->p[k][1] - cur[i].p[3 - k][1]));
          cur[i].p[3 - k][0] = det->p[k][0];
          cur[i].p[3 - k][1] = det->p[k][1];
        }
        if (shift > tracking_.converge_px) {
          moving = true;
          (*ok)[i] = 0;
          if (iter == 0 && shift > tracking_.max_first_shift_px) lost = true;
        }
      }
      if (lost || !moving) break;
    }
    return dets;
  };

  std::vector<char> ok;
  const std::vector<DetectedQuad> predicted = quads;
  zarray_t *dets = solve(predicted, std::vector<char>(n, 1), &ok);
  std::vector<size_t> failed;
  for (size_t i = 0; i < n; ++i) {
    if (!ok[i]) failed.push_back(i);
  }
  if (failed.empty() && static_cast<size_t>(zarray_size(dets)) == n) {
    UpdateTracks(dets);
    return dets;
  }
  if (!tracking_.roi_search || failed.empty()) return nullptr;

  // Failed tracks restart from their searched corners; the others from the same predicted corners as
  // before, which gives the same detections.
  std::vector<char> strict(n, 1);
  for (size_t i : failed) {
    if (!SearchQuad(gray_frame, tracks_[i], &quads[i])) return nullptr;
    strict[i] = 0;
  }
  dets = solve(quads, strict, &ok);
  if (static_cast<size_t>(zarray_size(dets)) != n) return nullptr;
  for (size_t i = 0; i < n; ++i) {
    if (!ok[i]) return nullptr;
  }
  *searched = true;
  UpdateTracks(dets);
  return dets;
}

}  // namespace apriltag_vulkan
