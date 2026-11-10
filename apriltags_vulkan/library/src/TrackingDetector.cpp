#include "vkapriltag/TrackingDetector.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>

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
  if (const char *v = std::getenv("APRILTAG_VK_TRACKING_CAPTURE")) c->capture = v[0] != '0';
  if (const char *v = std::getenv("APRILTAG_VK_TRACKING_MAX_SPEED")) c->max_speed_px = static_cast<float>(std::atof(v));
  if (const char *v = std::getenv("APRILTAG_VK_TRACKING_CAPTURE_SPEED")) c->capture_speed_px = static_cast<float>(std::atof(v));
  if (const char *v = std::getenv("APRILTAG_VK_TRACKING_ASYNC")) c->async = v[0] != '0';
  if (const char *v = std::getenv("APRILTAG_VK_TRACKING_ASYNC_INTERVAL")) {
    c->async_interval = static_cast<uint32_t>(std::max(1, std::atoi(v)));
  }
  if (const char *v = std::getenv("APRILTAG_VK_TRACKING_ASYNC_THREADS")) {
    c->async_threads = static_cast<uint32_t>(std::max(0, std::atoi(v)));
  }
  if (const char *v = std::getenv("APRILTAG_VK_TRACKING_ASYNC_COPY")) c->async_copy = v[0] != '0';
  if (const char *v = std::getenv("APRILTAG_VK_TRACKING_ASYNC_DROP")) c->async_drop_lost = v[0] != '0';
}

// Detection corners run opposite to quad corners (quad p[k] = detection p[3 - k]).
void QuadFromDetection(const apriltag_detection_t *det, DetectedQuad *q) {
  for (int k = 0; k < 4; ++k) {
    q->p[3 - k][0] = det->p[k][0];
    q->p[3 - k][1] = det->p[k][1];
  }
}

double MaxShift(const apriltag_detection_t *det, const DetectedQuad &q) {
  double shift = 0.0;
  for (int k = 0; k < 4; ++k) {
    shift = std::max(shift, std::hypot(det->p[k][0] - q.p[3 - k][0], det->p[k][1] - q.p[3 - k][1]));
  }
  return shift;
}

// Averages d x d blocks of the w x h window at (x0, y0).
void BoxDecimate(const uint8_t *src, int stride, int x0, int y0, int w, int h, int d, std::vector<uint8_t> *out) {
  const int ow = w / d, oh = h / d;
  out->resize(static_cast<size_t>(ow) * oh);
  const int area = d * d;
  for (int y = 0; y < oh; ++y) {
    for (int x = 0; x < ow; ++x) {
      int sum = 0;
      for (int dy = 0; dy < d; ++dy) {
        const uint8_t *row = src + static_cast<size_t>(y0 + y * d + dy) * stride + x0 + x * d;
        for (int dx = 0; dx < d; ++dx) sum += row[dx];
      }
      (*out)[static_cast<size_t>(y) * ow + x] = static_cast<uint8_t>((sum + area / 2) / area);
    }
  }
}

}  // namespace

double TrackingDetector::Speed(const Track &t) {
  double speed = 0.0;
  for (int k = 0; k < 4; ++k) speed = std::max(speed, std::hypot(t.v[k][0], t.v[k][1]));
  return speed;
}

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
  StopBackground();
  cap_decoder_.reset();
  if (cap_td_) apriltag_detector_destroy(cap_td_);
  bg_decoder_.reset();
  full_decoder_.reset();
  apriltag_detector_destroy(search_td_);
}

void TrackingDetector::WaitBackground() {
  std::unique_lock<std::mutex> lock(bg_mutex_);
  bg_idle_cv_.wait(lock, [&] { return !bg_running_; });
}

void TrackingDetector::Reset() {
  WaitBackground();
  {
    std::lock_guard<std::mutex> lock(bg_mutex_);
    bg_ready_ = false;
  }
  tracks_.clear();
  tracked_run_ = 0;
  skip_ = 0;
  backoff_ = 1;
}

void TrackingDetector::SetEnabled(bool enabled) {
  tracking_.enabled = enabled;
  Reset();
}

void TrackingDetector::SetAsync(bool async) {
  tracking_.async = async;
  Reset();
}

zarray_t *TrackingDetector::Detect(const uint8_t *gray_frame) {
  ++frame_index_;
  if (tracking_.enabled && tracking_.async) return DetectAsync(gray_frame);
  if (skip_ > 0) --skip_;
  if (tracking_.enabled && !tracks_.empty() && skip_ == 0 && tracked_run_ + 1 < tracking_.keyframe_interval) {
    Path path = Path::kTracked;
    if (zarray_t *dets = TrackFrame(gray_frame, false, &path)) {
      ++tracked_run_;
      backoff_ = 1;
      last_path_ = path;
      ++(path == Path::kSearch ? stats_.frames_search : path == Path::kCapture ? stats_.frames_capture : stats_.frames_tracked);
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

zarray_t *TrackingDetector::DetectAsync(const uint8_t *gray_frame) {
  EnsureBackground();
  {
    std::unique_lock<std::mutex> lock(bg_mutex_);
    bg_idle_cv_.wait(lock, [&] { return !(bg_running_ && bg_borrowed_); });
  }
  CollectBackground();
  if (!tracks_.empty()) {
    Path path = Path::kTracked;
    if (zarray_t *dets = TrackFrame(gray_frame, true, &path)) {
      last_path_ = path;
      ++(path == Path::kCapture ? stats_.frames_capture : stats_.frames_tracked);
      if (frame_index_ - last_bg_start_ >= tracking_.async_interval && frame_index_ - last_full_index_ > 2) {
        StartBackground(gray_frame);
      }
      return dets;
    }
    ++stats_.tracked_attempts_failed;
  }
  WaitBackground();
  {
    std::lock_guard<std::mutex> lock(bg_mutex_);
    bg_ready_ = false;
  }
  zarray_t *dets = DetectFull(gray_frame);
  last_full_index_ = frame_index_;
  last_path_ = Path::kFull;
  ++stats_.frames_full;
  UpdateTracks(dets);
  return dets;
}

zarray_t *TrackingDetector::DetectFull(const uint8_t *gray_frame) {
  gpu_->Detect(gray_frame);
  const std::vector<DetectedQuad> quads = quad_decode_->Decode(gpu_->last_line_fit_points);
  return full_decoder_->Decode(quads, gray_frame, config_.width, config_.height,
                               config_.reversed_border);
}

void TrackingDetector::EnsureBackground() {
  if (bg_thread_.joinable()) return;
  bg_decoder_ = std::make_unique<TagDecoder>(td_, config_.decimation, tracking_.async_threads);
  bg_stop_ = false;
  bg_thread_ = std::thread([this] { BackgroundMain(); });
}

void TrackingDetector::StopBackground() {
  if (!bg_thread_.joinable()) return;
  {
    std::lock_guard<std::mutex> lock(bg_mutex_);
    bg_stop_ = true;
  }
  bg_job_cv_.notify_all();
  bg_thread_.join();
}

void TrackingDetector::StartBackground(const uint8_t *gray_frame) {
  {
    std::lock_guard<std::mutex> lock(bg_mutex_);
    if (bg_running_) return;
  }
  const uint8_t *frame = gray_frame;
  if (tracking_.async_copy) {
    const auto t0 = std::chrono::steady_clock::now();
    bg_copy_.resize(static_cast<size_t>(config_.width) * config_.height);
    std::memcpy(bg_copy_.data(), gray_frame, bg_copy_.size());
    frame = bg_copy_.data();
    stats_.copy_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
  {
    std::lock_guard<std::mutex> lock(bg_mutex_);
    bg_frame_ = frame;
    bg_borrowed_ = !tracking_.async_copy;
    bg_frame_index_ = frame_index_;
    bg_job_ = true;
    bg_running_ = true;
  }
  last_bg_start_ = frame_index_;
  ++stats_.background_runs;
  bg_job_cv_.notify_one();
}

void TrackingDetector::BackgroundMain() {
  std::unique_lock<std::mutex> lock(bg_mutex_);
  for (;;) {
    bg_job_cv_.wait(lock, [&] { return bg_stop_ || bg_job_; });
    if (bg_stop_) return;
    bg_job_ = false;
    const uint8_t *frame = bg_frame_;
    const uint64_t index = bg_frame_index_;
    lock.unlock();
    std::vector<Track> result;
    try {
      gpu_->Detect(frame);
      const std::vector<DetectedQuad> quads = quad_decode_->Decode(gpu_->last_line_fit_points);
      zarray_t *dets = bg_decoder_->Decode(quads, frame, config_.width, config_.height, config_.reversed_border);
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
        result.push_back(t);
      }
    } catch (...) {
      result.clear();
    }
    lock.lock();
    bg_result_ = std::move(result);
    bg_result_index_ = index;
    bg_ready_ = true;
    bg_running_ = false;
    bg_idle_cv_.notify_all();
  }
}

void TrackingDetector::CollectBackground() {
  std::vector<Track> result;
  uint64_t index;
  {
    std::lock_guard<std::mutex> lock(bg_mutex_);
    if (!bg_ready_) return;
    result = std::move(bg_result_);
    index = bg_result_index_;
    bg_ready_ = false;
  }
  if (index <= last_full_index_) return;
  for (size_t a = 0; a < result.size(); ++a) {
    for (size_t b = a + 1; b < result.size(); ++b) {
      if (result[a].id == result[b].id) return;
    }
  }
  for (Track &t : result) {
    const bool known = std::any_of(tracks_.begin(), tracks_.end(), [&](const Track &o) { return o.id == t.id; });
    if (known) continue;
    t.span = static_cast<uint32_t>(frame_index_ - index);
    t.tentative = true;
    tracks_.push_back(t);
  }
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
      const double span = std::max<uint32_t>(old->span, 1);
      for (int k = 0; k < 4; ++k) {
        for (int c = 0; c < 2; ++c) {
          t.v[k][c] = (t.p[k][c] - old->p[k][c]) / span;
          t.a[k][c] = old->has_v ? t.v[k][c] - old->v[k][c] : 0.0;
        }
      }
      t.has_v = true;
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

// Refines one quad on a box-decimated crop (by d) until it settles; the corners come back in frame pixels.
bool TrackingDetector::CoarseStart(const uint8_t *gray_frame, const Track &t, int d, DetectedQuad *out) {
  const int w = static_cast<int>(config_.width), h = static_cast<int>(config_.height);
  const double side = Side(t.p);
  if (side / d < 16.0) return false;
  DetectedQuad pred;
  PredictQuad(t, &pred);
  double x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  for (int k = 0; k < 4; ++k) {
    const double *both[2] = {t.p[k], pred.p[k]};
    for (const double *p : both) {
      x0 = std::min(x0, p[0]);
      x1 = std::max(x1, p[0]);
      y0 = std::min(y0, p[1]);
      y1 = std::max(y1, p[1]);
    }
  }
  const double pad = std::max(16.0, tracking_.capture_pad_frac * side) + 4.0 * d;
  const int rx0 = std::max(0, static_cast<int>(std::floor(x0 - pad)));
  const int ry0 = std::max(0, static_cast<int>(std::floor(y0 - pad)));
  const int rx1 = std::min(w, static_cast<int>(std::ceil(x1 + pad)));
  const int ry1 = std::min(h, static_cast<int>(std::ceil(y1 + pad)));
  const int ow = (rx1 - rx0) / d, oh = (ry1 - ry0) / d;
  if (ow < 16 || oh < 16) return false;
  BoxDecimate(gray_frame, w, rx0, ry0, ow * d, oh * d, d, &cap_img_);
  const double centre = 0.5 * (d - 1);
  DetectedQuad q;
  for (int k = 0; k < 4; ++k) {
    q.p[k][0] = (pred.p[k][0] - rx0 - centre) / d;
    q.p[k][1] = (pred.p[k][1] - ry0 - centre) / d;
  }
  cap_td_->quad_decimate = 2.0f;
  bool settled = false;
  for (int iter = 0; iter < 8 && !settled; ++iter) {
    zarray_t *dets = cap_decoder_->Decode({q}, cap_img_.data(), static_cast<uint32_t>(ow),
                                          static_cast<uint32_t>(oh), config_.reversed_border);
    if (zarray_size(dets) != 1) return false;
    apriltag_detection_t *det = nullptr;
    zarray_get(dets, 0, &det);
    if (det->id != t.id || det->decision_margin < 0.5f * tracking_.min_margin) return false;
    settled = MaxShift(det, q) <= tracking_.capture_converge_px;
    QuadFromDetection(det, &q);
  }
  if (!settled) return false;
  for (int k = 0; k < 4; ++k) {
    if (q.p[k][0] < 2.0 || q.p[k][1] < 2.0 || q.p[k][0] > ow - 3.0 || q.p[k][1] > oh - 3.0) return false;
    out->p[k][0] = q.p[k][0] * d + rx0 + centre;
    out->p[k][1] = q.p[k][1] * d + ry0 + centre;
  }
  const double ratio = Side(out->p) / side;
  return ratio <= tracking_.max_scale_change && ratio * tracking_.max_scale_change >= 1.0;
}

void TrackingDetector::EnsureCapture() {
  if (cap_td_) return;
  cap_td_ = apriltag_detector_create();
  for (int i = 0; i < zarray_size(td_->tag_families); ++i) {
    apriltag_family_t *family = nullptr;
    zarray_get(td_->tag_families, i, &family);
    apriltag_detector_add_family(cap_td_, family);
  }
  cap_td_->nthreads = 1;
  cap_td_->refine_edges = td_->refine_edges;
  cap_td_->quad_sigma = td_->quad_sigma;
  cap_td_->decode_sharpening = td_->decode_sharpening;
  cap_decoder_ = std::make_unique<TagDecoder>(cap_td_, 2, 1);
}

bool TrackingDetector::CaptureStart(const uint8_t *gray_frame, const Track &t, DetectedQuad *out) {
  // Other motion hypotheses at full resolution: keep the one whose first refine moves least.
  DetectedQuad starts[2];
  int count = 0;
  if (t.has_v) {
    for (int k = 0; k < 4; ++k) {
      starts[count].p[3 - k][0] = t.p[k][0] + t.v[k][0] + t.a[k][0];
      starts[count].p[3 - k][1] = t.p[k][1] + t.v[k][1] + t.a[k][1];
    }
    ++count;
  }
  for (int k = 0; k < 4; ++k) {
    starts[count].p[3 - k][0] = t.p[k][0];
    starts[count].p[3 - k][1] = t.p[k][1];
  }
  ++count;
  double best = 1e30;
  for (int s = 0; s < count; ++s) {
    zarray_t *dets = full_decoder_->Decode({starts[s]}, gray_frame, config_.width, config_.height,
                                           config_.reversed_border);
    if (zarray_size(dets) != 1) continue;
    apriltag_detection_t *det = nullptr;
    zarray_get(dets, 0, &det);
    if (!Accept(det, t, starts[s], false)) continue;
    const double shift = MaxShift(det, starts[s]);
    if (shift < best) {
      best = shift;
      QuadFromDetection(det, out);
    }
  }
  if (best <= tracking_.max_first_shift_px) return true;

  EnsureCapture();
  const double speed = std::max(4.0, Speed(t));
  const int d0 = std::clamp(static_cast<int>(std::ceil(speed / 8.0)), 2, 8);
  for (int d : {d0, 2 * d0}) {
    if (d <= 8 && CoarseStart(gray_frame, t, d, out)) return true;
  }
  return false;
}

// Decodes from cur[i] for i in `active`, feeding each tag's refined corners back until they stop moving.
// ok[i] is set when track i decodes as itself and has settled; the last decode's detections come back.
zarray_t *TrackingDetector::Solve(const uint8_t *gray_frame, std::vector<size_t> active,
                                  const std::vector<DetectedQuad> &start, const std::vector<char> &strict,
                                  std::vector<DetectedQuad> *cur, std::vector<char> *ok) {
  zarray_t *dets = nullptr;
  for (size_t i : active) (*ok)[i] = 0;
  for (uint32_t iter = 0; iter < tracking_.max_iterations && !active.empty(); ++iter) {
    std::vector<DetectedQuad> quads;
    for (size_t i : active) quads.push_back((*cur)[i]);
    dets = full_decoder_->Decode(quads, gray_frame, config_.width, config_.height, config_.reversed_border);
    std::vector<size_t> next;
    bool moving = false;
    for (size_t i : active) {
      const apriltag_detection_t *hit = nullptr;
      int hits = 0;
      for (int d = 0; d < zarray_size(dets); ++d) {
        apriltag_detection_t *det = nullptr;
        zarray_get(dets, d, &det);
        if (det->id == tracks_[i].id) {
          hit = det;
          ++hits;
        }
      }
      (*ok)[i] = 0;
      if (hits != 1 || !Accept(hit, tracks_[i], start[i], strict[i] != 0)) continue;
      const double shift = MaxShift(hit, (*cur)[i]);
      QuadFromDetection(hit, &(*cur)[i]);
      if (shift > tracking_.converge_px) {
        if (iter == 0 && shift > tracking_.max_first_shift_px) continue;
        moving = true;
      } else {
        (*ok)[i] = 1;
      }
      next.push_back(i);
    }
    active = std::move(next);
    if (!moving) break;
  }
  return dets;
}

zarray_t *TrackingDetector::TrackFrame(const uint8_t *gray_frame, bool partial, Path *path) {
  const size_t n = tracks_.size();
  const double w = config_.width, h = config_.height;
  bool captured = false, searched = false, rescued = false;
  std::vector<DetectedQuad> start(n), cur;
  std::vector<char> alive(n, 1), ok(n, 0), strict(n, 1);
  std::vector<size_t> active;
  for (size_t i = 0; i < n; ++i) {
    PredictQuad(tracks_[i], &start[i]);
    bool edge = false;
    for (int k = 0; k < 4; ++k) {
      edge = edge || start[i].p[k][0] < 2.0 || start[i].p[k][1] < 2.0 || start[i].p[k][0] > w - 3.0 ||
             start[i].p[k][1] > h - 3.0;
    }
    if (edge && partial && tracks_[i].tentative) {
      alive[i] = 0;
    } else if (edge) {
      return nullptr;
    } else if (tracking_.max_speed_px > 0.0f && !tracks_[i].tentative && Speed(tracks_[i]) > tracking_.max_speed_px) {
      return nullptr;
    } else {
      active.push_back(i);
    }
  }
  if (active.empty()) return nullptr;
  // A track from the background detector is not trusted to a refine from its old corners: it starts from
  // a decimated refine; so does a fast track when capture_speed_px is set.
  for (size_t i : std::vector<size_t>(active)) {
    const double speed = Speed(tracks_[i]);
    const bool fast = tracking_.capture && tracking_.capture_speed_px > 0.0f && speed > tracking_.capture_speed_px;
    if (!tracks_[i].tentative && !fast) continue;
    EnsureCapture();
    const int d0 = std::clamp(static_cast<int>(std::ceil(speed / 8.0)), 2, 8);
    DetectedQuad q;
    if (CoarseStart(gray_frame, tracks_[i], d0, &q) || (d0 < 8 && CoarseStart(gray_frame, tracks_[i], 2 * d0, &q))) {
      start[i] = q;
      captured = captured || fast;
    } else if (tracks_[i].tentative) {
      active.erase(std::find(active.begin(), active.end(), i));
    }
  }
  if (active.empty()) return nullptr;
  cur = start;
  zarray_t *dets = Solve(gray_frame, active, start, strict, &cur, &ok);

  std::vector<size_t> failed;
  for (size_t i : active) {
    if (!ok[i]) failed.push_back(i);
  }
  if (!failed.empty() && (tracking_.capture || tracking_.roi_search)) {
    std::vector<DetectedQuad> start2 = start;
    std::vector<size_t> second;
    for (size_t i : failed) {
      DetectedQuad q;
      if (tracking_.capture && CaptureStart(gray_frame, tracks_[i], &q)) {
        captured = true;
      } else if (tracking_.roi_search && SearchQuad(gray_frame, tracks_[i], &q)) {
        searched = true;
      } else {
        continue;
      }
      start2[i] = q;
      cur[i] = q;
      strict[i] = 0;
      second.push_back(i);
    }
    if (!second.empty()) {
      dets = Solve(gray_frame, second, start2, strict, &cur, &ok);
      rescued = true;
    }
  }

  std::vector<size_t> accepted;
  for (size_t i : active) {
    if (ok[i]) accepted.push_back(i);
  }
  if (accepted.empty() || (!partial && accepted.size() != n)) return nullptr;
  if (rescued) {
    std::vector<DetectedQuad> quads;
    for (size_t i : accepted) quads.push_back(cur[i]);
    dets = full_decoder_->Decode(quads, gray_frame, config_.width, config_.height, config_.reversed_border);
    std::vector<size_t> kept;
    for (size_t i : accepted) {
      int hits = 0;
      const apriltag_detection_t *hit = nullptr;
      for (int d = 0; d < zarray_size(dets); ++d) {
        apriltag_detection_t *det = nullptr;
        zarray_get(dets, d, &det);
        if (det->id == tracks_[i].id) {
          hit = det;
          ++hits;
        }
      }
      if (hits == 1 && Accept(hit, tracks_[i], cur[i], false)) kept.push_back(i);
    }
    if (kept.empty() || (!partial && kept.size() != n)) return nullptr;
    accepted = std::move(kept);
  }
  if (partial && !tracking_.async_drop_lost) {
    for (size_t i = 0; i < n; ++i) {
      if (!tracks_[i].tentative && std::find(accepted.begin(), accepted.end(), i) == accepted.end()) return nullptr;
    }
  }
  // Only accepted tracks are reported: drop the other detections (the decoder owns the rest).
  for (int d = zarray_size(dets) - 1; d >= 0; --d) {
    apriltag_detection_t *det = nullptr;
    zarray_get(dets, d, &det);
    const bool keep = std::any_of(accepted.begin(), accepted.end(), [&](size_t i) { return tracks_[i].id == det->id; });
    if (!keep) {
      apriltag_detection_destroy(det);
      zarray_remove_index(dets, d, 0);
    }
  }
  for (size_t i = 0; i < n; ++i) {
    if (!tracks_[i].tentative) continue;
    const bool in = std::find(accepted.begin(), accepted.end(), i) != accepted.end();
    ++(in ? stats_.tentative_accepted : stats_.tentative_rejected);
  }
  for (size_t i = 0; i < n; ++i) {
    if (!tracks_[i].tentative && std::find(accepted.begin(), accepted.end(), i) == accepted.end()) ++stats_.tracks_dropped;
  }
  UpdateTracks(dets);
  *path = searched ? Path::kSearch : captured ? Path::kCapture : Path::kTracked;
  return dets;
}

}  // namespace apriltag_vulkan
