#include "vkapriltag/TrackingDetector.h"

#include "vkapriltag/common/Options.h"

#include "Geometry.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace apriltag_vulkan {
namespace {

double Side(const double p[4][2]) { return geom::MeanSide(p); }

void ApplyEnv(TrackingConfig *c) {
  const Options env = Options::FromEnvironment();
  c->enabled = env.Bool("APRILTAG_VK_TRACKING", c->enabled);
  if (env.Has("APRILTAG_VK_TRACKING_KEYFRAME")) {
    c->keyframe_interval = static_cast<uint32_t>(std::max(1, env.Int("APRILTAG_VK_TRACKING_KEYFRAME", 1)));
  }
  c->min_margin = static_cast<float>(env.Float("APRILTAG_VK_TRACKING_MIN_MARGIN", c->min_margin));
  c->converge_px = static_cast<float>(env.Float("APRILTAG_VK_TRACKING_CONVERGE", c->converge_px));
  if (env.Has("APRILTAG_VK_TRACKING_ITERATIONS")) {
    c->max_iterations = static_cast<uint32_t>(std::max(1, env.Int("APRILTAG_VK_TRACKING_ITERATIONS", 1)));
  }
  c->async = env.Bool("APRILTAG_VK_TRACKING_ASYNC", c->async);
  if (env.Has("APRILTAG_VK_TRACKING_ASYNC_INTERVAL")) {
    c->async_interval = static_cast<uint32_t>(std::max(1, env.Int("APRILTAG_VK_TRACKING_ASYNC_INTERVAL", 1)));
  }
  if (env.Has("APRILTAG_VK_TRACKING_ASYNC_THREADS")) {
    c->async_threads = static_cast<uint32_t>(std::max(0, env.Int("APRILTAG_VK_TRACKING_ASYNC_THREADS", 0)));
  }
  c->async_copy = env.Bool("APRILTAG_VK_TRACKING_ASYNC_COPY", c->async_copy);
  c->async_drop_lost = env.Bool("APRILTAG_VK_TRACKING_ASYNC_DROP", c->async_drop_lost);
}

// Detection corners run opposite to quad corners (quad p[k] = detection p[3 - k]).
void QuadFromDetection(const Detection &det, DetectedQuad *q) {
  for (int k = 0; k < 4; ++k) {
    q->p[3 - k][0] = det.p[k][0];
    q->p[3 - k][1] = det.p[k][1];
  }
}

double MaxShift(const Detection &det, const DetectedQuad &q) {
  double shift = 0.0;
  for (int k = 0; k < 4; ++k) {
    shift = std::max(shift, std::hypot(det.p[k][0] - q.p[3 - k][0], det.p[k][1] - q.p[3 - k][1]));
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

TrackingDetector::TrackingDetector(vk::Context &ctx, const TagDecoderConfig &decoder, const DetectorConfig &config,
                                   const TrackingConfig &tracking)
    : config_(config), tracking_(tracking), decoder_config_(decoder) {
  ApplyEnv(&tracking_);
  gpu_ = std::make_unique<GpuDetector>(ctx, config_);
  pool_ = MakeWorkerPool(config_);
  quad_decode_ = std::make_unique<QuadDecode>(config_, pool_);
  full_decoder_ = std::make_unique<TagDecoder>(decoder_config_, config_.decimation, config_.cpu_threads, RefineEdgesMethod::kProfile,
                                              0.0f, config_.PoolOptions(), false, config_.small_tag_rescue, pool_);
}

TrackingDetector::~TrackingDetector() {
  StopBackground();
  coarse_decoder_.reset();
  bg_decoder_.reset();
  full_decoder_.reset();
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

const DetectionList &TrackingDetector::Detect(const uint8_t *gray_frame) {
  ++frame_index_;
  gpu_->SetAutoPrewarm(!tracking_.enabled);
  if (tracking_.enabled && tracking_.async) return *DetectAsync(gray_frame);
  if (skip_ > 0) --skip_;
  if (tracking_.enabled && !tracks_.empty() && skip_ == 0 && tracked_run_ + 1 < tracking_.keyframe_interval) {
    Path path = Path::kTracked;
    if (const DetectionList *dets = TrackFrame(gray_frame, false, &path)) {
      ++tracked_run_;
      gpu_->NoteFrame();
      if (tracked_run_ + 1 >= tracking_.keyframe_interval) gpu_->ExpectGpuFrame();
      backoff_ = 1;
      last_path_ = path;
      ++stats_.frames_tracked;
      return *dets;
    }
    ++stats_.tracked_attempts_failed;
    skip_ = backoff_ + 1;
    backoff_ = std::min(backoff_ * 2, tracking_.keyframe_interval);
  }
  const DetectionList *dets = DetectFull(gray_frame);
  last_path_ = Path::kFull;
  ++stats_.frames_full;
  if (tracking_.enabled) {
    UpdateTracks(*dets);
  } else {
    tracks_.clear();
  }
  tracked_run_ = 0;
  return *dets;
}

const DetectionList *TrackingDetector::DetectAsync(const uint8_t *gray_frame) {
  EnsureBackground();
  {
    std::unique_lock<std::mutex> lock(bg_mutex_);
    bg_idle_cv_.wait(lock, [&] { return !(bg_running_ && bg_borrowed_); });
  }
  CollectBackground();
  if (!tracks_.empty()) {
    Path path = Path::kTracked;
    if (const DetectionList *dets = TrackFrame(gray_frame, true, &path)) {
      last_path_ = path;
      ++stats_.frames_tracked;
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
  const DetectionList *dets = DetectFull(gray_frame);
  last_full_index_ = frame_index_;
  last_path_ = Path::kFull;
  ++stats_.frames_full;
  UpdateTracks(*dets);
  return dets;
}

const DetectionList *TrackingDetector::DetectFull(const uint8_t *gray_frame) {
  gpu_->Detect(gray_frame);
  const std::vector<DetectedQuad> quads = quad_decode_->Decode(gpu_->last_line_fit_points);
  return &full_decoder_->Decode(quads, gray_frame, config_.width, config_.height, config_.reversed_border);
}

void TrackingDetector::EnsureBackground() {
  if (bg_thread_.joinable()) return;
  // The background thread runs quad_decode_ while the caller tracks with full_decoder_: separate pools.
  quad_decode_ = std::make_unique<QuadDecode>(config_);
  bg_decoder_ = std::make_unique<TagDecoder>(decoder_config_, config_.decimation, tracking_.async_threads, RefineEdgesMethod::kProfile,
                                          0.0f, config_.PoolOptions(), true, config_.small_tag_rescue);
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
      const DetectionList &dets = bg_decoder_->Decode(quads, frame, config_.width, config_.height, config_.reversed_border);
      for (const Detection &det : dets) {
        Track t;
        t.id = det.id;
        t.margin = det.decision_margin;
        for (int k = 0; k < 4; ++k) {
          t.p[k][0] = det.p[k][0];
          t.p[k][1] = det.p[k][1];
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

void TrackingDetector::UpdateTracks(const DetectionList &dets) {
  std::vector<Track> next;
  for (const Detection &det : dets) {
    Track t;
    t.id = det.id;
    t.margin = det.decision_margin;
    for (int k = 0; k < 4; ++k) {
      t.p[k][0] = det.p[k][0];
      t.p[k][1] = det.p[k][1];
    }
    const auto old = std::find_if(tracks_.begin(), tracks_.end(),
                                  [&](const Track &o) { return o.id == t.id; });
    if (old != tracks_.end()) {
      const double span = std::max<uint32_t>(old->span, 1);
      for (int k = 0; k < 4; ++k) {
        for (int c = 0; c < 2; ++c) t.v[k][c] = (t.p[k][c] - old->p[k][c]) / span;
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

bool TrackingDetector::Accept(const Detection &det, const Track &t, const DetectedQuad &from) const {
  if (det.id != t.id) return false;
  if (det.decision_margin < tracking_.min_margin ||
      det.decision_margin < tracking_.min_margin_ratio * t.margin) {
    return false;
  }
  const double side = Side(t.p);
  const double ratio = Side(det.p) / side;
  if (ratio > tracking_.max_scale_change || ratio * tracking_.max_scale_change < 1.0) return false;
  double cx, cy;
  geom::QuadCentre(from.p, &cx, &cy);
  return std::hypot(det.c[0] - cx, det.c[1] - cy) <= tracking_.max_jump_frac * side;
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
  const double pad = std::max(16.0, tracking_.coarse_pad_frac * side) + 4.0 * d;
  const int rx0 = std::max(0, static_cast<int>(std::floor(x0 - pad)));
  const int ry0 = std::max(0, static_cast<int>(std::floor(y0 - pad)));
  const int rx1 = std::min(w, static_cast<int>(std::ceil(x1 + pad)));
  const int ry1 = std::min(h, static_cast<int>(std::ceil(y1 + pad)));
  const int ow = (rx1 - rx0) / d, oh = (ry1 - ry0) / d;
  if (ow < 16 || oh < 16) return false;
  BoxDecimate(gray_frame, w, rx0, ry0, ow * d, oh * d, d, &coarse_img_);
  const double centre = 0.5 * (d - 1);
  DetectedQuad q;
  for (int k = 0; k < 4; ++k) {
    q.p[k][0] = (pred.p[k][0] - rx0 - centre) / d;
    q.p[k][1] = (pred.p[k][1] - ry0 - centre) / d;
  }
  bool settled = false;
  for (int iter = 0; iter < 8 && !settled; ++iter) {
    const DetectionList &dets = coarse_decoder_->Decode({q}, coarse_img_.data(), static_cast<uint32_t>(ow),
                                                        static_cast<uint32_t>(oh), config_.reversed_border);
    if (dets.size() != 1) return false;
    const Detection &det = dets[0];
    if (det.id != t.id || det.decision_margin < 0.5f * tracking_.min_margin) return false;
    settled = MaxShift(det, q) <= tracking_.coarse_converge_px;
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

void TrackingDetector::EnsureCoarse() {
  if (coarse_decoder_) return;
  coarse_decoder_ = std::make_unique<TagDecoder>(decoder_config_, 2, 1, RefineEdgesMethod::kProfile, 0.0f,
                                                 WorkerPoolOptions(), false, config_.small_tag_rescue);
}

// Decodes from cur[i] for i in `active`, feeding each tag's refined corners back until they stop moving.
// ok[i] is set when track i decodes as itself and has settled; the last decode's detections come back.
const DetectionList *TrackingDetector::Solve(const uint8_t *gray_frame, std::vector<size_t> active,
                                              const std::vector<DetectedQuad> &start, std::vector<DetectedQuad> *cur,
                                              std::vector<char> *ok) {
  const DetectionList *dets = nullptr;
  for (size_t i : active) (*ok)[i] = 0;
  for (uint32_t iter = 0; iter < tracking_.max_iterations && !active.empty(); ++iter) {
    std::vector<DetectedQuad> quads;
    for (size_t i : active) quads.push_back((*cur)[i]);
    dets = &full_decoder_->Decode(quads, gray_frame, config_.width, config_.height, config_.reversed_border);
    std::vector<size_t> next;
    bool moving = false;
    for (size_t i : active) {
      const Detection *hit = nullptr;
      int hits = 0;
      for (const Detection &det : *dets) {
        if (det.id == tracks_[i].id) {
          hit = &det;
          ++hits;
        }
      }
      (*ok)[i] = 0;
      if (hits != 1 || !Accept(*hit, tracks_[i], start[i])) continue;
      const double shift = MaxShift(*hit, (*cur)[i]);
      QuadFromDetection(*hit, &(*cur)[i]);
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

const DetectionList *TrackingDetector::TrackFrame(const uint8_t *gray_frame, bool partial, Path *path) {
  const size_t n = tracks_.size();
  const double w = config_.width, h = config_.height;
  std::vector<DetectedQuad> start(n), cur;
  std::vector<char> ok(n, 0);
  std::vector<size_t> active;
  for (size_t i = 0; i < n; ++i) {
    PredictQuad(tracks_[i], &start[i]);
    bool edge = false;
    for (int k = 0; k < 4; ++k) {
      edge = edge || start[i].p[k][0] < 2.0 || start[i].p[k][1] < 2.0 || start[i].p[k][0] > w - 3.0 ||
             start[i].p[k][1] > h - 3.0;
    }
    if (edge && partial && tracks_[i].tentative) continue;
    if (edge) return nullptr;
    active.push_back(i);
  }
  if (active.empty()) return nullptr;
  // A track from the background detector is not trusted to a refine from its old corners: it starts from
  // a decimated refine.
  for (size_t i : std::vector<size_t>(active)) {
    if (!tracks_[i].tentative) continue;
    EnsureCoarse();
    const int d0 = std::clamp(static_cast<int>(std::ceil(Speed(tracks_[i]) / 8.0)), 2, 8);
    DetectedQuad q;
    if (CoarseStart(gray_frame, tracks_[i], d0, &q) || (d0 < 8 && CoarseStart(gray_frame, tracks_[i], 2 * d0, &q))) {
      start[i] = q;
    } else {
      active.erase(std::find(active.begin(), active.end(), i));
    }
  }
  if (active.empty()) return nullptr;
  cur = start;
  const DetectionList *solved = Solve(gray_frame, active, start, &cur, &ok);

  std::vector<size_t> accepted;
  for (size_t i : active) {
    if (ok[i]) accepted.push_back(i);
  }
  if (accepted.empty() || (!partial && accepted.size() != n)) return nullptr;
  if (partial && !tracking_.async_drop_lost) {
    for (size_t i = 0; i < n; ++i) {
      if (!tracks_[i].tentative && std::find(accepted.begin(), accepted.end(), i) == accepted.end()) return nullptr;
    }
  }
  // Only accepted tracks are reported: the other detections are left out.
  tracked_out_.clear();
  for (const Detection &det : *solved) {
    if (std::any_of(accepted.begin(), accepted.end(), [&](size_t i) { return tracks_[i].id == det.id; })) {
      tracked_out_.push_back(det);
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
  UpdateTracks(tracked_out_);
  *path = Path::kTracked;
  return &tracked_out_;
}

}  // namespace apriltag_vulkan
