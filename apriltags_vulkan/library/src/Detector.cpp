#include "vkapriltag/Detector.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <utility>

#include "vkapriltag/FramePipeline.h"
#include "vkapriltag/PoseEstimator.h"
#include "vkapriltag/TagDecoder.h"
#include "vkapriltag/TrackingDetector.h"
#include "vkapriltag/gpu/GpuDetector.h"
#include "vkapriltag/gpu/MultiLevelDetector.h"
#include "vkapriltag/gpu/QuadDecode.h"
#include "vkapriltag/vk/Context.h"

namespace apriltag_vulkan {

namespace {

using Clock = std::chrono::steady_clock;

double Ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

// Reports every internal failure as an Error.
template <typename F>
auto Guard(F &&f) -> decltype(f()) {
  try {
    return f();
  } catch (const Error &) {
    throw;
  } catch (const std::invalid_argument &e) {
    throw Error(ErrorCode::kInvalidArgument, e.what());
  } catch (const std::exception &e) {
    throw Error(ErrorCode::kRuntime, e.what());
  }
}

[[noreturn]] void Fail(ErrorCode code, const std::string &message) { throw Error(code, message); }

}  // namespace

struct Device::Impl {
  std::unique_ptr<vk::Context> context;
};

Device::Device() : impl_(std::make_unique<Impl>()) {}
Device::~Device() = default;

std::shared_ptr<Device> Device::Create(const DeviceOptions &options) {
  return Guard([&] {
    vk::ContextOptions co;
    co.device_index = options.device_index;
    co.allow_cpu_device = options.allow_cpu_device;
    co.expected_streams = std::max(1u, options.expected_streams);
    co.verbose = options.verbose;
    std::shared_ptr<Device> device(new Device());
    device->impl_->context = std::make_unique<vk::Context>(co);
    return device;
  });
}

std::string Device::Describe() const { return impl_->context->DescribeDevice(); }

struct Detector::Impl {
  DetectorOptions options;
  std::shared_ptr<Device> device;
  std::vector<const Family *> families;
  DetectorConfig config;
  std::shared_ptr<WorkerPool> pool;
  std::unique_ptr<vk::Lane> lane, lane_second;
  std::unique_ptr<GpuDetector> gpu, gpu_second;
  std::unique_ptr<QuadDecode> quad;
  std::unique_ptr<TagDecoder> decoder;
  std::unique_ptr<FramePipeline> pipeline;
  std::unique_ptr<TrackingDetector> tracker;
  std::unique_ptr<MultiLevelDetector> multi;
  std::unique_ptr<PoseEstimator> pose;
  std::vector<uint8_t> packed;
  Result result;

  explicit Impl(const DetectorOptions &o, std::shared_ptr<Device> d) : options(o), device(std::move(d)) {
    if (options.width == 0 || options.height == 0) Fail(ErrorCode::kInvalidArgument, "width and height must be set");
    if (options.decimation == 0) Fail(ErrorCode::kInvalidArgument, "decimation must be at least 1");
    if (options.families.empty() || options.families.size() > 8) {
      Fail(ErrorCode::kInvalidArgument, "between 1 and 8 families are needed");
    }
    if (options.mode == Mode::kPipelined && options.pipeline_depth != 1 && options.pipeline_depth != 2) {
      Fail(ErrorCode::kInvalidArgument, "pipeline_depth must be 1 or 2");
    }
    if (!options.levels.empty() && options.mode != Mode::kSync) {
      Fail(ErrorCode::kUnsupported, "levels need mode kSync");
    }
    for (const std::string &name : options.families) {
      const Family *f = Family::ByName(name);
      if (f == nullptr) Fail(ErrorCode::kInvalidArgument, "unknown tag family " + name);
      families.push_back(f);
    }
    for (const Family *f : families) {
      if (f->width_at_border != families[0]->width_at_border || f->reversed_border != families[0]->reversed_border) {
        Fail(ErrorCode::kUnsupported, "families in one detector must share their border width and polarity");
      }
    }
    if (!device) device = Device::Create();
    vk::Context &ctx = device->impl().context ? *device->impl().context : throw Error(ErrorCode::kDevice, "no device");

    config.width = options.width;
    config.height = options.height;
    config.decimation = options.decimation;
    config.tag_width = static_cast<uint32_t>(families[0]->width_at_border);
    config.reversed_border = families[0]->reversed_border;
    config.normal_border = !families[0]->reversed_border;
    config.small_tag_rescue = options.small_tag_rescue;
    config.hull_quad = options.hull_quad;
    config.cpu_threads = options.cpu_threads;

    TagDecoderConfig decoder_config;
    decoder_config.families = families;
    decoder_config.refine_edges = options.refine_edges;

    vk::Lane *lane_ptr = &ctx.default_lane();
    if (options.stream > 0) {
      // A device with fewer queues than streams shares its first queue between them.
      try {
        lane = ctx.CreateLane(options.stream);
      } catch (const std::runtime_error &) {
        lane.reset();
      }
      if (lane) lane_ptr = lane.get();
    }

    pool = MakeWorkerPool(config);
    uint32_t decimation = options.decimation;
    if (!options.levels.empty()) {
      std::vector<LevelSpec> specs;
      decimation = 1;
      for (const Level &l : options.levels) {
        LevelSpec spec;
        spec.decimation = l.decimation;
        spec.box = l.box;
        spec.min_tag_pixels = l.min_tag_pixels;
        spec.max_tag_pixels = l.max_tag_pixels;
        specs.push_back(spec);
        decimation = std::max(decimation, l.decimation);
      }
      multi = std::make_unique<MultiLevelDetector>(ctx, config, specs);
    }

    if (options.mode == Mode::kTracking) {
      TrackingConfig tracking;
      tracking.enabled = true;
      tracking.async = options.tracking_async;
      tracker = std::make_unique<TrackingDetector>(ctx, decoder_config, config, tracking);
    } else {
      decoder = std::make_unique<TagDecoder>(decoder_config, decimation, options.cpu_threads, RefineEdgesMethod::kProfile,
                                             0.0f, WorkerPoolOptions(), false, options.small_tag_rescue, pool);
      if (options.polish) decoder->SetPolish(true);
      if (!multi) {
        gpu = std::make_unique<GpuDetector>(ctx, *lane_ptr, config);
        quad = std::make_unique<QuadDecode>(config, pool);
        if (options.mode == Mode::kPipelined) {
          if (options.pipeline_depth == 2) {
            // A lane is used by one thread at a time, so the second detector gets its own lane on the same queue.
            lane_second = ctx.CreateSiblingLane(*lane_ptr);
            if (!lane_second) Fail(ErrorCode::kUnsupported, "pipeline_depth 2 needs a second lane");
            gpu_second = std::make_unique<GpuDetector>(ctx, *lane_second, config);
            pipeline = std::make_unique<FramePipeline>(*gpu, *gpu_second, *quad, *decoder);
          } else {
            pipeline = std::make_unique<FramePipeline>(*gpu, *quad, *decoder);
          }
        }
      }
    }
    if (options.tag_size > 0.0) {
      pose = std::make_unique<PoseEstimator>(options.intrinsics, options.tag_size, options.cpu_threads,
                                             PoseEstimator::kDefaultConvergenceTol, pool);
    }
  }

  GpuDetector &AnyGpu() {
    if (gpu) return *gpu;
    if (tracker) return tracker->gpu();
    Fail(ErrorCode::kUnsupported, "frames cannot be imported in multi-level mode");
  }

  void Publish(const DetectionList &list, Clock::time_point start) {
    result.detections.assign(list.begin(), list.end());
    result.ready = true;
    result.poses.clear();
    const auto t_pose = Clock::now();
    if (pose && !result.detections.empty()) pose->EstimateAll(result.detections, result.poses);
    const auto end = Clock::now();
    result.timing.pose_ms = pose ? Ms(t_pose, end) : 0.0;
    result.timing.total_ms = Ms(start, end);
  }

  const Result &Run(const uint8_t *grey, size_t stride) {
    const auto start = Clock::now();
    result.timing = Timing();
    const uint8_t *frame = grey;
    if (stride != 0 && stride != options.width) {
      packed.resize(static_cast<size_t>(options.width) * options.height);
      for (uint32_t y = 0; y < options.height; ++y) {
        std::memcpy(packed.data() + static_cast<size_t>(y) * options.width, grey + static_cast<size_t>(y) * stride,
                    options.width);
      }
      frame = packed.data();
    }

    if (tracker) {
      Publish(tracker->Detect(frame), start);
    } else if (pipeline) {
      const DetectionList *list = pipeline->Push(frame, options.width, options.height, config.reversed_border);
      result.timing.gpu_ms = pipeline->last_profile().total_ms;
      result.timing.submits = pipeline->last_profile().submits;
      result.timing.candidates = static_cast<uint32_t>(pipeline->last_quads().size());
      if (list != nullptr) {
        Publish(*list, start);
      } else {
        result.ready = false;
        result.detections.clear();
        result.poses.clear();
        result.timing.total_ms = Ms(start, Clock::now());
      }
    } else if (multi) {
      const auto t0 = Clock::now();
      const std::vector<DetectedQuad> quads = multi->Detect(frame);
      const auto t1 = Clock::now();
      result.timing.gpu_ms = multi->last_gpu_ms();
      result.timing.quad_ms = Ms(t0, t1) - result.timing.gpu_ms;
      result.timing.candidates = static_cast<uint32_t>(quads.size());
      const DetectionList &list = decoder->Decode(quads, frame, options.width, options.height, config.reversed_border);
      result.timing.decode_ms = Ms(t1, Clock::now());
      Publish(list, start);
    } else {
      gpu->Detect(frame);
      const auto t1 = Clock::now();
      result.timing.gpu_ms = gpu->last_profile().total_ms;
      result.timing.submits = gpu->last_profile().submits;
      const std::vector<DetectedQuad> quads = quad->Decode(gpu->last_line_fit_points);
      const auto t2 = Clock::now();
      result.timing.quad_ms = Ms(t1, t2);
      result.timing.candidates = static_cast<uint32_t>(quads.size());
      const DetectionList &list = decoder->Decode(quads, frame, options.width, options.height, config.reversed_border);
      result.timing.decode_ms = Ms(t2, Clock::now());
      Publish(list, start);
    }
    return result;
  }
};

Detector::Detector(const DetectorOptions &options) : Detector(options, nullptr) {}

Detector::Detector(const DetectorOptions &options, std::shared_ptr<Device> device)
    : impl_(Guard([&] { return std::make_unique<Impl>(options, std::move(device)); })) {}

Detector::~Detector() = default;

const Result &Detector::Detect(const uint8_t *grey, size_t stride) {
  if (grey == nullptr) throw Error(ErrorCode::kInvalidArgument, "the frame is null");
  return Guard([&]() -> const Result & { return impl_->Run(grey, stride); });
}

const Result &Detector::Flush() {
  return Guard([&]() -> const Result & {
    if (impl_->pipeline) {
      const auto start = Clock::now();
      const DetectionList *list = impl_->pipeline->Flush();
      if (list != nullptr) {
        impl_->Publish(*list, start);
      } else {
        impl_->result.ready = false;
      }
    }
    return impl_->result;
  });
}

void Detector::ImportHostFrame(const uint8_t *ptr, size_t bytes) {
  Guard([&] {
    impl_->AnyGpu().ImportHostFrame(ptr, bytes);
    if (impl_->gpu_second) impl_->gpu_second->ImportHostFrame(ptr, bytes);
  });
}

void Detector::ImportDmaBufFrame(const uint8_t *key, int dmabuf_fd, size_t bytes, bool yuyv) {
  Guard([&] {
    impl_->AnyGpu().ImportDmaBufFrame(key, dmabuf_fd, bytes, yuyv);
    if (impl_->gpu_second) impl_->gpu_second->ImportDmaBufFrame(key, dmabuf_fd, bytes, yuyv);
  });
}

void Detector::Reset() {
  if (impl_->tracker) impl_->tracker->Reset();
}

uint32_t Detector::width() const { return impl_->options.width; }
uint32_t Detector::height() const { return impl_->options.height; }

std::string Detector::Describe() const {
  std::string s = impl_->device->Describe();
  if (impl_->gpu) s += "; " + impl_->gpu->DescribeSizing();
  return s;
}

}  // namespace apriltag_vulkan
