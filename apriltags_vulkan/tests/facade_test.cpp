// The public Detector: every mode, pose, strided input, a shared device and the errors.
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "eval_common.h"
#include "test_check.h"
#include "vkapriltag/Detector.h"

namespace av = apriltag_vulkan;

namespace {

std::string g_corpus;

std::vector<eval::Image> Frames(const char *scenario) {
  std::vector<std::filesystem::path> files;
  for (const auto &e : std::filesystem::directory_iterator(std::filesystem::path(g_corpus) / scenario)) {
    if (e.path().extension() == ".png") files.push_back(e.path());
  }
  std::sort(files.begin(), files.end());
  std::vector<eval::Image> frames;
  for (const auto &f : files) frames.push_back(eval::LoadGrey(f.string()));
  return frames;
}

av::DetectorOptions Options(const eval::Image &im) {
  av::DetectorOptions o;
  o.width = static_cast<uint32_t>(im.width);
  o.height = static_cast<uint32_t>(im.height);
  return o;
}

std::multiset<int> Ids(const av::Result &r) {
  std::multiset<int> ids;
  for (const av::Detection &d : r.detections) ids.insert(d.id);
  return ids;
}

bool Throws(av::ErrorCode code, const std::function<void()> &f) {
  try {
    f();
  } catch (const av::Error &e) {
    return e.code() == code;
  }
  return false;
}

void TestSync() {
  const auto frames = Frames("one-tag");
  av::Detector detector(Options(frames[0]));
  const av::Result &r = detector.Detect(frames[0].grey.data());
  CHECK(r.ready);
  CHECK(Ids(r) == std::multiset<int>({316}));
  CHECK(r.detections[0].family != nullptr && std::string(r.detections[0].family->name) == "tag36h11");
  CHECK(r.timing.total_ms > 0.0 && r.timing.gpu_ms > 0.0 && r.timing.candidates >= 1);
  CHECK(!detector.Describe().empty());
}

void TestPipelined(uint32_t depth) {
  const auto frames = Frames("many-tags");
  av::DetectorOptions sync_options = Options(frames[0]);
  av::Detector sync(sync_options);
  std::vector<std::multiset<int>> expected;
  for (const auto &f : frames) expected.push_back(Ids(sync.Detect(f.grey.data())));

  av::DetectorOptions o = Options(frames[0]);
  o.mode = av::Mode::kPipelined;
  o.pipeline_depth = depth;
  av::Detector detector(o);
  std::vector<std::multiset<int>> got;
  for (const auto &f : frames) {
    const av::Result &r = detector.Detect(f.grey.data());
    if (r.ready) got.push_back(Ids(r));
  }
  const av::Result &last = detector.Flush();
  if (last.ready) got.push_back(Ids(last));
  CHECK(got.size() >= frames.size() - depth);
  for (size_t i = 0; i < got.size() && i < expected.size(); ++i) {
    if (got[i] != expected[i]) {
      std::printf("pipelined depth %u result %zu: ", depth, i);
      for (int id : got[i]) std::printf("%d ", id);
      std::printf("expected: ");
      for (int id : expected[i]) std::printf("%d ", id);
      std::printf("\n");
    }
    CHECK(got[i] == expected[i]);
  }
}

void TestTracking(bool async) {
  const auto frames = Frames("one-tag");
  av::DetectorOptions o = Options(frames[0]);
  o.mode = av::Mode::kTracking;
  o.tracking_async = async;
  av::Detector detector(o);
  int found = 0;
  for (int round = 0; round < 3; ++round) {
    for (const auto &f : frames) found += Ids(detector.Detect(f.grey.data())).count(316) > 0 ? 1 : 0;
  }
  CHECK(found >= 3);
  detector.Reset();
}

void TestLevels() {
  const auto frames = Frames("one-tag");
  av::DetectorOptions o = Options(frames[0]);
  av::Level a, b;
  a.decimation = 2;
  b.decimation = 1;
  o.levels = {a, b};
  av::Detector detector(o);
  CHECK(Ids(detector.Detect(frames[0].grey.data())).count(316) == 1);
}

void TestPose() {
  const auto frames = Frames("one-tag");
  av::DetectorOptions o = Options(frames[0]);
  o.intrinsics.fx = o.intrinsics.fy = 600.0;
  o.intrinsics.cx = 320.0;
  o.intrinsics.cy = 240.0;
  o.tag_size = 0.1;
  av::Detector detector(o);
  const av::Result &r = detector.Detect(frames[0].grey.data());
  CHECK(r.detections.size() == 1 && r.poses.size() == 1);
  CHECK(r.poses[0].valid && r.poses[0].t[2] > 0.0);
}

void TestStride() {
  const auto frames = Frames("one-tag");
  const eval::Image &im = frames[0];
  const size_t stride = static_cast<size_t>(im.width) + 24;
  std::vector<uint8_t> padded(stride * im.height, 0xAB);
  for (int y = 0; y < im.height; ++y) {
    std::memcpy(padded.data() + y * stride, im.grey.data() + static_cast<size_t>(y) * im.width, im.width);
  }
  av::Detector detector(Options(im));
  const std::multiset<int> a = Ids(detector.Detect(im.grey.data()));
  const std::multiset<int> b = Ids(detector.Detect(padded.data(), stride));
  CHECK(a == b && !a.empty());
}

void TestSharedDevice() {
  const auto frames = Frames("one-tag");
  av::DeviceOptions d;
  d.expected_streams = 2;
  auto device = av::Device::Create(d);
  av::DetectorOptions o0 = Options(frames[0]), o1 = Options(frames[0]);
  o1.stream = 1;
  av::Detector first(o0, device), second(o1, device);
  CHECK(Ids(first.Detect(frames[0].grey.data())).count(316) == 1);
  CHECK(Ids(second.Detect(frames[0].grey.data())).count(316) == 1);
}

// One mode of the threaded test: how a detector is configured and what its results are compared with.
struct ThreadMode {
  const char *name;
  std::function<void(av::DetectorOptions &)> configure;
  bool tracking;   // results are checked for the tag of the one-tag scenario, not frame by frame
  bool pipelined;  // results arrive in order and lag the calls
  int calls;
};

// Detectors on one shared device, each driven by its own thread, with another made and destroyed meanwhile.
void RunThreadMode(const ThreadMode &mode) {
  auto frames = Frames(mode.tracking ? "one-tag" : "many-tags");
  frames.resize(mode.tracking ? 1 : std::min<size_t>(frames.size(), 3));
  av::DeviceOptions d;
  d.expected_streams = 2;
  auto device = av::Device::Create(d);
  av::DetectorOptions base = Options(frames[0]);
  mode.configure(base);
  std::vector<std::multiset<int>> expected;
  {
    av::DetectorOptions ro = base;
    ro.mode = av::Mode::kSync;
    av::Detector reference(ro);
    for (const eval::Image &im : frames) expected.push_back(Ids(reference.Detect(im.grey.data())));
  }
  const std::vector<uint32_t> streams = {0, 0, 1, 3};
  std::vector<std::unique_ptr<av::Detector>> owned;
  for (uint32_t s : streams) {
    av::DetectorOptions o = base;
    o.stream = s;
    owned.push_back(std::make_unique<av::Detector>(o, device));
  }
  std::atomic<bool> go{false};
  std::atomic<int> bad{0}, errors{0}, hits{0}, calls{0};
  const auto drive = [&](av::Detector &det, int n) {
    size_t ready = 0;
    const auto check = [&](const av::Result &r) {
      if (!r.ready) return;
      if (mode.tracking) {
        const std::multiset<int> ids = Ids(r);
        if (ids.count(316) == 1) ++hits;
        if (ids.size() > 1) ++bad;
      } else if (Ids(r) != expected[ready++ % frames.size()]) {
        ++bad;
      }
    };
    for (int i = 0; i < n; ++i) {
      check(det.Detect(frames[static_cast<size_t>(i) % frames.size()].grey.data()));
      ++calls;
    }
    if (mode.pipelined) check(det.Flush());
  };
  std::vector<std::thread> threads;
  for (auto &det : owned) {
    threads.emplace_back([&, p = det.get()] {
      while (!go) std::this_thread::yield();
      try {
        drive(*p, mode.calls);
      } catch (const std::exception &) {
        ++errors;
      }
    });
  }
  threads.emplace_back([&] {
    while (!go) std::this_thread::yield();
    try {
      for (int i = 0; i < 6; ++i) {
        av::DetectorOptions o = base;
        o.stream = static_cast<uint32_t>(i % 3);
        av::Detector extra(o, device);
        drive(extra, 3);
      }
    } catch (const std::exception &) {
      ++errors;
    }
  });
  go = true;
  for (std::thread &t : threads) t.join();
  if (bad != 0 || errors != 0) std::printf("threads %s: %d wrong, %d errors\n", mode.name, bad.load(), errors.load());
  CHECK(bad == 0);
  CHECK(errors == 0);
  if (mode.tracking) {
    CHECK(hits * 2 >= calls);
  }
}

void TestSharedDeviceThreads(const std::string &only) {
  const std::vector<ThreadMode> modes = {
      {"sync", [](av::DetectorOptions &) {}, false, false, 300},
      {"pipelined1", [](av::DetectorOptions &o) { o.mode = av::Mode::kPipelined; o.pipeline_depth = 1; }, false, true, 200},
      {"pipelined2", [](av::DetectorOptions &o) { o.mode = av::Mode::kPipelined; o.pipeline_depth = 2; }, false, true, 200},
      {"tracking", [](av::DetectorOptions &o) { o.mode = av::Mode::kTracking; }, true, false, 200},
      {"trackingasync", [](av::DetectorOptions &o) { o.mode = av::Mode::kTracking; o.tracking_async = true; }, true, false, 200},
      {"levels",
       [](av::DetectorOptions &o) {
         av::Level a, b;
         a.decimation = 2;
         b.decimation = 1;
         o.levels = {a, b};
       },
       false, false, 100},
  };
  for (const ThreadMode &m : modes) {
    if (only.empty() || only == m.name) RunThreadMode(m);
  }
}

void TestErrors() {
  const auto frames = Frames("one-tag");
  const av::DetectorOptions base = Options(frames[0]);
  CHECK(Throws(av::ErrorCode::kInvalidArgument, [&] { av::DetectorOptions o = base; o.width = 0; av::Detector d(o); }));
  CHECK(Throws(av::ErrorCode::kInvalidArgument, [&] { av::DetectorOptions o = base; o.families = {"tagNope"}; av::Detector d(o); }));
  CHECK(Throws(av::ErrorCode::kInvalidArgument, [&] { av::DetectorOptions o = base; o.families = {}; av::Detector d(o); }));
  CHECK(Throws(av::ErrorCode::kInvalidArgument, [&] {
    av::DetectorOptions o = base;
    o.mode = av::Mode::kPipelined;
    o.pipeline_depth = 3;
    av::Detector d(o);
  }));
  CHECK(Throws(av::ErrorCode::kUnsupported, [&] {
    av::DetectorOptions o = base;
    o.mode = av::Mode::kTracking;
    o.levels = {av::Level()};
    av::Detector d(o);
  }));
  CHECK(Throws(av::ErrorCode::kUnsupported, [&] {
    av::DetectorOptions o = base;
    o.families = {"tag36h11", "tag16h5"};
    av::Detector d(o);
  }));
  CHECK(Throws(av::ErrorCode::kInvalidArgument, [&] {
    av::Detector d(base);
    d.Detect(nullptr);
  }));
  CHECK(Throws(av::ErrorCode::kInvalidArgument, [&] { av::DetectorOptions o = base; o.probe_margin = -1.0f; av::Detector d(o); }));
  CHECK(Throws(av::ErrorCode::kInvalidArgument, [&] {
    av::DetectorOptions o = base;
    o.probe_margin = std::numeric_limits<float>::quiet_NaN();
    av::Detector d(o);
  }));
  CHECK(Throws(av::ErrorCode::kInvalidArgument, [&] {
    av::DetectorOptions o = base;
    o.probe_margin = std::numeric_limits<float>::infinity();
    av::Detector d(o);
  }));
  CHECK(Throws(av::ErrorCode::kInvalidArgument, [&] { av::DetectorOptions o = base; o.min_white_black_diff = 256; av::Detector d(o); }));
  // A cluster minimum below 2 is raised, not rejected.
  CHECK(!Throws(av::ErrorCode::kInvalidArgument, [&] { av::DetectorOptions o = base; o.min_cluster_pixels = 0; av::Detector d(o); }));
  // The frame size must divide by the decimation.
  CHECK(Throws(av::ErrorCode::kRuntime, [&] { av::DetectorOptions o = base; o.width = 642; o.decimation = 4; av::Detector d(o); }));
}

// The candidate filters, on grey-margin frame 0: ids 551 (largest tag), 134 and 340 (smallest).
std::multiset<int> FilteredIds(const eval::Image &im, const std::function<void(av::DetectorOptions &)> &set,
                               uint32_t *candidates = nullptr) {
  av::DetectorOptions o = Options(im);
  set(o);
  av::Detector detector(o);
  const av::Result &r = detector.Detect(im.grey.data());
  if (candidates != nullptr) *candidates = r.timing.candidates;
  return Ids(r);
}

void TestFilters() {
  const auto frames = Frames("grey-margin");
  const eval::Image &im = frames[0];
  const std::multiset<int> all = {551, 134, 340};
  const std::multiset<int> big = {551};
  CHECK(FilteredIds(im, [](av::DetectorOptions &) {}) == all);

  // Untouched and explicitly default options agree.
  CHECK(FilteredIds(im, [](av::DetectorOptions &o) {
          o.min_white_black_diff = 5;
          o.min_cluster_pixels = 24;
          o.min_tag_pixels = 0;
          o.probe_margin = 0.0f;
        }) == all);

  CHECK(FilteredIds(im, [](av::DetectorOptions &o) { o.min_white_black_diff = 255; }).empty());
  CHECK(FilteredIds(im, [](av::DetectorOptions &o) { o.min_tag_pixels = 400; }) == big);
  CHECK(FilteredIds(im, [](av::DetectorOptions &o) { o.min_cluster_pixels = 100000; }).empty());
  CHECK(FilteredIds(im, [](av::DetectorOptions &o) { o.min_cluster_pixels = 800; }) == big);
  // Below 2 is raised to 2, so the run still detects.
  CHECK(FilteredIds(im, [](av::DetectorOptions &o) { o.min_cluster_pixels = 0; }).count(551) == 1);

  // The margin gate drops a candidate before refinement, so every quad is still handed to the decoder but none
  // decodes; it needs edge refinement and, as here, one family.
  uint32_t plain = 0, gated = 0;
  CHECK(FilteredIds(im, [](av::DetectorOptions &) {}, &plain) == all);
  CHECK(FilteredIds(im, [](av::DetectorOptions &o) { o.probe_margin = 1.0e6f; }, &gated).empty());
  CHECK(plain >= 3 && gated == plain);
  CHECK(FilteredIds(im, [](av::DetectorOptions &o) { o.probe_margin = 1.0e6f; o.refine_edges = false; }) ==
        FilteredIds(im, [](av::DetectorOptions &o) { o.refine_edges = false; }));
  CHECK(FilteredIds(im, [](av::DetectorOptions &o) { o.probe_margin = 1.0f; }) == all);

  // The other modes take the same fields.
  for (uint32_t depth : {1u, 2u}) {
    av::DetectorOptions o = Options(im);
    o.mode = av::Mode::kPipelined;
    o.pipeline_depth = depth;
    o.min_white_black_diff = 255;
    av::Detector detector(o);
    detector.Detect(im.grey.data());
    detector.Detect(im.grey.data());
    CHECK(detector.Flush().detections.empty());
  }
  for (bool async : {false, true}) {
    av::DetectorOptions o = Options(im);
    o.mode = av::Mode::kTracking;
    o.tracking_async = async;
    o.probe_margin = 1.0e6f;
    av::Detector detector(o);
    CHECK(detector.Detect(im.grey.data()).detections.empty());
    o.probe_margin = 0.0f;
    o.min_tag_pixels = 400;
    av::Detector sized(o);
    const av::Result &r = sized.Detect(im.grey.data());
    CHECK(Ids(r).count(134) == 0 && Ids(r).count(340) == 0);
  }
}

// A level's own min_tag_pixels wins for that level; the detector-wide value fills the levels that have none.
void TestFilterLevels() {
  const auto frames = Frames("grey-margin");
  const eval::Image &im = frames[0];
  av::Level a, b;
  a.decimation = 2;
  b.decimation = 1;
  const auto run = [&](const av::Level &second) {
    return FilteredIds(im, [&](av::DetectorOptions &o) {
      o.min_tag_pixels = 400;
      o.levels = {a, second};
    });
  };
  CHECK(run(b) == std::multiset<int>({551}));
  av::Level own = b;
  own.min_tag_pixels = 40;
  CHECK(run(own) == std::multiset<int>({551, 134, 340}));
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::printf("usage: facade_test CORPUS_DIR [group]\n");
    return 2;
  }
  g_corpus = argv[1];
  const std::string which = argc > 2 ? argv[2] : "all";
  const bool all = which == "all";
  try {
    if (all || which == "sync") TestSync();
    if (all || which == "pipelined") TestPipelined(1), TestPipelined(2);
    if (all || which == "tracking") TestTracking(false), TestTracking(true);
    if (all || which == "levels") TestLevels();
    if (all || which == "pose") TestPose();
    if (all || which == "stride") TestStride();
    if (all || which == "device") TestSharedDevice();
    if (all || which.rfind("threads", 0) == 0) TestSharedDeviceThreads(which.size() > 8 ? which.substr(8) : "");
    if (all || which == "filters") TestFilters(), TestFilterLevels();
    if (all || which == "errors") TestErrors();
  } catch (const std::exception &e) {
    std::printf("FAIL: unexpected exception: %s\n", e.what());
    return 1;
  }
  if (testcheck::Failures() != 0) {
    std::printf("%d check(s) failed\n", testcheck::Failures());
    return 1;
  }
  std::printf("ok\n");
  return 0;
}
