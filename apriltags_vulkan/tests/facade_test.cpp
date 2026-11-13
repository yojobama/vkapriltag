// The public Detector: every mode, pose, strided input, a shared device and the errors.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <set>
#include <string>
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
  for (size_t i = 0; i < got.size() && i < expected.size(); ++i) CHECK(got[i] == expected[i]);
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
  // The frame size must divide by the decimation.
  CHECK(Throws(av::ErrorCode::kRuntime, [&] { av::DetectorOptions o = base; o.width = 642; o.decimation = 4; av::Detector d(o); }));
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
