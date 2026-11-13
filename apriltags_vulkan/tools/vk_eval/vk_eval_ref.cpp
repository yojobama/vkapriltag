#include "vk_eval_ref.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <thread>

#include "eval_common.h"
#include "upstream/UpstreamAdapter.h"

namespace {

std::vector<eval::Det> ToDets(const zarray_t *z) {
  std::vector<eval::Det> dets;
  for (const apriltag_vulkan::Detection &d : upstream::ToDetections(z)) {
    eval::Det e{d.id, d.hamming, d.decision_margin, {d.c[0], d.c[1]}, {}};
    for (int k = 0; k < 4; ++k) {
      e.p[k][0] = d.p[k][0];
      e.p[k][1] = d.p[k][1];
    }
    dets.push_back(e);
  }
  return dets;
}

}  // namespace

void RunReference(const std::vector<std::string> &paths, const std::string &family, uint32_t decimation,
                  uint32_t threads, int warmup, int iterations, std::ofstream *out, std::vector<double> *all_total) {
  apriltag_family_t *tf = upstream::CreateFamily(family.c_str());
  if (tf == nullptr) throw std::runtime_error("unknown tag family " + family);
  apriltag_detector_t *td = apriltag_detector_create();
  apriltag_detector_add_family(td, tf);
  td->quad_decimate = float(decimation);
  td->nthreads = int(threads ? threads : std::max(1u, std::thread::hardware_concurrency()));
  std::printf("libapriltag CPU, quad_decimate %u, %d threads\n", decimation, td->nthreads);
  for (size_t n = 0; n < paths.size(); n++) {
    eval::Image im = eval::LoadGrey(paths[n]);
    image_u8_t u8{im.width, im.height, im.width, im.grey.data()};
    eval::Timings t;
    zarray_t *dets = nullptr;
    for (int it = 0; it < warmup + iterations; it++) {
      if (dets) apriltag_detections_destroy(dets);
      if (const char *g = std::getenv("VK_EVAL_GAP_MS")) std::this_thread::sleep_for(std::chrono::microseconds(int(std::atof(g) * 1000)));
      const auto t0 = eval::Clock::now();
      dets = apriltag_detector_detect(td, &u8);
      const auto t1 = eval::Clock::now();
      if (it >= warmup) t.Add(0.0, eval::Ms(t0, t1));
    }
    if (iterations > 1) eval::PrintTimings(paths[n], t, size_t(td->nquads), size_t(zarray_size(dets)));
    all_total->push_back(eval::Median(t.total));
    if (out != nullptr) eval::WriteDets(*out, int(n), ToDets(dets));
    apriltag_detections_destroy(dets);
  }
  apriltag_detector_destroy(td);
  upstream::DestroyFamily(tf, family.c_str());
}
