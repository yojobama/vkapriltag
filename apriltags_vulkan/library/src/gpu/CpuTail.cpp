#include "vkapriltag/gpu/CpuTail.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace apriltag_vulkan {

namespace {

constexpr uint32_t kNone = 0xFFFFFFFFu;
constexpr uint32_t kMaxThetaKey = (1u << 20) - 1u;
constexpr uint32_t kInitialTable = 256u;

uint32_t HashPair(uint32_t k0, uint32_t k1) {
  uint32_t h = k0 * 0x9E3779B1u + k1 * 0x85EBCA6Bu;
  h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
  return h;
}

void InsertionSort(uint64_t *keys, uint32_t count) {
  for (uint32_t i = 1; i < count; ++i) {
    const uint64_t v = keys[i];
    uint32_t j = i;
    while (j > 0 && keys[j - 1] > v) {
      keys[j] = keys[j - 1];
      --j;
    }
    keys[j] = v;
  }
}

// Orders (angle key << 32 | point index) words; returns the sorted words (in keys or in tmp).
const uint64_t *SortKeys(uint64_t *keys, uint32_t count, std::vector<uint64_t> &tmp, std::vector<uint32_t> &hist) {
  if (count <= 24) {
    InsertionSort(keys, count);
    return keys;
  }
  if (count > 4096) {
    std::sort(keys, keys + count);
    return keys;
  }
  // Bucket by angle, then insertion-sort the nearly ordered result.
  uint32_t buckets = 32;
  while (buckets < count) buckets *= 2;
  if (hist.size() < buckets) hist.resize(buckets);
  if (tmp.size() < count) tmp.resize(count);
  std::fill(hist.begin(), hist.begin() + buckets, 0u);
  for (uint32_t i = 0; i < count; ++i) ++hist[((keys[i] >> 32) * buckets) >> 20];
  uint32_t sum = 0;
  for (uint32_t b = 0; b < buckets; ++b) {
    const uint32_t c = hist[b];
    hist[b] = sum;
    sum += c;
  }
  for (uint32_t i = 0; i < count; ++i) tmp[hist[((keys[i] >> 32) * buckets) >> 20]++] = keys[i];
  InsertionSort(tmp.data(), count);
  return tmp.data();
}

}  // namespace

CpuTail::CpuTail(const CpuTailConfig &config) : config_(config), table_(kInitialTable) {}

void CpuTail::GrowTable() {
  std::vector<Slot> bigger(table_.size() * 2);
  const uint32_t mask = static_cast<uint32_t>(bigger.size()) - 1u;
  for (const Slot &s : table_) {
    if (s.id1 == 0u) continue;
    uint32_t h = HashPair(s.k0, s.k1) & mask;
    while (bigger[h].id1 != 0u) h = (h + 1u) & mask;
    bigger[h] = s;
  }
  table_.swap(bigger);
}

int32_t CpuTail::DecimatedAt(int32_t x, int32_t y) const {
  const int32_t k = static_cast<int32_t>(config_.decimation);
  const int32_t w = static_cast<int32_t>(config_.width);
  if (k == 1) return gray_[y * w + x];
  if (!config_.box_decimate) return gray_[y * k * w + x * k];
  if (k == 2) {
    const uint8_t *row = gray_ + 2 * y * w + 2 * x;
    return (row[0] + row[1] + row[w] + row[w + 1] + 2) >> 2;
  }
  int32_t sum = 0;
  for (int32_t yy = 0; yy < k; ++yy) {
    const uint8_t *row = gray_ + (y * k + yy) * w + x * k;
    for (int32_t xx = 0; xx < k; ++xx) sum += row[xx];
  }
  return (sum + k * k / 2) / (k * k);
}

RawLineFitPoint CpuTail::LineFitPoint(uint32_t packed, uint32_t blob) const {
  const int32_t ix2 = static_cast<int32_t>(packed & 0x3FFFu) + 1;
  const int32_t iy2 = static_cast<int32_t>((packed >> 14) & 0x3FFFu) + 1;
  const int32_t ix = ix2 / 2;
  const int32_t iy = iy2 / 2;
  const int32_t dw = static_cast<int32_t>(config_.width / config_.decimation);
  const int32_t dh = static_cast<int32_t>(config_.height / config_.decimation);
  int32_t weight = 1;
  if (ix > 0 && ix + 1 < dw && iy > 0 && iy + 1 < dh) {
    const int32_t gx = DecimatedAt(ix + 1, iy) - DecimatedAt(ix - 1, iy);
    const int32_t gy = DecimatedAt(ix, iy + 1) - DecimatedAt(ix, iy - 1);
    const float v = static_cast<float>(gx * gx + gy * gy);
    const float r = config_.sqrt_via_rsqrt ? 1.0f / (1.0f / std::sqrt(v)) : std::sqrt(v);
    weight = static_cast<int32_t>(r) + 1;
  }
  RawLineFitPoint out;
  out.xy2 = (static_cast<uint32_t>(ix2) & 0x3FFFu) | ((static_cast<uint32_t>(iy2) & 0x3FFFu) << 14);
  out.w_blob = (static_cast<uint32_t>(weight) & 0x3FFu) | (blob << 10);
  return out;
}

void CpuTail::Run(const uint32_t *compacted, const uint32_t *keys, uint32_t count, const uint8_t *gray) {
  gray_ = gray;
  raw_blobs_ = 0;
  qualifying_ = 0;
  oversized_ = 0;
  num_points_ = 0;
  selected_.clear();
  if (count == 0) return;

  // Group by (rep0, rep1) and accumulate the extents.
  std::fill(table_.begin(), table_.end(), Slot{});
  raw_.clear();
  if (point_blob_.size() < count) point_blob_.resize(count);
  uint32_t last_k0 = 0, last_k1 = 0, last_id = kNone;
  bool have_last = false;
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t k0 = keys[2 * i];
    const uint32_t k1 = keys[2 * i + 1];
    uint32_t id;
    if (have_last && k0 == last_k0 && k1 == last_k1) {
      id = last_id;
    } else {
      uint32_t mask = static_cast<uint32_t>(table_.size()) - 1u;
      uint32_t h = HashPair(k0, k1) & mask;
      for (;;) {
        Slot &s = table_[h];
        if (s.id1 == 0u) {
          if ((raw_blobs_ + 1u) * 2u > table_.size()) {
            GrowTable();
            mask = static_cast<uint32_t>(table_.size()) - 1u;
            h = HashPair(k0, k1) & mask;
            continue;
          }
          s.k0 = k0;
          s.k1 = k1;
          s.id1 = ++raw_blobs_;
          if (s.id1 - 1u < config_.max_raw_blobs) {
            MinMaxExtentsGpu e;
            e.min_x = e.min_y = 2147483647;
            e.max_x = e.max_y = -2147483647;
            raw_.push_back(e);
          }
          id = s.id1 - 1u;
          break;
        }
        if (s.k0 == k0 && s.k1 == k1) {
          id = s.id1 - 1u;
          break;
        }
        h = (h + 1u) & mask;
      }
      last_k0 = k0;
      last_k1 = k1;
      last_id = id;
      have_last = true;
    }
    if (id >= config_.max_raw_blobs) {
      point_blob_[i] = kNone;
      continue;
    }
    point_blob_[i] = id;
    const uint32_t packed = compacted[i];
    const int32_t x = static_cast<int32_t>(packed & 0x3FFFu);
    const int32_t y = static_cast<int32_t>((packed >> 14) & 0x3FFFu);
    const int32_t gx = static_cast<int32_t>((packed >> 28) & 3u) - 1;
    const int32_t gy = static_cast<int32_t>((packed >> 30) & 3u) - 1;
    MinMaxExtentsGpu &e = raw_[id];
    e.min_x = std::min(e.min_x, x);
    e.min_y = std::min(e.min_y, y);
    e.max_x = std::max(e.max_x, x);
    e.max_y = std::max(e.max_y, y);
    e.count += 1u;
    e.gx_sum = static_cast<int32_t>(static_cast<uint32_t>(e.gx_sum) + static_cast<uint32_t>(gx));
    e.gy_sum = static_cast<int32_t>(static_cast<uint32_t>(e.gy_sum) + static_cast<uint32_t>(gy));
    e.pxgx_plus_pygy_sum = static_cast<int32_t>(static_cast<uint32_t>(e.pxgx_plus_pygy_sum) +
                                                static_cast<uint32_t>(x * gx + y * gy));
  }

  // Select the blobs that can be a tag border, in order of first appearance.
  const uint32_t raw_n = std::min<uint32_t>(raw_blobs_, config_.max_raw_blobs);
  remap_.assign(raw_n, kNone);
  offset_.clear();
  uint32_t total = 0;
  for (uint32_t i = 0; i < raw_n; ++i) {
    const MinMaxExtentsGpu &e = raw_[i];
    if (e.count < config_.min_cluster_pixels || e.count > config_.max_cluster_pixels) continue;
    const uint32_t bbox_w = static_cast<uint32_t>(e.max_x - e.min_x);
    const uint32_t bbox_h = static_cast<uint32_t>(e.max_y - e.min_y);
    if (bbox_w == 0u || bbox_h == 0u) continue;
    if (config_.aspect_max > 0.0f) {
      const float long_side = static_cast<float>(std::max(bbox_w, bbox_h));
      const float short_side = static_cast<float>(std::min(bbox_w, bbox_h));
      if (long_side > short_side * config_.aspect_max) continue;
    }
    if (config_.fill_min > 0.0f || config_.fill_max > 0.0f) {
      const float perimeter = 2.0f * static_cast<float>(bbox_w + bbox_h);
      const float fill = static_cast<float>(e.count) / perimeter;
      if (config_.fill_min > 0.0f && fill < config_.fill_min) continue;
      if (config_.fill_max > 0.0f && fill > config_.fill_max) continue;
    }
    const uint32_t sum2 = static_cast<uint32_t>(e.pxgx_plus_pygy_sum) * 2u -
                          static_cast<uint32_t>(e.min_x + e.max_x) * static_cast<uint32_t>(e.gx_sum) -
                          static_cast<uint32_t>(e.min_y + e.max_y) * static_cast<uint32_t>(e.gy_sum);
    const float two_dot = static_cast<float>(static_cast<int32_t>(sum2)) - 0.10236f * static_cast<float>(e.gx_sum) +
                          0.057162f * static_cast<float>(e.gy_sum);
    const bool reversed = two_dot < 0.0f;
    if (!config_.reversed_border && reversed) continue;
    if (!config_.normal_border && !reversed) continue;
    const uint32_t pos = qualifying_++;
    if (pos >= config_.max_blobs) continue;
    remap_[i] = pos;
    selected_.push_back(e);
    total += e.count;
    offset_.push_back(total - e.count);
  }
  if (selected_.empty()) return;

  // Place each point of a selected blob at its rank within the blob, keyed by its angle.
  cursor_.assign(selected_.size(), 0u);
  if (sort_keys_.size() < total) sort_keys_.resize(total);
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t id = point_blob_[i];
    if (id == kNone || id >= raw_n) continue;
    const uint32_t sel = remap_[id];
    if (sel == kNone) continue;
    const MinMaxExtentsGpu &e = selected_[sel];
    const uint32_t packed = compacted[i];
    const float cx = (static_cast<float>(e.min_x) + static_cast<float>(e.max_x)) * 0.5f + 0.05118f;
    const float cy = (static_cast<float>(e.min_y) + static_cast<float>(e.max_y)) * 0.5f + -0.028581f;
    const float dx = static_cast<float>(packed & 0x3FFFu) - cx;
    const float dy = static_cast<float>((packed >> 14) & 0x3FFFu) - cy;
    const float d = std::fabs(dx) + std::fabs(dy);
    const float ratio = d > 0.0f ? dy / d : 0.0f;
    const float pseudo = dx >= 0.0f ? (dy >= 0.0f ? ratio : 4.0f + ratio) : 2.0f - ratio;
    const int32_t theta_int = static_cast<int32_t>(std::lrintf(pseudo * 2.5e5f));
    const uint32_t theta = std::min(static_cast<uint32_t>(std::max(theta_int, 0)), kMaxThetaKey);
    const uint32_t rank = cursor_[sel]++;
    if (rank >= e.count) continue;
    sort_keys_[offset_[sel] + rank] = (static_cast<uint64_t>(theta) << 32) | i;
  }

  // Sort each blob by angle and emit its line-fit records.
  if (out_.size() < total) out_.resize(total);
  for (uint32_t sel = 0; sel < selected_.size(); ++sel) {
    const uint32_t n = std::min(cursor_[sel], selected_[sel].count);
    uint64_t *keys_begin = sort_keys_.data() + offset_[sel];
    const uint64_t *sorted = keys_begin;
    if (n > config_.large_sort_cap) {
      ++oversized_;
    } else {
      sorted = SortKeys(keys_begin, n, sort_tmp_, histogram_);
    }
    RawLineFitPoint *out = out_.data() + offset_[sel];
    for (uint32_t j = 0; j < n; ++j) out[j] = LineFitPoint(compacted[static_cast<uint32_t>(sorted[j])], sel);
  }
  num_points_ = std::min(total, config_.point_capacity);
}

}  // namespace apriltag_vulkan
