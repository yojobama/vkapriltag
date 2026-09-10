#include "vkapriltag/vk/PipelineCache.h"

#include "vkapriltag/vk/Context.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace apriltag_vulkan::vk {

namespace {

namespace fs = std::filesystem;

uint64_t Fnv1a(const void *data, size_t size, uint64_t seed = 14695981039346656037ull) {
  const uint8_t *p = static_cast<const uint8_t *>(data);
  uint64_t h = seed;
  for (size_t i = 0; i < size; ++i) {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  return h;
}

std::string HexU32(uint32_t v) {
  std::ostringstream os;
  os.width(8);
  os.fill('0');
  os << std::hex << v;
  return os.str();
}

std::string HexBytes(const uint8_t *data, size_t size) {
  std::ostringstream os;
  os << std::hex;
  for (size_t i = 0; i < size; ++i) {
    os.width(2);
    os.fill('0');
    os << static_cast<unsigned>(data[i]);
  }
  return os.str();
}

// Hashes the compiled .spv corpus under `shader_dir` so a shader rebuild
// lands on a different cache filename instead of feeding stale entries to
// the driver. Not a correctness requirement (the driver keys on the full
// create-info of each pipeline, so a stale entry is just a miss) - purely to
// keep the cache directory from accumulating dead files across rebuilds.
uint64_t HashShaderDir(const std::string &shader_dir) {
  std::vector<fs::path> files;
  std::error_code ec;
  for (const auto &entry : fs::directory_iterator(shader_dir, ec)) {
    if (entry.is_regular_file() && entry.path().extension() == ".spv") {
      files.push_back(entry.path());
    }
  }
  std::sort(files.begin(), files.end());

  uint64_t h = 14695981039346656037ull;
  for (const auto &path : files) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) continue;
    std::vector<char> buf((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    h = Fnv1a(buf.data(), buf.size(), h);
  }
  return h;
}

// Where cache files live, honoring APRILTAG_VK_CACHE_DIR, then platform
// conventions. Empty return disables caching (e.g. no writable home in a
// stripped-down container).
std::string CacheDirectory() {
  if (const char *override_dir = std::getenv("APRILTAG_VK_CACHE_DIR")) {
    if (override_dir[0] != '\0') return override_dir;
  }
#ifdef _WIN32
  if (const char *local_appdata = std::getenv("LOCALAPPDATA")) {
    if (local_appdata[0] != '\0') return std::string(local_appdata) + "\\vkapriltag";
  }
#else
  if (const char *xdg_cache = std::getenv("XDG_CACHE_HOME")) {
    if (xdg_cache[0] != '\0') return std::string(xdg_cache) + "/vkapriltag";
  }
  if (const char *home = std::getenv("HOME")) {
    if (home[0] != '\0') return std::string(home) + "/.cache/vkapriltag";
  }
#endif
  return "";
}

// A cache blob larger than this is treated as corrupt rather than trusted -
// this pipeline creates on the order of 30 small compute pipelines, so a
// well-formed cache has no legitimate reason to approach this size.
constexpr size_t kMaxCacheFileBytes = 16u * 1024 * 1024;

// Our own framing around the driver's blob: an 8-byte FNV-1a of the payload,
// so a process killed mid-write is detected as corrupt rather than handed to
// the driver. The driver's own VkPipelineCacheHeaderVersionOne is validated
// separately, on the payload itself.
constexpr size_t kOurHeaderBytes = sizeof(uint64_t);

std::vector<uint8_t> LoadValidated(const std::string &file_path, VkPhysicalDevice physical_device,
                                   bool verbose) {
  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) return {};

  const std::streamsize size = file.tellg();
  if (size <= static_cast<std::streamsize>(kOurHeaderBytes) ||
      size > static_cast<std::streamsize>(kMaxCacheFileBytes)) {
    return {};
  }
  file.seekg(0);
  std::vector<uint8_t> raw(static_cast<size_t>(size));
  file.read(reinterpret_cast<char *>(raw.data()), size);
  if (!file) return {};

  uint64_t stored_hash = 0;
  std::memcpy(&stored_hash, raw.data(), sizeof(stored_hash));
  std::vector<uint8_t> payload(raw.begin() + kOurHeaderBytes, raw.end());
  if (Fnv1a(payload.data(), payload.size()) != stored_hash) {
    if (verbose) {
      std::cerr << "apriltag_vulkan: pipeline cache " << file_path
                << " failed checksum validation; ignoring." << std::endl;
    }
    return {};
  }

  if (payload.size() < sizeof(VkPipelineCacheHeaderVersionOne)) return {};
  VkPipelineCacheHeaderVersionOne header{};
  std::memcpy(&header, payload.data(), sizeof(header));

  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(physical_device, &props);

  if (header.headerVersion != VK_PIPELINE_CACHE_HEADER_VERSION_ONE ||
      header.vendorID != props.vendorID || header.deviceID != props.deviceID ||
      std::memcmp(header.pipelineCacheUUID, props.pipelineCacheUUID, VK_UUID_SIZE) != 0) {
    if (verbose) {
      std::cerr << "apriltag_vulkan: pipeline cache " << file_path
                << " does not match the selected device; ignoring." << std::endl;
    }
    return {};
  }

  return payload;
}

}  // namespace

std::string PipelineCache::CacheFilePath(VkPhysicalDevice physical_device,
                                         uint64_t shader_corpus_hash, bool verbose) const {
  const std::string dir = CacheDirectory();
  if (dir.empty()) {
    if (verbose) {
      std::cerr << "apriltag_vulkan: no writable cache directory found "
                   "(set APRILTAG_VK_CACHE_DIR to enable pipeline caching); disabled."
                << std::endl;
    }
    return "";
  }
  std::error_code ec;
  fs::create_directories(dir, ec);
  if (ec) {
    if (verbose) {
      std::cerr << "apriltag_vulkan: could not create pipeline cache directory " << dir << ": "
                << ec.message() << "; disabled." << std::endl;
    }
    return "";
  }

  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(physical_device, &props);

  std::ostringstream name;
  name << "vkapriltag-" << HexU32(props.vendorID) << "-" << HexU32(props.deviceID) << "-"
       << HexBytes(props.pipelineCacheUUID, VK_UUID_SIZE) << "-" << std::hex << shader_corpus_hash
       << ".vkpc";

  return (fs::path(dir) / name.str()).string();
}

PipelineCache::PipelineCache(VkDevice device, VkPhysicalDevice physical_device,
                             const std::string &shader_dir, bool enabled, bool verbose)
    // Skip the directory scan entirely when caching is off - the hash is only
    // ever used to name the cache file, which this build will not write.
    : PipelineCache(device, physical_device, enabled ? HashShaderDir(shader_dir) : 0, enabled,
                    verbose) {}

PipelineCache::PipelineCache(VkDevice device, VkPhysicalDevice physical_device,
                             uint64_t shader_corpus_hash, bool enabled, bool verbose)
    : device_(device), verbose_(verbose) {
  if (!enabled) return;

  file_path_ = CacheFilePath(physical_device, shader_corpus_hash, verbose);

  std::vector<uint8_t> initial_data;
  if (!file_path_.empty()) {
    initial_data = LoadValidated(file_path_, physical_device, verbose);
  }

  VkPipelineCacheCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
  if (!initial_data.empty()) {
    create_info.initialDataSize = initial_data.size();
    create_info.pInitialData = initial_data.data();
  }

  VkResult result = vkCreatePipelineCache(device_, &create_info, nullptr, &cache_);
  if (result != VK_SUCCESS && !initial_data.empty()) {
    // A blob that passed our own validation can still be one this exact
    // driver build refuses (a corrupted-but-checksum-consistent file, or a
    // driver quirk) - fall back to an empty cache rather than failing
    // detector startup entirely.
    if (verbose) {
      std::cerr << "apriltag_vulkan: driver rejected pipeline cache data from " << file_path_
                << "; starting with an empty cache." << std::endl;
    }
    create_info.initialDataSize = 0;
    create_info.pInitialData = nullptr;
    result = vkCreatePipelineCache(device_, &create_info, nullptr, &cache_);
    initial_data.clear();
  }
  CheckVk(result, "vkCreatePipelineCache");

  if (!initial_data.empty()) {
    last_synced_hash_ = Fnv1a(initial_data.data(), initial_data.size());
    has_synced_hash_ = true;
    if (verbose) {
      std::cerr << "apriltag_vulkan: loaded pipeline cache from " << file_path_ << " ("
                << initial_data.size() << " bytes)." << std::endl;
    }
  } else if (verbose && !file_path_.empty()) {
    std::cerr << "apriltag_vulkan: starting a fresh pipeline cache at " << file_path_ << "."
              << std::endl;
  }
}

PipelineCache::~PipelineCache() {
  Save();
  Destroy();
}

void PipelineCache::Destroy() {
  if (cache_) vkDestroyPipelineCache(device_, cache_, nullptr);
  cache_ = VK_NULL_HANDLE;
}

PipelineCache::PipelineCache(PipelineCache &&other) noexcept
    : device_(other.device_),
      cache_(other.cache_),
      file_path_(std::move(other.file_path_)),
      verbose_(other.verbose_),
      last_synced_hash_(other.last_synced_hash_),
      has_synced_hash_(other.has_synced_hash_) {
  other.cache_ = VK_NULL_HANDLE;
  other.file_path_.clear();
}

PipelineCache &PipelineCache::operator=(PipelineCache &&other) noexcept {
  if (this != &other) {
    Save();
    Destroy();
    device_ = other.device_;
    cache_ = other.cache_;
    file_path_ = std::move(other.file_path_);
    verbose_ = other.verbose_;
    last_synced_hash_ = other.last_synced_hash_;
    has_synced_hash_ = other.has_synced_hash_;
    other.cache_ = VK_NULL_HANDLE;
    other.file_path_.clear();
  }
  return *this;
}

void PipelineCache::Save() const {
  if (cache_ == VK_NULL_HANDLE || file_path_.empty()) return;

  size_t data_size = 0;
  if (vkGetPipelineCacheData(device_, cache_, &data_size, nullptr) != VK_SUCCESS ||
      data_size == 0) {
    return;
  }
  std::vector<uint8_t> data(data_size);
  if (vkGetPipelineCacheData(device_, cache_, &data_size, data.data()) != VK_SUCCESS) {
    return;
  }
  data.resize(data_size);

  const uint64_t hash = Fnv1a(data.data(), data.size());
  if (has_synced_hash_ && hash == last_synced_hash_) {
    // Nothing new since we loaded (or last saved) - the common warm-run
    // case - so skip touching the filesystem at all.
    return;
  }
  if (data.size() > kMaxCacheFileBytes) {
    if (verbose_) {
      std::cerr << "apriltag_vulkan: pipeline cache grew to " << data.size()
                << " bytes, above the " << kMaxCacheFileBytes << "-byte sanity cap; not writing."
                << std::endl;
    }
    return;
  }

#ifdef _WIN32
  const unsigned long pid = GetCurrentProcessId();
#else
  const long pid = static_cast<long>(getpid());
#endif
  const std::string tmp_path = file_path_ + ".tmp." + std::to_string(pid);

  {
    std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
      if (verbose_) {
        std::cerr << "apriltag_vulkan: could not write pipeline cache to " << tmp_path
                  << std::endl;
      }
      return;
    }
    out.write(reinterpret_cast<const char *>(&hash), sizeof(hash));
    out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
  }

  std::error_code ec;
  fs::rename(tmp_path, file_path_, ec);
  if (ec) {
    if (verbose_) {
      std::cerr << "apriltag_vulkan: could not replace pipeline cache " << file_path_ << ": "
                << ec.message() << std::endl;
    }
    fs::remove(tmp_path, ec);
    return;
  }

  last_synced_hash_ = hash;
  has_synced_hash_ = true;
  if (verbose_) {
    std::cerr << "apriltag_vulkan: saved pipeline cache to " << file_path_ << " (" << data.size()
              << " bytes)." << std::endl;
  }
}

}  // namespace apriltag_vulkan::vk
