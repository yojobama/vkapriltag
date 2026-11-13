#include "vkapriltag/common/Options.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <iterator>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace apriltag_vulkan {
namespace {

using T = OptionType;
using C = OptionCategory;
using P = OptionPrecedence;

// Grouped by category; within a group, by the file that reads them.
const OptionDescriptor kTable[] = {
    // --- Features: change what is detected. ---
    {"APRILTAG_VK_SMALL_RESCUE", T::kBool, "off", C::kFeature, P::kEnvWins, "DetectorConfig::small_tag_rescue",
     "Rescue small tags: label small blobs, fit boxes, refit from the full-resolution image."},
    {"APRILTAG_VK_HULL_QUAD", T::kBool, "off", C::kFeature, P::kEnvWins, "DetectorConfig::hull_quad",
     "Retry a failed quad fit on the convex hull of the blob (partly covered border)."},
    {"APRILTAG_VK_QUADFIT", T::kEnum, "dp", C::kFeature, P::kEnvWins, "DetectorConfig::quad_fit_method",
     "Corner seeding: dp or peaks (other values are ignored)."},
    {"APRILTAG_VK_REFINE", T::kEnum, "profile", C::kFeature, P::kEnvWins, "TagDecoder refine_method",
     "Edge refinement: exact or profile (other values are ignored); used when TagDecoderConfig::refine_edges is set."},
    {"APRILTAG_VK_POLISH", T::kBool, "off", C::kFeature, P::kEnvWins, "TagDecoder::SetPolish",
     "Whole-tag photometric polish of decoded tags."},
    {"APRILTAG_VK_PROFILE_FEATURES", T::kMask, "31", C::kFeature, P::kEnvOnly, "",
     "Bit mask of the profile fit's refinements: 1 outliers, 2 coarse to fine, 4 halfway crossing, 8 clip bias, 16 widen blur."},
    {"APRILTAG_VK_RS_SPACING1", T::kFloat, "1.5", C::kFeature, P::kEnvOnly, "", "experiment"},
    {"APRILTAG_VK_RS_HALF", T::kFloat, "1", C::kFeature, P::kEnvOnly, "", "experiment"},
    {"APRILTAG_VK_RS_MAXN", T::kInt, "96", C::kFeature, P::kEnvOnly, "", "experiment"},
    {"APRILTAG_VK_RS_ABORT", T::kFloat, "-1", C::kFeature, P::kEnvOnly, "", "experiment"},
    {"APRILTAG_VK_GREY_MARGIN", T::kBool, "on", C::kFeature, P::kEnvOnly, "",
     "Retry an undecoded quad as its inner square (grey margin)."},
    {"APRILTAG_VK_SIDE_ONE_OUT", T::kBool, "on", C::kFeature, P::kEnvOnly, "",
     "Refit a side whose border cells read white, one cell further in."},
    {"APRILTAG_VK_THIRD_BIT", T::kBool, "on", C::kFeature, P::kEnvOnly, "",
     "Accept a third corrected bit when the three wrong cells touch."},
    {"APRILTAG_VK_FAMILY_LIMITS", T::kBool, "on", C::kFeature, P::kEnvOnly, "",
     "Reject detections that needed more corrected bits than their family can bear."},
    {"APRILTAG_VK_BORDER_PRECHECK", T::kBool, "on", C::kFeature, P::kEnvOnly, "",
     "Reject a quad whose border cells are not mostly dark before refining."},
    {"APRILTAG_VK_DARK_REJECT", T::kBool, "on", C::kFeature, P::kEnvOnly, "",
     "Reject a quad whose payload has fewer white cells than any code."},
    {"APRILTAG_VK_EARLY_REJECT", T::kBool, "on", C::kFeature, P::kEnvOnly, "",
     "With grey margin on, the border and dark checks reject a quad only if its inner cells fail them too."},
    {"APRILTAG_VK_PROBE_MARGIN", T::kFloat, "0 (constructor argument)", C::kFeature, P::kEnvWins, "TagDecoder probe_margin",
     "Margin gate in grey levels before edge refinement; 0 is off."},
    {"APRILTAG_VK_MIN_DIFF", T::kInt, "5", C::kFeature, P::kEnvWins, "DetectorConfig::min_white_black_diff",
     "Minimum white-black difference for a block to be labelled."},
    {"APRILTAG_VK_MIN_TAG_PX", T::kInt, "0", C::kFeature, P::kEnvWins, "DetectorConfig::min_tag_pixels",
     "Smallest tag side in full-resolution pixels; raises the minimum cluster size."},
    {"APRILTAG_VK_BOX_DECIMATE", T::kBool, "on", C::kFeature, P::kEnvWins, "DetectorConfig::box_decimate",
     "Average each decimation block instead of copying its top-left pixel."},
    {"APRILTAG_VK_ADAPTIVE_FLOOR", T::kFloat, "2.0", C::kFeature, P::kEnvWins, "DetectorConfig::adaptive_floor_scale",
     "Contrast floor as a multiple of the frame's median block range; 0 keeps the fixed floor."},
    {"APRILTAG_VK_TRACKING", T::kBool, "off", C::kFeature, P::kEnvWins, "TrackingConfig::enabled",
     "Track tags between keyframes instead of running the full detector every frame."},
    {"APRILTAG_VK_TRACKING_KEYFRAME", T::kInt, "8", C::kFeature, P::kEnvWins, "TrackingConfig::keyframe_interval",
     "At most this many consecutive frames skip the GPU (at least 1)."},
    {"APRILTAG_VK_TRACKING_MIN_MARGIN", T::kFloat, "10", C::kFeature, P::kEnvWins, "TrackingConfig::min_margin",
     "Smallest decision margin for a tracked tag."},
    {"APRILTAG_VK_TRACKING_CONVERGE", T::kFloat, "0.05", C::kFeature, P::kEnvWins, "TrackingConfig::converge_px",
     "Corner movement in pixels below which a tracked tag has settled."},
    {"APRILTAG_VK_TRACKING_ITERATIONS", T::kInt, "3", C::kFeature, P::kEnvWins, "TrackingConfig::max_iterations",
     "Most decodes per tracked tag (at least 1)."},
    {"APRILTAG_VK_TRACKING_ASYNC", T::kBool, "off", C::kFeature, P::kEnvWins, "TrackingConfig::async",
     "Run the full detector on a background thread while tracked frames return at the tracked cost."},
    {"APRILTAG_VK_TRACKING_ASYNC_INTERVAL", T::kInt, "1", C::kFeature, P::kEnvWins, "TrackingConfig::async_interval",
     "The background detector starts at most every this many frames (at least 1)."},
    {"APRILTAG_VK_TRACKING_ASYNC_COPY", T::kBool, "on", C::kFeature, P::kEnvWins, "TrackingConfig::async_copy",
     "Copy each frame for the background detector."},
    {"APRILTAG_VK_TRACKING_ASYNC_DROP", T::kBool, "on", C::kFeature, P::kEnvWins, "TrackingConfig::async_drop_lost",
     "A lost track stays out of the output until a background result finds it again."},

    // --- Tuning: same detections, different cost. ---
    {"APRILTAG_VK_WG", T::kInt, "device", C::kTuning, P::kEnvWins, "ContextOptions::workgroup_size_override",
     "1D workgroup size (power of two, clamped to limits); must be positive."},
    {"APRILTAG_VK_WG2D", T::kSize, "device", C::kTuning, P::kEnvWins, "ContextOptions::workgroup_size_2d_x/y",
     "2D workgroup size as <w>x<h>."},
    {"APRILTAG_VK_PIPELINE_CACHE", T::kBool, "on", C::kTuning, P::kDisableOnly, "ContextOptions::use_pipeline_cache",
     "Any value other than a non-zero one disables the on-disk pipeline cache."},
    {"APRILTAG_VK_CACHE_DIR", T::kString, "platform cache directory", C::kTuning, P::kEnvOnly, "",
     "Directory of the on-disk pipeline cache."},
    {"APRILTAG_VK_SPIN_WAIT", T::kBool, "off", C::kTuning, P::kEnvOnly, "",
     "Poll a fence for up to 2 ms before sleeping in the driver (read once per process)."},
    {"APRILTAG_CPU_THREADS", T::kInt, "0 (hardware concurrency)", C::kTuning, P::kConfigWins, "DetectorConfig::cpu_threads",
     "Parallelism of the CPU tail, including the calling thread; used only when the configured value is 0."},
    {"APRILTAG_CPU_BIG_CORES", T::kBool, "on", C::kTuning, P::kEnvWins, "DetectorConfig::cpu_big_cores",
     "Pin workers to the big cores where core capacities differ."},
    {"APRILTAG_VK_MAX_POINTS", T::kInt, "0 (dense worst case)", C::kTuning, P::kEnvWins, "DetectorConfig::max_boundary_points",
     "Cap on boundary points per frame; must be positive."},
    {"APRILTAG_VK_UF_CHUNK", T::kInt, "2", C::kTuning, P::kEnvWins, "DetectorConfig::uf_iterations_per_chunk",
     "Labelling iterations issued per convergence check; must be positive."},
    {"APRILTAG_VK_SORT_WG", T::kInt, "device maximum (256 integrated, 1024 other)", C::kTuning, P::kEnvOnly, "",
     "Workgroup size of the point sort."},
    {"APRILTAG_VK_SORT_SPLIT", T::kInt, "1024 (bounded by shared memory)", C::kTuning, P::kEnvOnly, "",
     "Blobs above this many points are sorted in runs of this length (power of two)."},
    {"APRILTAG_VK_SORT_RANK_ALL", T::kInt, "256", C::kTuning, P::kEnvOnly, "",
     "Blobs of at most this many points are sorted by ranking every key; 0 disables."},
    {"APRILTAG_VK_FLAG_POLL", T::kBool, "on with device-coherent host memory", C::kTuning, P::kEnvOnly, "",
     "Detect completion by polling a flag instead of waiting on a fence."},
    {"APRILTAG_VK_KEEPALIVE_MS", T::kInt, "-1 (6 integrated, 0 other)", C::kTuning, P::kEnvWins, "DetectorConfig::keepalive_ms",
     "Milliseconds between keep-alive dispatches; 0 disables."},
    {"APRILTAG_VK_PREWARM_LEAD_US", T::kInt, "-1 (3000)", C::kTuning, P::kEnvWins, "DetectorConfig::prewarm_lead_us",
     "Microseconds before the next frame is due that a pre-warm burst starts; 0 disables."},
    {"APRILTAG_VK_PREWARM_BURST", T::kInt, "2", C::kTuning, P::kEnvWins, "DetectorConfig::prewarm_burst",
     "Dispatches per pre-warm burst (at least 1)."},
    {"APRILTAG_VK_PREWARM_GAP_US", T::kInt, "1000", C::kTuning, P::kEnvWins, "DetectorConfig::prewarm_gap_us",
     "Microseconds between the dispatches of a burst."},
    {"APRILTAG_VK_PREWARM_GROUPS", T::kInt, "keep-alive default", C::kTuning, P::kEnvOnly, "",
     "Workgroups per pre-warm dispatch (at least 1)."},
    {"APRILTAG_VK_FUSE_SUBMITS", T::kBool, "on when the readbacks are direct", C::kTuning, P::kDisableOnly, "",
     "Record the last three submissions of a frame as one; 0 keeps the unfused flow."},
    {"APRILTAG_VK_UF_VERIFY", T::kBool, "off", C::kTuning, P::kEnvOnly, "",
     "Run a second merge that checks the labelling converged, with a retry path."},
    {"APRILTAG_VK_PRERECORD", T::kBool, "on when fused without a transfer lane", C::kTuning, P::kDisableOnly, "",
     "Replay pre-recorded command buffers; 0 records every frame."},
    {"APRILTAG_VK_DOT", T::kBool, "on where supported", C::kTuning, P::kDisableOnly, "",
     "Integer dot product box decimation; 0 uses the plain shader."},
    {"APRILTAG_VK_FUSED_THRESHOLD", T::kBool, "on with 8-bit storage", C::kTuning, P::kDisableOnly, "",
     "Fused block min/max and threshold; 0 keeps the separate passes."},
    {"APRILTAG_VK_UF_INIT_SCAN", T::kBool, "on where supported", C::kTuning, P::kDisableOnly, "",
     "Initialise the labelling with a workgroup scan; 0 uses uf_init and a compression."},
    {"APRILTAG_VK_COUNT_RUNS", T::kBool, "on for integrated GPUs", C::kTuning, P::kEnvOnly, "",
     "Count blob sizes per run piece (needs the init scan)."},
    {"APRILTAG_VK_SPARSE", T::kBool, "on", C::kTuning, P::kEnvWins, "DetectorConfig::sparse_labelling",
     "Dispatch the labelling over the cells that hold contrast only (needs fused threshold, init scan, count runs)."},
    {"APRILTAG_VK_MERGE_X4", T::kBool, "on for integrated GPUs", C::kTuning, P::kEnvOnly, "",
     "Merge four pixels per thread (needs 8-bit storage and a width divisible by 4)."},
    {"APRILTAG_VK_MERGE_BAND", T::kInt, "8 discrete, 0 integrated or sparse", C::kTuning, P::kEnvOnly, "",
     "Rows merged in bands of this many (power of two) before the band boundaries; 0 is one pass."},
    {"APRILTAG_VK_LEAN_TAIL", T::kMask, "3 integrated, 0 other", C::kTuning, P::kEnvOnly, "",
     "Leaner tail: bit 1 fixed-grid kernels, bit 2 offsets from select."},
    {"APRILTAG_VK_TRACKING_ASYNC_THREADS", T::kInt, "1", C::kTuning, P::kEnvWins, "TrackingConfig::async_threads",
     "Threads of the background decoder; 0 is all cores."},
    {"APRILTAG_VK_POLISH_N", T::kInt, "8000", C::kTuning, P::kEnvOnly, "",
     "Most pixels the polish fit uses per tag (read once per process)."},
    {"APRILTAG_VK_POLISH_BLUREXTRA", T::kFloat, "2.0", C::kTuning, P::kEnvOnly, "",
     "Polish: extra blur width, in pixels, added to the window (read once per process)."},
    {"APRILTAG_VK_POLISH_WINMULT", T::kFloat, "0.8", C::kTuning, P::kEnvOnly, "",
     "Polish: window size as a multiple of the blur width."},
    {"APRILTAG_VK_POLISH_MINWIN", T::kFloat, "0.5", C::kTuning, P::kEnvOnly, "",
     "Polish: smallest window, in pixels."},
    {"APRILTAG_VK_POLISH_SMAXM", T::kFloat, "1.3", C::kTuning, P::kEnvOnly, "",
     "Polish: largest blur width as a multiple of the starting estimate."},
    {"APRILTAG_VK_POLISH_TOL", T::kFloat, "0.004", C::kTuning, P::kEnvOnly, "",
     "Polish: convergence tolerance of the parameter step."},
    {"APRILTAG_VK_POLISH_ITERS", T::kFloat, "6", C::kTuning, P::kEnvOnly, "",
     "Polish: most iterations."},
    {"APRILTAG_VK_POLISH_REACH", T::kFloat, "1.6", C::kTuning, P::kEnvOnly, "",
     "Polish: how far from the edge, in blur widths, pixels still count."},

    // --- Device forcing: retarget the binary at other hardware limits. ---
    {"APRILTAG_VK_ALLOW_CPU", T::kBool, "off", C::kDeviceForcing, P::kEnableOnly, "ContextOptions::allow_cpu_device",
     "Permit a software Vulkan device (benchmarking and CI only)."},
    {"APRILTAG_VK_DEVICE", T::kInt, "-1 (best score)", C::kDeviceForcing, P::kEnvWins, "ContextOptions::device_index",
     "Pick a physical device by index."},
    {"APRILTAG_VK_FORCE_NO_8BIT", T::kBool, "off", C::kDeviceForcing, P::kEnableOnly, "ContextOptions::force_no_8bit_storage",
     "Use the 32-bit-per-pixel shader family even where 8-bit storage exists."},
    {"APRILTAG_VK_FORCE_NO_SUBGROUP", T::kBool, "off", C::kDeviceForcing, P::kEnableOnly, "ContextOptions::force_no_subgroup",
     "Use the scalar shader variants even where subgroup operations exist."},
    {"APRILTAG_VK_FORCE_NO_INT64_ATOMIC", T::kBool, "off", C::kDeviceForcing, P::kEnableOnly,
     "ContextOptions::force_no_int64_atomics", "Use the 32-bit atomic extents reduction."},
    {"APRILTAG_VK_FORCE_NO_CONDITIONAL_RENDERING", T::kBool, "off", C::kDeviceForcing, P::kEnableOnly,
     "ContextOptions::force_no_conditional_rendering", "Skip converged compression passes with the in-shader guard."},
    {"APRILTAG_VK_MAX_INVOCATIONS", T::kInt, "device", C::kDeviceForcing, P::kEnvWins,
     "ContextOptions::max_invocations_override", "Pretend the device caps workgroups at this many invocations; must be positive."},

    // --- Diagnostics: output and analysis aids. ---
    {"APRILTAG_VK_VALIDATION", T::kBool, "off", C::kDiagnostic, P::kEnableOnly, "ContextOptions::enable_validation",
     "Enable the Khronos validation layer if installed."},
    {"APRILTAG_VK_TIMESTAMPS", T::kInt, "off", C::kDiagnostic, P::kEnvOnly, "",
     "Non-zero emits the per-dispatch GPU timestamp breakdown (tracing also enables it)."},
    {"APRILTAG_VK_TRACE", T::kString, "off", C::kDiagnostic, P::kEnvOnly, "",
     "Write a Chrome-trace JSON of the CPU and GPU spans to this file."},
    {"APRILTAG_VK_DISPATCH_TIMES", T::kBool, "off", C::kDiagnostic, P::kEnvOnly, "",
     "When set to any value, print the median GPU time of every dispatch at exit (read once per process)."},
    {"APRILTAG_VK_PROBE_STATS", T::kBool, "off", C::kDiagnostic, P::kEnvOnly, "",
     "Histogram each unrefined quad's Hamming distance by outcome."},
    {"APRILTAG_VK_PROBE_CSV", T::kString, "off", C::kDiagnostic, P::kEnvOnly, "",
     "Append one CSV row per quad to this file (also turns on the probe statistics)."},
    {"APRILTAG_VK_POLISH_DEV", T::kMask, "0", C::kDiagnostic, P::kEnvOnly, "",
     "Development bit mask of the polish fit (read once per process)."},

    // --- Platform variables the library reads (not APRILTAG switches). ---
    {"LOCALAPPDATA", T::kString, "unset", C::kPlatform, P::kPlatform, "",
     "Windows: the pipeline cache is kept under %LOCALAPPDATA%\\vkapriltag."},
    {"XDG_CACHE_HOME", T::kString, "unset", C::kPlatform, P::kPlatform, "",
     "Elsewhere: the pipeline cache is kept under $XDG_CACHE_HOME/vkapriltag."},
    {"HOME", T::kString, "unset", C::kPlatform, P::kPlatform, "",
     "Elsewhere, without XDG_CACHE_HOME: the cache is kept under $HOME/.cache/vkapriltag."},
};

const std::unordered_map<std::string_view, size_t> &Index() {
  static const std::unordered_map<std::string_view, size_t> index = [] {
    std::unordered_map<std::string_view, size_t> m;
    for (size_t i = 0; i < std::size(kTable); ++i) m.emplace(kTable[i].name, i);
    return m;
  }();
  return index;
}

}  // namespace

std::span<const OptionDescriptor> OptionTable() { return kTable; }

const char *OptionTypeName(OptionType type) {
  switch (type) {
    case T::kBool: return "bool";
    case T::kInt: return "int";
    case T::kFloat: return "float";
    case T::kEnum: return "enum";
    case T::kString: return "string";
    case T::kMask: return "bit mask";
    case T::kSize: return "WxH";
  }
  return "";
}

const char *OptionCategoryName(OptionCategory category) {
  switch (category) {
    case C::kFeature: return "Features (change what is detected)";
    case C::kTuning: return "Tuning (same detections, different cost)";
    case C::kDeviceForcing: return "Device forcing";
    case C::kDiagnostic: return "Diagnostics";
    case C::kPlatform: return "Platform variables read";
  }
  return "";
}

const char *OptionPrecedenceName(OptionPrecedence precedence) {
  switch (precedence) {
    case P::kEnvWins: return "environment wins";
    case P::kConfigWins: return "configuration wins";
    case P::kEnvOnly: return "environment only";
    case P::kEnableOnly: return "environment can only enable";
    case P::kDisableOnly: return "environment can only disable";
    case P::kPlatform: return "fallback";
  }
  return "";
}

std::string OptionsMarkdown() {
  std::string out;
  for (OptionCategory category : {C::kFeature, C::kTuning, C::kDeviceForcing, C::kDiagnostic, C::kPlatform}) {
    out += "### ";
    out += OptionCategoryName(category);
    out += "\n\n| Variable | Type | Default | Configuration field | Precedence | Effect |\n| --- | --- | --- | --- | --- | --- |\n";
    for (const OptionDescriptor &d : kTable) {
      if (d.category != category) continue;
      out += "| `";
      out += d.name;
      out += "` | ";
      out += OptionTypeName(d.type);
      out += " | ";
      out += d.default_text;
      out += " | ";
      if (d.config[0] != '\0') {
        out += "`";
        out += d.config;
        out += "`";
      }
      out += " | ";
      out += OptionPrecedenceName(d.precedence);
      out += " | ";
      out += d.doc;
      out += " |\n";
    }
    out += "\n";
  }
  return out;
}

Options Options::FromEnvironment() {
  Options o;
  o.values_.resize(std::size(kTable));
  for (size_t i = 0; i < std::size(kTable); ++i) {
    if (const char *v = std::getenv(kTable[i].name)) o.values_[i] = v;
  }
  return o;
}

const std::optional<std::string> &Options::Raw(const char *name) const {
  const auto it = Index().find(name);
  if (it == Index().end()) {
    throw std::logic_error(std::string("environment variable is not in the options registry: ") + name);
  }
  return values_.at(it->second);
}

bool Options::Has(const char *name) const { return Raw(name).has_value(); }

bool Options::Bool(const char *name, bool fallback) const {
  const auto &v = Raw(name);
  return v ? (*v)[0] != '0' : fallback;
}

bool Options::Flag(const char *name) const {
  const auto &v = Raw(name);
  return v && !v->empty() && (*v)[0] != '0';
}

int Options::Int(const char *name, int fallback) const {
  const auto &v = Raw(name);
  return v ? std::atoi(v->c_str()) : fallback;
}

std::optional<long> Options::Long(const char *name) const {
  const auto &v = Raw(name);
  if (!v) return std::nullopt;
  return std::strtol(v->c_str(), nullptr, 10);
}

std::optional<unsigned long> Options::ULong(const char *name) const {
  const auto &v = Raw(name);
  if (!v) return std::nullopt;
  return std::strtoul(v->c_str(), nullptr, 10);
}

double Options::Float(const char *name, double fallback) const {
  const auto &v = Raw(name);
  return v ? std::atof(v->c_str()) : fallback;
}

std::optional<int> Options::StrictInt(const char *name) const {
  const auto &v = Raw(name);
  if (!v || v->empty()) return std::nullopt;
  char *end = nullptr;
  errno = 0;
  const long parsed = std::strtol(v->c_str(), &end, 10);
  if (end == v->c_str() || *end != '\0' || errno == ERANGE || parsed < INT_MIN || parsed > INT_MAX) {
    throw std::runtime_error(std::string(name) + "=" + *v + " is not an integer");
  }
  return static_cast<int>(parsed);
}

std::optional<std::pair<uint32_t, uint32_t>> Options::WxH(const char *name) const {
  const auto &v = Raw(name);
  if (!v || v->empty()) return std::nullopt;
  char *end = nullptr;
  const long w = std::strtol(v->c_str(), &end, 10);
  if (end == v->c_str() || *end != 'x' || w <= 0) return std::nullopt;
  const char *h_start = end + 1;
  const long h = std::strtol(h_start, &end, 10);
  if (end == h_start || *end != '\0' || h <= 0) return std::nullopt;
  return std::make_pair(static_cast<uint32_t>(w), static_cast<uint32_t>(h));
}

int Options::Choice(const char *name, std::initializer_list<const char *> values) const {
  const auto &v = Raw(name);
  if (!v) return -1;
  int i = 0;
  for (const char *candidate : values) {
    if (*v == candidate) return i;
    ++i;
  }
  return -1;
}

std::optional<std::string> Options::String(const char *name) const { return Raw(name); }

bool ResolveSmallTagRescue(const Options &env, bool configured) {
  return env.Bool("APRILTAG_VK_SMALL_RESCUE", configured);
}

}  // namespace apriltag_vulkan
