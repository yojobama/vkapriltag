#pragma once

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace apriltag_vulkan {

// The one place the library reads the environment. Every variable is a row of Table(); reading an
// unregistered name throws std::logic_error.
//
// Precedence: an environment variable overrides the configuration field it matches. Each row's
// `precedence` names the exceptions: APRILTAG_CPU_THREADS (a non-zero configured value wins), switches
// that can only turn a feature on or only turn it off, and variables with no configuration field.
//
// Resolution: Options::FromEnvironment() copies every registered variable once; the accessors then read
// that copy. A detector, decoder or context takes one copy at construction. A process-wide static takes
// one at first use, so it is read once per process.

enum class OptionType { kBool, kInt, kFloat, kEnum, kString, kMask, kSize };
enum class OptionCategory { kFeature, kTuning, kDeviceForcing, kDiagnostic, kPlatform };
enum class OptionPrecedence { kEnvWins, kConfigWins, kEnvOnly, kEnableOnly, kDisableOnly, kPlatform };

struct OptionDescriptor {
  const char *name;
  OptionType type;
  const char *default_text;  // the value used when the variable is unset; may depend on the device
  OptionCategory category;
  OptionPrecedence precedence;
  const char *config;  // the configuration field it overrides ("" if none)
  const char *doc;     // one line
};

// Every registered variable, grouped by category.
std::span<const OptionDescriptor> OptionTable();
const char *OptionTypeName(OptionType type);
const char *OptionCategoryName(OptionCategory category);
const char *OptionPrecedenceName(OptionPrecedence precedence);

// The switch reference as markdown, one table per category (printed by `vk_eval --list-options`).
std::string OptionsMarkdown();

class Options {
 public:
  // Reads every registered variable once.
  static Options FromEnvironment();

  // Set, even to an empty string.
  bool Has(const char *name) const;
  // Set: true unless the first character is '0' (an empty value is true); unset: `fallback`.
  bool Bool(const char *name, bool fallback) const;
  // Set, non-empty and not starting with '0'.
  bool Flag(const char *name) const;
  // Set: atoi of the value; unset: `fallback`.
  int Int(const char *name, int fallback) const;
  // Set: strtol / strtoul of the value (base 10, no error check).
  std::optional<long> Long(const char *name) const;
  std::optional<unsigned long> ULong(const char *name) const;
  // Set: atof of the value; unset: `fallback`.
  double Float(const char *name, double fallback) const;
  // Unset or empty: nothing. Otherwise the integer, or throws std::runtime_error if malformed.
  std::optional<int> StrictInt(const char *name) const;
  // "<w>x<h>" with both positive; nothing if unset or malformed.
  std::optional<std::pair<uint32_t, uint32_t>> WxH(const char *name) const;
  // Index in `values` of the exact value, or -1 (unset or not listed).
  int Choice(const char *name, std::initializer_list<const char *> values) const;
  // The raw value if set.
  std::optional<std::string> String(const char *name) const;

 private:
  const std::optional<std::string> &Raw(const char *name) const;
  std::vector<std::optional<std::string>> values_;
};

// Small-tag rescue is one concept read by the GPU labelling, the quad fit and the tag decoder; all
// three take their value from here.
bool ResolveSmallTagRescue(const Options &env, bool configured);

}  // namespace apriltag_vulkan
