#pragma once

namespace apriltag_vulkan {

// The on/off choices of the per-candidate decode steps; the defaults are TagDecoder's.
struct DecodeSwitches {
  bool own_decode = true;
  bool grey_margin = true;
  bool small_rescue = false;
  bool polish = false;
  bool side_one_out = true;
  bool third_bit = true;
  bool family_limits = true;
  bool border_precheck = true;
  bool dark_reject = true;
  bool early_reject = true;
  bool collect_probe = false;
  float probe_margin = 0.0f;
};

}  // namespace apriltag_vulkan
