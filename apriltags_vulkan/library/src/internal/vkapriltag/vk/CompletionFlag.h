#pragma once

#include <vulkan/vulkan.h>

#include <chrono>

#include "vkapriltag/vk/Buffer.h"
#include "vkapriltag/vk/Context.h"

namespace apriltag_vulkan::vk {

// A word the last command of a submission sets, so the host can see the work is done without
// waiting for the fence to wake it. Only memory with VK_AMD_device_coherent_memory makes the host
// see the write while the submission is still running; elsewhere it is plain host-visible memory and
// the observation is outside the specification.
class CompletionFlag {
 public:
  explicit CompletionFlag(const Context &ctx);

  // True when the word is in device-coherent memory.
  bool spec_defined() const { return spec_defined_; }

  // Clears the word; call before the submission that will set it.
  void Arm();

  // Records the write that sets the word after every earlier command in `cmd`.
  void RecordSet(VkCommandBuffer cmd) const;

  // Polls the word for up to `limit`; true when it was set.
  bool Wait(std::chrono::microseconds limit) const;

 private:
  Buffer buffer_;
  bool spec_defined_ = false;
};

}  // namespace apriltag_vulkan::vk
