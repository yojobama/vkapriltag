#pragma once

#include <cstddef>
#include <cstdint>

namespace apriltag_vulkan::vk {

// SPIR-V compiled into the binary at build time. With VKAPRILTAG_EMBED_SHADERS=OFF the table is
// empty and HasEmbeddedShaders() is false.
struct EmbeddedShader {
  const char *name;      // shader base name, e.g. "decimate"
  const uint32_t *code;  // SPIR-V words
  size_t bytes;          // byte count, always a multiple of 4
};

// The shader named `name` (e.g. "decimate"), or nullptr if embedding is disabled or it is unknown.
const EmbeddedShader *FindEmbeddedShader(const char *name);

// True when this build has shaders compiled into it.
bool HasEmbeddedShaders();

// Build-time digest of every embedded shader's name and contents (0 if embedding is disabled).
uint64_t EmbeddedShaderCorpusHash();

}  // namespace apriltag_vulkan::vk
