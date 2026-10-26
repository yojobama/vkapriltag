#pragma once

// Captures chosen frames with RenderDoc's in-application API when the tool runs under RenderDoc
// (built with VKAPRILTAG_HAVE_RENDERDOC); otherwise every call does nothing.

#include <cstdint>
#include <cstdio>
#include <string>

#ifdef VKAPRILTAG_HAVE_RENDERDOC
#include <renderdoc_app.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

namespace apriltag_vulkan::validate {

class RenderDocCapture {
 public:
  // `frame` < 0 disables; `path_template` empty keeps RenderDoc's default.
  RenderDocCapture(int frame, const std::string &path_template) : frame_(frame) {
#ifdef VKAPRILTAG_HAVE_RENDERDOC
    if (frame_ < 0) return;
    pRENDERDOC_GetAPI get = nullptr;
#ifdef _WIN32
    if (HMODULE m = GetModuleHandleA("renderdoc.dll")) {
      get = reinterpret_cast<pRENDERDOC_GetAPI>(GetProcAddress(m, "RENDERDOC_GetAPI"));
    }
#else
    if (void *m = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD)) {
      get = reinterpret_cast<pRENDERDOC_GetAPI>(dlsym(m, "RENDERDOC_GetAPI"));
    }
#endif
    if (get == nullptr || get(eRENDERDOC_API_Version_1_6_0, reinterpret_cast<void **>(&api_)) != 1) {
      api_ = nullptr;
      std::fprintf(stderr, "--renderdoc-capture: not running under RenderDoc; no capture taken\n");
      return;
    }
    if (!path_template.empty()) api_->SetCaptureFilePathTemplate(path_template.c_str());
#else
    if (frame_ >= 0) {
      std::fprintf(stderr, "--renderdoc-capture: built without renderdoc_app.h; no capture taken\n");
    }
#endif
  }

  void BeforeFrame(int index) {
#ifdef VKAPRILTAG_HAVE_RENDERDOC
    if (api_ != nullptr && index == frame_) api_->StartFrameCapture(nullptr, nullptr);
#else
    (void)index;
#endif
  }

  void AfterFrame(int index) {
#ifdef VKAPRILTAG_HAVE_RENDERDOC
    if (api_ != nullptr && index == frame_) {
      std::printf("RenderDoc capture of frame %d: %s\n", index,
                  api_->EndFrameCapture(nullptr, nullptr) == 1 ? "written" : "failed");
    }
#else
    (void)index;
#endif
  }

 private:
  int frame_;
#ifdef VKAPRILTAG_HAVE_RENDERDOC
  RENDERDOC_API_1_6_0 *api_ = nullptr;
#endif
};

}  // namespace apriltag_vulkan::validate
