// SWIG interface of the public Detector (include/vkapriltag/Detector.h): Python is built and tested, Java and C#
// are generated from the same file.
%module vkapriltag

%{
#include "vkapriltag/Detector.h"
%}

%include <stdint.i>
%include <std_string.i>
%include <std_vector.i>
%include <std_shared_ptr.i>
%include <exception.i>

// Every library error becomes the target language's closest exception.
%exception {
  try {
    $action
  } catch (const apriltag_vulkan::Error &e) {
    switch (e.code()) {
      case apriltag_vulkan::ErrorCode::kInvalidArgument: SWIG_exception(SWIG_ValueError, e.what());
      case apriltag_vulkan::ErrorCode::kDevice: SWIG_exception(SWIG_IOError, e.what());
      default: SWIG_exception(SWIG_RuntimeError, e.what());
    }
  } catch (const std::exception &e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

%shared_ptr(apriltag_vulkan::Device)

// Raw pointers, internals and fixed arrays are reached through the accessors below instead.
%ignore apriltag_vulkan::Family;
%ignore apriltag_vulkan::Error;
%ignore apriltag_vulkan::Detection::family;
%ignore apriltag_vulkan::Detection::H;
%ignore apriltag_vulkan::Detection::c;
%ignore apriltag_vulkan::Detection::p;
%ignore apriltag_vulkan::TagPose::R;
%ignore apriltag_vulkan::TagPose::t;
%ignore apriltag_vulkan::Device::impl;
%ignore apriltag_vulkan::Device::Impl;
%ignore apriltag_vulkan::Detector::Impl;
%ignore apriltag_vulkan::Detector::Detect;
%ignore apriltag_vulkan::Detector::ImportHostFrame;
%ignore apriltag_vulkan::Detector::ImportDmaBufFrame;

// A result's members are copied out, so they stay valid after the result itself is dropped.
%naturalvar apriltag_vulkan::Result::detections;
%naturalvar apriltag_vulkan::Result::poses;
%naturalvar apriltag_vulkan::Result::timing;

%template(StringVector) std::vector<std::string>;
%template(LevelVector) std::vector<apriltag_vulkan::Level>;
%template(DetectionVector) std::vector<apriltag_vulkan::Detection>;
%template(PoseVector) std::vector<apriltag_vulkan::TagPose>;
%template(ByteVector) std::vector<uint8_t>;

%extend apriltag_vulkan::Detection {
  // The tag family's name.
  std::string family_name() const { return $self->family != nullptr ? $self->family->name : std::string(); }
  // The centre in pixels.
  double centre_x() const { return $self->c[0]; }
  double centre_y() const { return $self->c[1]; }
  // Corner i (0 to 3) in pixels.
  double corner_x(int i) const { return $self->p[i & 3][0]; }
  double corner_y(int i) const { return $self->p[i & 3][1]; }
  // Element i (0 to 8, row-major) of the homography from tag coordinates to the image.
  double homography(int i) const { return $self->H[i % 9]; }
}

%extend apriltag_vulkan::TagPose {
  // Element (i, j) of the rotation and element i of the translation.
  double rotation(int i, int j) const { return $self->R[i % 3][j % 3]; }
  double translation(int i) const { return $self->t[i % 3]; }
}

%extend apriltag_vulkan::Detector {
  // Detects the tags of a frame given as a list of bytes (width * height of them, or rows `stride` bytes apart).
  // The result is a copy that stays valid.
  apriltag_vulkan::Result DetectBytes(const std::vector<uint8_t> &frame, size_t stride = 0) {
    const size_t row = stride != 0 ? stride : $self->width();
    if (frame.size() < (static_cast<size_t>($self->height()) - 1) * row + $self->width()) {
      throw apriltag_vulkan::Error(apriltag_vulkan::ErrorCode::kInvalidArgument, "the frame is smaller than width * height");
    }
    return $self->Detect(frame.data(), stride);
  }
#ifdef SWIGPYTHON
  // Detects the tags of any contiguous buffer (bytes, bytearray, memoryview, a numpy array) without copying it.
  apriltag_vulkan::Result DetectBuffer(const uint8_t *data, size_t length, size_t stride = 0) {
    const size_t row = stride != 0 ? stride : $self->width();
    if (length < (static_cast<size_t>($self->height()) - 1) * row + $self->width()) {
      throw apriltag_vulkan::Error(apriltag_vulkan::ErrorCode::kInvalidArgument, "the frame is smaller than width * height");
    }
    return $self->Detect(data, stride);
  }
#endif
}

#ifdef SWIGPYTHON
%typemap(in) (const uint8_t *data, size_t length) (Py_buffer view) {
  if (PyObject_GetBuffer($input, &view, PyBUF_C_CONTIGUOUS) != 0) {
    SWIG_exception_fail(SWIG_TypeError, "a contiguous buffer (bytes, bytearray, memoryview or numpy array) is expected");
  }
  $1 = ($1_ltype)view.buf;
  $2 = static_cast<size_t>(view.len);
}
%typecheck(SWIG_TYPECHECK_POINTER) (const uint8_t *data, size_t length) {
  $1 = PyObject_CheckBuffer($input) ? 1 : 0;
}
%typemap(freearg) (const uint8_t *data, size_t length) {
  PyBuffer_Release(&view$argnum);
}
#endif

%include "vkapriltag/Version.h"
%include "vkapriltag/Pose.h"
%include "vkapriltag/Detection.h"
%include "vkapriltag/Detector.h"
