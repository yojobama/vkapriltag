# SWIG bindings

`vkapriltag.i` wraps the public `Detector` (`include/vkapriltag/Detector.h`) for other languages. Python is built and
tested; Java and C# are generated from the same file.

## Python

    cmake -S apriltags_vulkan -B build -DVKAPRILTAG_BUILD_SWIG=ON
    cmake --build build --target vkapriltag_python
    ctest --test-dir build -R swig_python

The module is `build/bindings/swig/python/vkapriltag.py` with its `_vkapriltag*` library next to it. It needs SWIG 4.1 or
later and the Python development files.

    import vkapriltag as vk

    options = vk.DetectorOptions()
    options.width, options.height = 1280, 720
    options.families = vk.StringVector(["tag36h11"])
    detector = vk.Detector(options)

    result = detector.DetectBuffer(frame)          # bytes, bytearray, memoryview or a numpy uint8 array (no copy)
    for tag in result.detections:
        print(tag.id, tag.family_name(), tag.centre_x(), tag.centre_y(), tag.corner_x(0), tag.corner_y(0))

- `DetectBuffer(frame, stride=0)` reads any contiguous buffer; `DetectBytes(vk.ByteVector(...), stride=0)` takes a
  list of bytes and is the form the other languages use. Both return a copy of the result that stays valid.
- Detections, poses and timings are copied out of a result, so they stay valid after the result is dropped. Assign a
  `DetectorOptions` member such as `options.intrinsics` through the options object (`options.intrinsics.fx = 600`).
- The Python call releases the GIL while it runs.
- Errors become exceptions: an invalid option is a `ValueError`, a Vulkan failure an `IOError`, anything else a
  `RuntimeError`, each with the library's message.
- Enumerations are `vk.Mode_kSync`, `vk.Mode_kPipelined` and `vk.Mode_kTracking`.

## Java and C#

    swig -c++ -java   -I<repo>/apriltags_vulkan/library/include -outdir out vkapriltag.i
    swig -c++ -csharp -I<repo>/apriltags_vulkan/library/include -outdir out vkapriltag.i

This writes the wrapper source and the proxy classes; build the wrapper with the static `vkapriltag` library into the
shared library your runtime loads. Use `DetectBytes` for frames (the buffer form is Python only). These two targets are
generated in the project's checks but not compiled or run there.
