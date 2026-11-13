#!/usr/bin/env python3
"""Checks that the library, the app and the tools that do not compare against upstream do not use libapriltag.

1. No source or header under library/, apps/, tests/ or tools/ includes an upstream header or names an upstream type,
   except the files that exist to compare against upstream (ALLOWED).
2. The vkapriltag target does not link the apriltag target.
3. Nothing fetches or finds upstream apriltag outside tools/upstream/CMakeLists.txt, which the build reaches only
   under VKAPRILTAG_UPSTREAM_VALIDATION (default OFF); no patch file or find_dependency of apriltag exists elsewhere.
"""
import argparse
import pathlib
import re
import sys

# Files that may use upstream: the vector generator and the reference validation tools.
ALLOWED = {
    "tools/upstream/gen_decode_vectors.cpp",
    "tools/upstream/UpstreamAdapter.h",
    "tools/vk_eval/vk_eval_ref.cpp",
    "tools/validate_against_libapriltag/validate_common.h",
    "tools/validate_against_libapriltag/validate_against_libapriltag.cpp",
    "tools/validate_against_libapriltag/validate_against_libapriltag_opencv.cpp",
    "tools/validate_pose/validate_pose.cpp",
    "tools/validate_pose_e2e/validate_pose_e2e.cpp",
    "tools/synth_pose_ground_truth/synth_pose_ground_truth.cpp",
}

INCLUDE = re.compile(
    r'#\s*include\s*[<"](apriltag(_pose)?\.h|tag(16h5|25h9|36h11|Circle21h7|Circle49h12|Standard41h12|Standard52h13|Custom48h12)\.h'
    r'|common/(zarray|matd|image_u8|g2d|homography|pjpeg|unionfind|workerpool|math_util|pam|svd22)\.h)[>"]')
TYPES = re.compile(r"\b(apriltag_(detector|family|detection|pose|detection_info)_t|zarray_t|matd_t|image_u8_t)\b|\bstruct quad\b")
SUFFIXES = {".h", ".hpp", ".cpp", ".cc", ".c"}


FETCH = re.compile(r"FetchContent_(Declare|MakeAvailable|Populate)\(\s*apriltag\b|AprilRobotics/apriltag|find_(package|dependency)\(\s*apriltag\b", re.I)


def walk(base, pattern):
    """Files under base matching pattern, skipping untracked top-level build and out directories."""
    return [p for p in base.rglob(pattern) if p.is_file() and p.relative_to(base).parts[:1] not in (("build",), ("out",))]


def check_cmake(root):
    """The default configure must not reach upstream."""
    failures = []
    allowed = {"tools/upstream/CMakeLists.txt"}
    for path in list(walk(root, "CMakeLists.txt")) + list(walk(root, "*.cmake")) + list(walk(root, "*.cmake.in")):
        rel = path.relative_to(root).as_posix()
        if rel in allowed or "tests/corpus" in rel:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if FETCH.search(line):
                failures.append(f"{rel}:{n}: fetches or finds upstream apriltag outside tools/upstream: {line.strip()}")
    for path in walk(root, "*.patch"):
        rel = path.relative_to(root).as_posix()
        if not rel.startswith("tools/upstream/patches/"):
            failures.append(f"{rel}: patch files belong under tools/upstream/patches")
    top = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    if not re.search(r'option\(VKAPRILTAG_UPSTREAM_VALIDATION\b[^)]*\bOFF\)', top):
        failures.append("CMakeLists.txt: VKAPRILTAG_UPSTREAM_VALIDATION must default to OFF")
    tools = (root / "tools" / "CMakeLists.txt").read_text(encoding="utf-8")
    if not re.search(r"if\(VKAPRILTAG_UPSTREAM_VALIDATION\)\s*add_subdirectory\(upstream\)\s*endif\(\)", tools):
        failures.append("tools/CMakeLists.txt: add_subdirectory(upstream) must sit directly under if(VKAPRILTAG_UPSTREAM_VALIDATION)")
    for path in walk(root, "*"):
        if path.is_file() and path.name == "CMakeLists.txt" and "tests/corpus" not in path.as_posix():
            text = path.read_text(encoding="utf-8", errors="replace")
            if path.relative_to(root).as_posix() != "tools/upstream/CMakeLists.txt" and re.search(r"ApplyPatches|PATCH_COMMAND", text):
                failures.append(f"{path.relative_to(root).as_posix()}: applies patches outside tools/upstream")
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="the apriltags_vulkan directory")
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    failures = []
    checked = 0
    for d in ("library", "apps", "tests", "tools"):
        for path in walk(root / d, "*"):
            if path.suffix not in SUFFIXES or "stb_image" in path.name:
                continue
            rel = path.relative_to(root).as_posix()
            if rel in ALLOWED:
                continue
            checked += 1
            for n, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if INCLUDE.search(line) or TYPES.search(line):
                    failures.append(f"{rel}:{n}: uses upstream libapriltag: {line.strip()}")
    cmake = (root / "library" / "CMakeLists.txt").read_text(encoding="utf-8")
    for block in re.finditer(r"target_link_libraries\(\s*vkapriltag\b[^)]*\)", cmake):
        if re.search(r"\bapriltag\b", block.group(0).replace("vkapriltag", "")):
            failures.append("library/CMakeLists.txt: the vkapriltag target links apriltag")
    failures += check_cmake(root)
    for f in sorted(set(failures)):
        print("FAIL", f)
    print(f"checked {checked} files outside the upstream-comparing set ({len(ALLOWED)} files)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
