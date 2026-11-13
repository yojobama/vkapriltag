#!/usr/bin/env python3
"""Checks that the library, the app and the tools that do not compare against upstream do not use libapriltag.

1. No source or header under library/, apps/, tests/ or tools/ includes an upstream header or names an upstream type,
   except the files that exist to compare against upstream (ALLOWED).
2. The vkapriltag target does not link the apriltag target.
"""
import argparse
import pathlib
import re
import sys

# Files that may use upstream: the differential test and the reference validation tools (S3 makes them optional).
ALLOWED = {
    "tests/own_decode_test.cpp",
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="the apriltags_vulkan directory")
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    failures = []
    checked = 0
    for d in ("library", "apps", "tests", "tools"):
        for path in (root / d).rglob("*"):
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
    for f in sorted(set(failures)):
        print("FAIL", f)
    print(f"checked {checked} files outside the upstream-comparing set ({len(ALLOWED)} files)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
