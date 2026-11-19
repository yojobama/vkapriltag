#!/usr/bin/env python3
"""Installs the build to a temporary prefix and checks that the package has no libapriltag dependency, then builds
tests/consumer against it with find_package(vkapriltag) and runs a detection on a corpus frame.

1. The installed tree holds no upstream library, header or CMake package, and the installed CMake files name no
   apriltag package besides vkapriltag.
2. tests/consumer configures, builds and runs; it must print the expected tag id.
3. The consumer executable's dynamic dependencies (dumpbin /dependents or ldd, when available) name no apriltag library.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

UPSTREAM_FILE = re.compile(
    r"^(lib)?apriltag(\.|-|_pose|config|targets)|^tag(16h5|25h9|36h11|circle21h7|circle49h12|standard41h12|standard52h13|custom48h12)\.h$",
    re.IGNORECASE)
APRILTAG_WORD = re.compile(r"(?<![A-Za-z])apriltag(?![A-Za-z_])", re.IGNORECASE)


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmake", required=True)
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--config", default="Release")
    ap.add_argument("--generator", required=True)
    ap.add_argument("--make-program", default="")
    ap.add_argument("--cxx", default="")
    ap.add_argument("--consumer", required=True, help="the tests/consumer directory")
    ap.add_argument("--stb-dir", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--expect-id", type=int, required=True)
    args = ap.parse_args()

    work = tempfile.mkdtemp(prefix="vkapriltag-consumer-")
    prefix = os.path.join(work, "prefix")
    failures = []
    try:
        run([args.cmake, "--install", args.build_dir, "--prefix", prefix, "--config", args.config], stdout=subprocess.DEVNULL)

        installed = []
        for dirpath, _, files in os.walk(prefix):
            for f in files:
                installed.append(os.path.relpath(os.path.join(dirpath, f), prefix).replace("\\", "/"))
        for rel in installed:
            parts = rel.split("/")
            if UPSTREAM_FILE.search(parts[-1]) or any(p.lower() == "apriltag" for p in parts[:-1]):
                failures.append(f"installed file from upstream libapriltag: {rel}")
            if rel.endswith(".cmake"):
                text = open(os.path.join(prefix, rel), encoding="utf-8", errors="replace").read()
                for n, line in enumerate(text.splitlines(), 1):
                    if APRILTAG_WORD.search(line) and "vkapriltag" not in line:
                        failures.append(f"{rel}:{n}: names an apriltag package: {line.strip()}")
        print(f"installed {len(installed)} files")
        if not any(p.endswith("vkapriltagConfig.cmake") for p in installed):
            failures.append("vkapriltagConfig.cmake was not installed")

        build = os.path.join(work, "consumer-build")
        cfg = [args.cmake, "-S", args.consumer, "-B", build, "-G", args.generator, f"-DCMAKE_PREFIX_PATH={prefix}",
               f"-DCMAKE_BUILD_TYPE={args.config}", f"-DSTB_DIR={args.stb_dir}"]
        if args.make_program:
            cfg.append(f"-DCMAKE_MAKE_PROGRAM={args.make_program}")
        if args.cxx:
            cfg.append(f"-DCMAKE_CXX_COMPILER={args.cxx}")
        run(cfg)
        run([args.cmake, "--build", build, "--config", args.config])
        exe = None
        for dirpath, _, files in os.walk(build):
            for f in files:
                if f in ("consumer", "consumer.exe"):
                    exe = os.path.join(dirpath, f)
        if exe is None:
            failures.append("the consumer was not built")
        else:
            out = subprocess.run([exe, args.image], capture_output=True, text=True)
            print(out.stdout, out.stderr)
            if out.returncode != 0 or not re.search(rf"^id {args.expect_id} hamming", out.stdout, re.M):
                failures.append(f"the consumer did not detect tag {args.expect_id} (exit code {out.returncode})")
            tool = shutil.which("dumpbin") or shutil.which("ldd")
            if tool:
                cmd = [tool, "/dependents", exe] if os.path.basename(tool).startswith("dumpbin") else [tool, exe]
                deps = subprocess.run(cmd, capture_output=True, text=True).stdout
                bad = [line.strip() for line in deps.splitlines() if re.search(r"apriltag", line, re.I) and "consumer" not in line.lower()]
                print("dependencies checked with", os.path.basename(tool), "-", "none from apriltag" if not bad else bad)
                failures += [f"the consumer depends on {b}" for b in bad]
            else:
                print("dumpbin and ldd not found: dynamic dependencies not checked")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    for f in failures:
        print("FAIL", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
