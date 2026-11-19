#!/usr/bin/env python3
"""Checks that the options registry is the only place that reads the environment.

1. No getenv("APRILTAG_...") outside library/src/common/Options.cpp, and no getenv at all in the library.
2. Every APRILTAG_VK_* / APRILTAG_CPU_* name mentioned in the sources is a registry entry.
3. The switch reference in PERFORMANCE.md equals what `vk_eval --list-options` prints.
"""
import argparse
import pathlib
import re
import subprocess
import sys

BEGIN = "<!-- options:begin -->"
END = "<!-- options:end -->"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="the apriltags_vulkan directory")
    ap.add_argument("--vk-eval", required=True)
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    registry = (root / "library" / "src" / "common" / "Options.cpp").resolve()

    names = set(subprocess.run([args.vk_eval, "--list-option-names"], capture_output=True, text=True,
                               check=True).stdout.split())
    reference = subprocess.run([args.vk_eval, "--list-options"], capture_output=True, text=True,
                               check=True).stdout.replace("\r\n", "\n")
    failures = []

    sources = [p for d in ("library", "apps", "tools") for p in (root / d).rglob("*")
               if p.suffix in (".cpp", ".h", ".hpp", ".mm") and p.resolve() != registry]
    for path in sources:
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(root).as_posix()
        for m in re.finditer(r'getenv\s*\(\s*"APRILTAG_', text):
            failures.append(f"{rel}: reads an APRILTAG_ variable outside the registry")
        if rel.startswith("library/") and re.search(r"\bgetenv\s*\(", text):
            failures.append(f"{rel}: getenv outside the registry")
        for m in re.finditer(r"\bAPRILTAG_(?:VK|CPU)_[A-Z0-9_]*[A-Z0-9]", text):
            if m.group(0) not in names:
                failures.append(f"{rel}: {m.group(0)} is not in the registry")

    doc = (root / "PERFORMANCE.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    if BEGIN not in doc or END not in doc:
        failures.append("PERFORMANCE.md: options markers missing")
    else:
        block = doc.split(BEGIN, 1)[1].split(END, 1)[0].strip("\n")
        if block != reference.strip("\n"):
            failures.append("PERFORMANCE.md: switch reference is out of date; regenerate it with "
                            "`vk_eval --list-options` between the options markers")

    for f in sorted(set(failures)):
        print("FAIL", f)
    print(f"checked {len(sources)} sources against {len(names)} registered variables")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
