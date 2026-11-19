#!/usr/bin/env python3
"""Checks every relative link in the repository's Markdown files.

A link is `[text](target)` or a reference definition `[label]: target`. External links (a scheme such as
https:) are skipped; a relative target must name an existing file or directory, and a `#fragment` on a
Markdown target must match a heading of that file (GitHub's slug rules). Links inside code spans and
fenced code blocks are ignored. Exits non-zero if any link is broken.

usage: check_links.py [ROOT]    (default: the repository root, two levels above this script)
"""
import pathlib
import re
import sys
import urllib.parse

INLINE = re.compile(r"(?<!\\)\[(?:[^\[\]]|\[[^\]]*\])*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
REFDEF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*<?(\S+?)>?(?:\s+.*)?$")
HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
SKIP_DIRS = {".git", "node_modules", "build", "_deps"}


def slug(text):
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text).strip().lower()
    text = re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE)
    return text.replace(" ", "-")


def headings(path, cache={}):
    if path not in cache:
        found, counts, fence = set(), {}, False
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lstrip().startswith(("```", "~~~")):
                fence = not fence
                continue
            m = None if fence else HEADING.match(line)
            if m:
                base = slug(m.group(2))
                n = counts.get(base, 0)
                counts[base] = n + 1
                found.add(base if n == 0 else f"{base}-{n}")
        cache[path] = found
    return cache[path]


def links(path):
    fence = False
    for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if line.lstrip().startswith(("```", "~~~")):
            fence = not fence
            continue
        if fence:
            continue
        stripped = re.sub(r"`[^`]*`", "", line)
        for m in INLINE.finditer(stripped):
            yield number, m.group(1)
        m = REFDEF.match(stripped)
        if m:
            yield number, m.group(1)


def main():
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[2]
    root = root.resolve()
    files = [p for p in root.rglob("*.md") if not (set(p.relative_to(root).parts) & SKIP_DIRS)]
    broken = checked = 0
    for path in sorted(files):
        for number, target in links(path):
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("//"):
                continue
            checked += 1
            raw, _, fragment = target.partition("#")
            dest = path if raw == "" else (path.parent / urllib.parse.unquote(raw)).resolve()
            where = f"{path.relative_to(root).as_posix()}:{number}"
            if not dest.exists():
                print(f"BROKEN {where}: {target} (no such file)")
                broken += 1
            elif fragment and dest.suffix == ".md" and urllib.parse.unquote(fragment).lower() not in headings(dest):
                print(f"BROKEN {where}: {target} (no such heading)")
                broken += 1
    print(f"checked {checked} relative links in {len(files)} Markdown files, {broken} broken")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
