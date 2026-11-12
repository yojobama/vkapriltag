"""Renders the committed test subset: the first frames of chosen gen_corpus.py scenarios, as PNG.

usage: make_subset.py APRILTAG_SRC OUT_DIR [--per-scenario N]
The frames use gen_corpus.py's fixed seeds at 640x480, so they equal the first frames of the full
corpus. Each scenario gets OUT_DIR/<scenario>/ with PNG frames, list.txt (names) and gt.json.
"""
import argparse
import json
import os
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_corpus as gc

SCENARIOS = ["one-tag", "many-tags", "small-tags", "blur", "motion-blur", "grey-margin", "low-contrast",
             "border-occlusion", "clutter-only", "family16h5"]


def write_png(path, img):
    h, w = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)) +
                chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--per-scenario", type=int, default=5)
    a = ap.parse_args()
    w, h = 640, 480
    for name in SCENARIOS:
        family = "tag16h5" if name == "family16h5" else "tag36h11"
        cfg = {"tags": (1, 3), "bg": "clutter", "decoys": (1, 4)} if name == "family16h5" else gc.SCENARIOS[name]
        fam = gc.load_family(a.src, family)
        os.makedirs(os.path.join(a.out, name), exist_ok=True)
        names, truth = [], {}
        for idx in range(a.per_scenario):
            rng = np.random.default_rng(gc.seed_for(name, w, h, idx, family))
            img, tags = gc.render_frame(fam, cfg, w, h, rng)
            fname = f"{name}_{w}x{h}_{idx:03d}.png"
            write_png(os.path.join(a.out, name, fname), img)
            names.append(fname)
            truth[fname] = {"size": [w, h], "tags": tags, "decoys": 0}
        with open(os.path.join(a.out, name, "list.txt"), "w") as f:
            f.write("".join(n + "\n" for n in names))
        json.dump(truth, open(os.path.join(a.out, name, "gt.json"), "w"))
        print(f"{name}: {len(names)} frames")


if __name__ == "__main__":
    main()
