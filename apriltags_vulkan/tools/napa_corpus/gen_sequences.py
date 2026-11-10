"""Renders video-like AprilTag sequences with exact per-frame ground truth (reuses gen_corpus.py's renderer).

usage: gen_sequences.py APRILTAG_SRC OUT_DIR [--frames N] [--only NAME[,NAME...]] [--jobs J]

Each sequence gets OUT_DIR/<name>/ with its frames in order, list.txt and gt.json in gen_corpus.py's format
(a tag is listed only while its black square is inside the frame). Tags move on smooth Lissajous paths
whose peak speed is the sequence's `speed` in pixels per frame, or cross the frame once (enter/leave).
Motion blur averages 5 renders across half a frame time. The same seed always gives the same frames.
"""
import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_corpus as gc  # noqa: E402

# name -> configuration
SEQUENCES = {
    "slow-1tag-640": {"size": (640, 480), "tags": 1, "speed": 2.0},
    "slow-3tag-640": {"size": (640, 480), "tags": 3, "speed": 2.0},
    "slow-1tag-1280": {"size": (1280, 720), "tags": 1, "speed": 2.0},
    "medium-2tag-640": {"size": (640, 480), "tags": 2, "speed": 10.0},
    "medium-4tag-1280": {"size": (1280, 720), "tags": 4, "speed": 10.0, "bg": "clutter"},
    "fast-2tag-640": {"size": (640, 480), "tags": 2, "speed": 30.0, "blur": True},
    "fast-1tag-1280": {"size": (1280, 720), "tags": 1, "speed": 30.0, "blur": True},
    "enterleave-3tag-640": {"size": (640, 480), "tags": 3, "speed": 5.0, "cross": True},
    "enter-3tag-640": {"size": (640, 480), "tags": 3, "speed": 3.0, "enter": True},
    "noise-2tag-640": {"size": (640, 480), "tags": 2, "speed": 6.0, "noise": 8.0},
}


def tag_paths(cfg, rng, n_frames, w, h):
    """Per tag: (size, function frame -> (centre, theta, tilt, axis)); each tag keeps to its own grid cell."""
    tags = []
    n = cfg["tags"]
    cols, rows = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2)}[n]
    cw, ch = w / cols, h / rows
    side = 0.22 * h if w == 640 else 0.2 * h
    for i in range(n):
        size = min(side * float(rng.uniform(0.8, 1.2)), 0.55 * min(cw, ch))
        extent = 0.85 * size
        th0 = float(rng.uniform(0, 2 * math.pi))
        spin = float(rng.uniform(-0.012, 0.012))
        tilt0 = math.radians(float(rng.uniform(10, 30)))
        axis = float(rng.uniform(0, 2 * math.pi))
        if cfg.get("cross"):
            # a straight crossing in its own horizontal band, starting off-frame at a staggered time
            start = i * n_frames // (n + 1) * 0.5
            vx = cfg["speed"] * float(rng.choice([-1, 1]))
            vy = float(rng.uniform(-0.1, 0.1)) * cfg["speed"]
            y0 = (i + 0.5) * h / n
            x_start = -1.2 * extent if vx > 0 else w + 1.2 * extent

            def path(f, x_start=x_start, vx=vx, vy=vy, y0=y0, start=start, th0=th0, spin=spin, tilt0=tilt0, axis=axis):
                t = f - start
                return (x_start + vx * t, y0 + vy * t), th0 + spin * f, tilt0, axis

            tags.append((size, path))
            continue
        ax, ay = max(8.0, cw / 2 - extent), max(8.0, ch / 2 - extent)
        cx, cy = (i % cols + 0.5) * cw, (i // cols + 0.5) * ch
        if cfg.get("enter") and i > 0:
            # enters from the nearer side edge at a staggered time at 5 px/frame, then stays in its cell
            start = 40 + 60 * (i - 1)
            sign = 1 if cx < w / 2 else -1
            x_off = -1.2 * extent if sign > 0 else w + 1.2 * extent

            def path(f, cx=cx, cy=cy, x_off=x_off, sign=sign, start=start, th0=th0, spin=spin, tilt0=tilt0, axis=axis):
                x = x_off + sign * 5.0 * max(0.0, f - start)
                return ((min(x, cx) if sign > 0 else max(x, cx)), cy), th0 + spin * f, tilt0, axis

            tags.append((size, path))
            continue
        # peak speed = amplitude * omega, split between the axes
        sx, sy = float(rng.uniform(0.5, 1.0)), float(rng.uniform(0.5, 1.0))
        norm = math.hypot(sx, sy)
        wx, wy = cfg["speed"] * sx / norm / ax, cfg["speed"] * sy / norm / ay
        px, py = float(rng.uniform(0, 2 * math.pi)), float(rng.uniform(0, 2 * math.pi))

        def path(f, ax=ax, ay=ay, cx=cx, cy=cy, wx=wx, wy=wy, px=px, py=py, th0=th0, spin=spin, tilt0=tilt0, axis=axis):
            return ((cx + ax * math.sin(wx * f + px), cy + ay * math.sin(wy * f + py)), th0 + spin * f,
                    tilt0 * (1 + 0.3 * math.sin(0.03 * f + px)), axis)

        tags.append((size, path))
    return tags


def render(fam, cfg, tags, ids, bg, bw, f, w, h):
    img = bg.copy()
    gt = []
    for (size, path), tid in zip(tags, ids):
        centre, theta, tilt, axis = path(f)
        project, corners, outline = gc.pose_quad(fam, size, theta, tilt, axis, centre, w)
        gc.paint_tag(img, fam, gc.cells(fam, fam["codes"][tid]), project, outline, bw)
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        inside = min(xs) >= 0 and min(ys) >= 0 and max(xs) <= w and max(ys) <= h
        gt.append((inside, tid, corners, size))
    return img, gt


def job(args):
    src, out, name, cfg, n_frames = args
    fam = gc.load_family(src, "tag36h11")
    w, h = cfg["size"]
    rng = np.random.default_rng(gc.seed_for("sequence", name))
    base = float(rng.uniform(110, 190))
    bw = (float(rng.uniform(20, 45)), float(rng.uniform(200, 240)))
    bg = gc.background(rng, h, w, cfg.get("bg", "plain"), base)
    ids = [int(i) for i in rng.choice(len(fam["codes"]), cfg["tags"], replace=False)]
    tags = tag_paths(cfg, rng, n_frames, w, h)
    sigma = cfg.get("noise", 1.5)
    os.makedirs(os.path.join(out, name), exist_ok=True)
    results = {}
    for f in range(n_frames):
        if cfg.get("blur"):
            # five renders across half a frame time, averaged; the truth is the middle one's
            acc = None
            for j in range(5):
                img, gt = render(fam, cfg, tags, ids, bg, bw, f + (j - 2) * 0.125, w, h)
                if j == 2:
                    _, gt_mid = img, gt
                acc = img.copy() if acc is None else acc + img
            img = acc / 5
            gt = gt_mid
        else:
            img, gt = render(fam, cfg, tags, ids, bg, bw, f, w, h)
        fr = np.random.default_rng(gc.seed_for("noise", name, f))
        if sigma > 0:
            img = img + fr.normal(0, sigma, img.shape).astype(np.float32)
        img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
        fname = f"{name}_{f:03d}.pgm"
        with open(os.path.join(out, name, fname), "wb") as fh:
            fh.write(b"P5\n%d %d\n255\n" % (w, h))
            fh.write(img.tobytes())
        results[fname] = {"size": [w, h], "decoys": 0,
                          "tags": [{"id": tid, "corners": [[round(x, 4), round(y, 4)] for x, y in c],
                                    "size_px": round(size, 2)} for inside, tid, c, size in gt if inside]}
    names = sorted(results)
    with open(os.path.join(out, name, "list.txt"), "w") as fh:
        for n in names:
            fh.write(os.path.abspath(os.path.join(out, name, n)).replace("\\", "/") + "\n")
    json.dump(results, open(os.path.join(out, name, "gt.json"), "w"))
    return name, n_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--only", default="")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    jobs = [(a.src, a.out, n, c, a.frames) for n, c in SEQUENCES.items() if not only or n in only]
    with ProcessPoolExecutor(min(a.jobs, len(jobs))) as ex:
        for name, n in ex.map(job, jobs):
            print(f"{name}: {n} frames", flush=True)


if __name__ == "__main__":
    main()
