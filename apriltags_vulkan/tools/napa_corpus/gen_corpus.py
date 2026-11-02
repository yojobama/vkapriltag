"""Renders a labelled AprilTag test corpus (numpy only; frames are written as binary PGM).

usage: gen_corpus.py APRILTAG_SRC OUT_DIR [--count N] [--sizes 640x480,1280x720,1920x1080] [--jobs J]
                     [--only SCENARIO[,SCENARIO...]]

APRILTAG_SRC is a directory holding tag36h11.c and tag16h5.c (the fetched libapriltag source).
Each scenario gets OUT_DIR/<scenario>/ with its frames, list.txt (the frames in order) and gt.json
(per frame: the true corners of every tag, in libapriltag's winding, and the decoy count).

Every pixel is the mean of a 4x4 grid of point samples, so the corner truth is exact. Pixel i covers
[i, i + 1) and corners are in those coordinates. The same seed always gives the same frames.
"""
import argparse
import json
import math
import os
import re
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SS = 4  # supersampling per axis


def load_family(src, name):
    text = open(os.path.join(src, name + ".c")).read()
    codes = [int(c, 16) for c in re.findall(r"0x([0-9a-fA-F]+)UL", text)]
    bx = {int(i): int(v) for i, v in re.findall(r"bit_x\[(\d+)\] = (\d+);", text)}
    by = {int(i): int(v) for i, v in re.findall(r"bit_y\[(\d+)\] = (\d+);", text)}
    width = int(re.search(r"width_at_border = (\d+);", text).group(1))
    total = int(re.search(r"total_width = (\d+);", text).group(1))
    nbits = int(re.search(r"nbits = (\d+);", text).group(1))
    return {"codes": codes[: int(re.search(r"ncodes = (\d+);", text).group(1))], "bx": bx, "by": by,
            "width": width, "total": total, "nbits": nbits}


def cells(fam, code):
    """The family's total_width^2 cell grid for a code: 1 white, 0 black, margin ring white."""
    t, w = fam["total"], fam["width"]
    m = (t - w) // 2
    g = np.zeros((t, t), np.float32)
    g[:m, :] = g[-m:, :] = g[:, :m] = g[:, -m:] = 1.0
    for i in range(fam["nbits"]):
        if (code >> (fam["nbits"] - 1 - i)) & 1:
            g[fam["by"][i] + m, fam["bx"][i] + m] = 1.0
    return g


def random_cells(fam, rng):
    """A decoy: a black border and quiet zone around random payload bits."""
    code = int(rng.integers(0, 1 << fam["nbits"]))
    return cells(fam, code)


def homography(src, dst):
    a = []
    for (x, y), (u, v) in zip(src, dst):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.array(a, float))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


def pose_quad(fam, size_px, theta, tilt, tilt_axis, centre, fw):
    """Texture-cell corner positions (4, 2) -> pixel positions for the texture's outline, and the
    black square's corners in libapriltag's winding (bottom-left, bottom-right, top-right, top-left)."""
    t, w = fam["total"], fam["width"]
    f = float(fw)
    cell = 1.0 / w  # the black square is one world unit wide
    d = f / size_px

    def project(px, py):  # texture cell coordinates -> pixels
        x, y = (px - t / 2) * cell, (py - t / 2) * cell
        c, s = math.cos(theta), math.sin(theta)
        x, y = c * x - s * y, s * x + c * y
        ca, sa = math.cos(tilt_axis), math.sin(tilt_axis)
        ex, ey = ca * x + sa * y, -sa * x + ca * y  # along and across the tilt axis
        ex, z = math.cos(tilt) * ex, math.sin(tilt) * ex
        x, y = ca * ex - sa * ey, sa * ex + ca * ey
        zz = d + z
        return f * x / zz + centre[0], f * y / zz + centre[1]

    m = (t - w) / 2
    corners = [project(m, m + w), project(m + w, m + w), project(m + w, m), project(m, m)]
    outline = [project(0, 0), project(t, 0), project(t, t), project(0, t)]
    return project, corners, outline


def paint_tag(img, fam, grid, project, outline, levels):
    """Paints one tag into img (float32, in place) with 4x4 supersampling over its bounding box."""
    t = fam["total"]
    h, w = img.shape
    xs = [p[0] for p in outline]
    ys = [p[1] for p in outline]
    x0, x1 = max(0, int(math.floor(min(xs))) - 1), min(w, int(math.ceil(max(xs))) + 1)
    y0, y1 = max(0, int(math.floor(min(ys))) - 1), min(h, int(math.ceil(max(ys))) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    tex_to_px = homography([(0, 0), (t, 0), (t, t), (0, t)], outline)
    inv = np.linalg.inv(tex_to_px)
    n = (y1 - y0) * SS, (x1 - x0) * SS
    gy, gx = np.mgrid[0:n[0], 0:n[1]]
    px = x0 + (gx + 0.5) / SS
    py = y0 + (gy + 0.5) / SS
    den = inv[2, 0] * px + inv[2, 1] * py + inv[2, 2]
    u = (inv[0, 0] * px + inv[0, 1] * py + inv[0, 2]) / den
    v = (inv[1, 0] * px + inv[1, 1] * py + inv[1, 2]) / den
    inside = (u >= 0) & (u < t) & (v >= 0) & (v < t) & (den > 0)
    ui = np.clip(u.astype(np.int64), 0, t - 1)
    vi = np.clip(v.astype(np.int64), 0, t - 1)
    bit = grid[vi, ui]
    value = levels[0] + (levels[1] - levels[0]) * bit
    patch = img[y0:y1, x0:x1]
    sup = np.repeat(np.repeat(patch, SS, axis=0), SS, axis=1)
    sup = np.where(inside, value, sup)
    img[y0:y1, x0:x1] = sup.reshape(y1 - y0, SS, x1 - x0, SS).mean(axis=(1, 3))


def bar(img, p0, p1, width, level):
    """Paints a bar of the given width (pixels) between two points."""
    h, w = img.shape
    x0, x1 = int(max(0, min(p0[0], p1[0]) - width)), int(min(w, max(p0[0], p1[0]) + width + 1))
    y0, y1 = int(max(0, min(p0[1], p1[1]) - width)), int(min(h, max(p0[1], p1[1]) + width + 1))
    if x1 <= x0 or y1 <= y0:
        return 0
    gy, gx = np.mgrid[y0:y1, x0:x1]
    d = np.array(p1) - np.array(p0)
    t = np.clip(((gx + 0.5 - p0[0]) * d[0] + (gy + 0.5 - p0[1]) * d[1]) / (d @ d), 0, 1)
    dist = np.hypot(gx + 0.5 - (p0[0] + t * d[0]), gy + 0.5 - (p0[1] + t * d[1]))
    cover = np.clip(width / 2 + 0.5 - dist, 0, 1)
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - cover) + level * cover
    return 1


def smooth_noise(rng, h, w, scale):
    small = rng.random((max(2, h // scale + 2), max(2, w // scale + 2))).astype(np.float32)
    ys = np.linspace(0, small.shape[0] - 1.001, h)
    xs = np.linspace(0, small.shape[1] - 1.001, w)
    y0, x0 = ys.astype(int), xs.astype(int)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    a = small[y0][:, x0] * (1 - fx) + small[y0][:, x0 + 1] * fx
    b = small[y0 + 1][:, x0] * (1 - fx) + small[y0 + 1][:, x0 + 1] * fx
    return a * (1 - fy) + b * fy


def background(rng, h, w, kind, base):
    img = np.full((h, w), base, np.float32)
    if kind == "plain":
        return img
    if kind == "natural":
        return img + (smooth_noise(rng, h, w, 24) - 0.5) * 90 + (smooth_noise(rng, h, w, 6) - 0.5) * 30
    # clutter: rectangles, bars and checker patches
    for _ in range(int(rng.integers(8, 30))):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        bw, bh = int(rng.integers(4, w // 4)), int(rng.integers(4, h // 4))
        level = float(rng.integers(10, 245))
        if rng.random() < 0.3:  # checker patch
            c = int(rng.integers(3, 14))
            gy, gx = np.mgrid[y:min(h, y + bh), x:min(w, x + bw)]
            img[y:y + bh, x:x + bw] = np.where(((gx // c) + (gy // c)) % 2 == 0, level, 255 - level)[:bh, :bw]
        else:
            img[y:y + bh, x:x + bw] = level
    return img


def blur(img, sigma):
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    for axis in (0, 1):
        pad = np.pad(img, [(r, r) if a == axis else (0, 0) for a in (0, 1)], mode="edge")
        out = np.zeros_like(img)
        for i, wgt in enumerate(k):
            sl = [slice(None), slice(None)]
            sl[axis] = slice(i, i + img.shape[axis])
            out += wgt * pad[tuple(sl)]
        img = out
    return img


def motion_blur(img, length, angle):
    pad = length
    p = np.pad(img, pad, mode="edge")
    out = np.zeros_like(img)
    ts = np.linspace(-(length - 1) / 2, (length - 1) / 2, length * 2 + 1)
    for t in ts:
        dx, dy = int(round(t * math.cos(angle))), int(round(t * math.sin(angle)))
        out += p[pad + dy:pad + dy + img.shape[0], pad + dx:pad + dx + img.shape[1]]
    return out / len(ts)


def jpeg_like(img, quality):
    """8x8 block DCT quantisation with the JPEG luminance table, as a stand-in for a codec."""
    q = np.array([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], np.float32)
    s = 5000 / quality if quality < 50 else 200 - 2 * quality
    q = np.maximum(1, np.floor((q * s + 50) / 100))
    n = np.arange(8)
    c = np.cos((2 * n[None, :] + 1) * n[:, None] * math.pi / 16) * np.where(n[:, None] == 0, math.sqrt(1 / 8), math.sqrt(2 / 8))
    h, w = img.shape
    hh, ww = h // 8 * 8, w // 8 * 8
    blocks = (img[:hh, :ww] - 128).reshape(hh // 8, 8, ww // 8, 8).transpose(0, 2, 1, 3)
    coef = np.einsum("ij,abjk,lk->abil", c, blocks, c)
    coef = np.round(coef / q) * q
    rec = np.einsum("ji,abjk,kl->abil", c, coef, c)
    out = img.copy()
    out[:hh, :ww] = rec.transpose(0, 2, 1, 3).reshape(hh, ww) + 128
    return out


# name -> (positives, config); config keys are read in render_frame
SCENARIOS = {
    "empty": {"tags": (0, 0), "bg": "plain"},
    "clutter-only": {"tags": (0, 0), "bg": "clutter"},
    "decoys": {"tags": (0, 0), "bg": "plain", "decoys": (2, 6)},
    "one-tag": {"tags": (1, 1), "bg": "plain"},
    "many-tags": {"tags": (2, 12), "bg": "plain", "size": (0.35, 0.8)},
    "dense-tags": {"tags": (20, 40), "bg": "plain", "size_abs": (36, 60)},
    "small-tags": {"tags": (1, 4), "bg": "plain", "size_abs": (32, 53)},
    "large-tags": {"tags": (1, 1), "bg": "plain", "size_frac": (0.45, 0.8)},
    "steep-angle": {"tags": (1, 3), "bg": "plain", "tilt": (40, 65)},
    "blur": {"tags": (1, 3), "bg": "plain", "blur": (0.8, 2.0)},
    "motion-blur": {"tags": (1, 3), "bg": "plain", "motion": (3, 11)},
    "noise": {"tags": (1, 3), "bg": "plain", "noise": (4, 14)},
    "uneven-light": {"tags": (1, 3), "bg": "plain", "light": True},
    "low-contrast": {"tags": (1, 3), "bg": "plain", "contrast": (40, 90)},
    "underexposed": {"tags": (1, 3), "bg": "plain", "gain": (0.15, 0.35), "noise": (3, 6)},
    "overexposed": {"tags": (1, 3), "bg": "plain", "gain": (1.3, 1.8)},
    "jpeg": {"tags": (1, 3), "bg": "natural", "jpeg": (25, 70)},
    "grey-margin": {"tags": (1, 3), "bg": "plain", "grey": True},
    "border-occlusion": {"tags": (1, 2), "bg": "plain", "cross": True},
    "busy-background": {"tags": (1, 3), "bg": "clutter"},
    "frame-edge": {"tags": (1, 2), "bg": "plain", "edge": True},
    "noise-free": {"tags": (1, 3), "bg": "plain", "noise": (0, 0)},
    "dark-background": {"tags": (1, 3), "bg": "plain", "dark": True},
}


def render_frame(fam, cfg, w, h, rng, size_override=None, obstruct=None, light_noise=1.5):
    lo, hi = cfg.get("tags", (1, 3))
    ntags = int(rng.integers(lo, hi + 1))
    contrast = cfg.get("contrast")
    bw = (20.0, 235.0)
    if contrast:
        span = float(rng.uniform(*contrast))
        mid = float(rng.uniform(span / 2 + 10, 245 - span / 2))
        bw = (mid - span / 2, mid + span / 2)
    else:
        bw = (float(rng.uniform(10, 50)), float(rng.uniform(190, 245)))
    base = float(rng.uniform(100, 200))
    if cfg.get("grey"):  # a grey quiet zone on a bright wall: the margin is darker than its surroundings
        base = float(rng.uniform(225, 250))
        bw = (float(rng.uniform(10, 40)), float(rng.uniform(100, 140)))
    if cfg.get("dark"):
        base = float(rng.uniform(15, 90))
    img = background(rng, h, w, cfg.get("bg", "plain"), base)
    gt = []
    placed = []

    def draw(grid, size, edge, truth):
        for _ in range(60):
            theta = float(rng.uniform(0, 2 * math.pi))
            tmax = cfg.get("tilt", (0, 45))
            tilt = math.radians(float(rng.uniform(*tmax)))
            axis = float(rng.uniform(0, 2 * math.pi))
            c = (float(rng.uniform(0, w)), float(rng.uniform(0, h)))
            project, corners, outline = pose_quad(fam, size, theta, tilt, axis, c, w)
            xs = [p[0] for p in outline]
            ys = [p[1] for p in outline]
            if edge:
                c = (c[0] - min(xs), c[1] - min(ys))
                project, corners, outline = pose_quad(fam, size, theta, tilt, axis, c, w)
                xs = [p[0] for p in outline]
                ys = [p[1] for p in outline]
            if min(xs) < -1e-6 or min(ys) < -1e-6 or max(xs) > w or max(ys) > h:
                continue
            box = (min(xs) - 2, min(ys) - 2, max(xs) + 2, max(ys) + 2)
            if any(not (box[2] < b[0] or b[2] < box[0] or box[3] < b[1] or b[3] < box[1]) for b in placed):
                continue
            placed.append(box)
            paint_tag(img, fam, grid, project, outline, bw)
            if truth is not None:
                gt.append({"id": truth, "corners": [[round(x, 4), round(y, 4)] for x, y in corners],
                           "size_px": round(size, 2)})
            return True
        return False

    for _ in range(ntags):
        if "size_abs" in cfg:
            size = float(rng.uniform(*cfg["size_abs"]))
        elif "size_frac" in cfg:
            size = float(rng.uniform(*cfg["size_frac"])) * h
        elif size_override is not None:
            size = size_override
        else:
            lo_s, hi_s = cfg.get("size", (48 / 480, 170 / 480))
            size = float(rng.uniform(lo_s, hi_s)) * h
        tid = int(rng.integers(0, len(fam["codes"])))
        grid = cells(fam, fam["codes"][tid])
        if obstruct is not None:
            grid = obstruct_cells(fam, grid, obstruct, rng)
        draw(grid, size, cfg.get("edge", False), tid)
    ndecoy = int(rng.integers(*cfg["decoys"])) if "decoys" in cfg else 0
    for _ in range(ndecoy):
        draw(random_cells(fam, rng), float(rng.uniform(40, 170)) * h / 480, False, None)

    if cfg.get("cross"):
        # a bar (a cable) across the tag, over its border on two sides: it hides 1/4 to 1/3 of each
        for t in gt:
            c = np.array(t["corners"])
            a, b = c[int(rng.integers(0, 4))], None
            ctr = c.mean(axis=0)
            p0 = ctr + (a - ctr) * float(rng.uniform(1.0, 1.15))
            p1 = ctr + (a - ctr) * float(rng.uniform(-0.6, 0.2))
            ncross = bar(img, p0, p1, 0.12 * t["size_px"], float(rng.choice([bw[0], bw[1], 0.5 * (bw[0] + bw[1])])))
    if cfg.get("light"):
        gx, gy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        ang = rng.uniform(0, 2 * math.pi)
        grad = 1 - float(rng.uniform(0.3, 0.6)) * (0.5 + 0.5 * (gx * math.cos(ang) + gy * math.sin(ang)))
        vig = 1 - 0.35 * (gx ** 2 + gy ** 2) / 2
        img = img * (grad * vig).astype(np.float32)
    if "blur" in cfg:
        img = blur(img, float(rng.uniform(*cfg["blur"])))
    if "motion" in cfg:
        length = int(rng.choice(np.arange(cfg["motion"][0], cfg["motion"][1] + 1, 2)))
        img = motion_blur(img, length, float(rng.uniform(0, math.pi)))
    if "gain" in cfg:
        img = img * float(rng.uniform(*cfg["gain"]))
    sigma = float(rng.uniform(*cfg["noise"])) if "noise" in cfg else light_noise
    if sigma > 0:
        img = img + rng.normal(0, sigma, img.shape).astype(np.float32)
    if "jpeg" in cfg:
        img = jpeg_like(np.clip(img, 0, 255), int(rng.uniform(*cfg["jpeg"])))
    return np.clip(np.rint(img), 0, 255).astype(np.uint8), gt


def obstruct_cells(fam, grid, spec, rng):
    """spec = (where, fraction): covers the payload cells nearest an anchor (the centre, the middle of
    one border, or one corner) with white or grey, `fraction` of the black square's cells in all. The
    border itself stays visible, so the tag is still found and only its payload bits are damaged."""
    where, frac = spec
    t, w = fam["total"], fam["width"]
    m = (t - w) // 2
    k = int(round(frac * w * w))
    if k == 0:
        return grid
    anchor = {"centre": (w / 2 - 0.5, w / 2 - 0.5), "border": (w / 2 - 0.5, 1.0), "corner": (1.0, 1.0)}[where]
    cand = [(math.hypot(r - anchor[0], c - anchor[1]), r, c) for r in range(1, w - 1) for c in range(1, w - 1)]
    cand.sort()
    out = grid.copy()
    level = 1.0 if rng.random() < 0.5 else 0.5
    for _, r, c in cand[:k]:
        out[r + m, c + m] = level
    return out


def seed_for(*parts):
    return zlib.crc32("/".join(str(p) for p in parts).encode())


def job(args):
    src, out, scenario, cfg, w, h, idx, family, extra = args
    fam = load_family(src, family)
    rng = np.random.default_rng(seed_for(scenario, w, h, idx, family))
    kw = {}
    if extra.get("size_override"):
        kw["size_override"] = extra["size_override"]
    if extra.get("obstruct"):
        kw["obstruct"] = extra["obstruct"]
    img, gt = render_frame(fam, cfg, w, h, rng, **kw)
    name = f"{scenario}_{w}x{h}_{idx:03d}{extra.get('tag', '')}.pgm"
    with open(os.path.join(out, scenario, name), "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (w, h))
        f.write(img.tobytes())
    return name, {"size": [w, h], "tags": gt, "decoys": 0}


def write_set(out, scenario, results):
    results.sort()
    with open(os.path.join(out, scenario, "list.txt"), "w") as f:
        for name, _ in results:
            f.write(os.path.abspath(os.path.join(out, scenario, name)).replace("\\", "/") + "\n")
    json.dump({n: g for n, g in results}, open(os.path.join(out, scenario, "gt.json"), "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--sizes", default="640x480,1280x720,1920x1080")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    sizes = [tuple(int(v) for v in s.split("x")) for s in a.sizes.split(",")]
    only = set(a.only.split(",")) if a.only else None

    sets = {}  # name -> list of job tuples
    for name, cfg in SCENARIOS.items():
        sets[name] = [(a.src, a.out, name, cfg, w, h, i, "tag36h11", {}) for (w, h) in sizes for i in range(a.count)]
    # range sweep: one tag per frame with a black square of exactly 12..96 px at 640x480
    rng_cfg = {"tags": (1, 1), "bg": "plain", "tilt": (0, 15)}
    sets["range"] = [(a.src, a.out, "range", rng_cfg, 640, 480, i, "tag36h11",
                      {"size_override": float(s), "tag": f"_s{s:02d}"})
                     for i, s in enumerate(s for s in range(12, 97, 6) for _ in range(max(1, a.count // 2)))]
    # obstruction sweep: payload cells covered over 0..12% of one tag, at the smallest resolution
    ob_cfg = {"tags": (1, 1), "bg": "plain", "size": (0.3, 0.6)}
    ob = []
    for wi, where in enumerate(("centre", "border", "corner")):
        for fi, frac in enumerate((0.0, 0.02, 0.04, 0.06, 0.08, 0.12)):
            for k in range(max(1, a.count // 3)):
                ob.append((a.src, a.out, "obstruction", ob_cfg, 640, 480, (wi * 6 + fi) * 100 + k, "tag36h11",
                           {"obstruct": (where, frac), "tag": f"_{where}{int(frac * 100):02d}"}))
    sets["obstruction"] = ob
    # tag16h5 with decoys and clutter, for wrong IDs
    f_cfg = {"tags": (1, 3), "bg": "clutter", "decoys": (1, 4)}
    sets["family16h5"] = [(a.src, a.out, "family16h5", f_cfg, 640, 480, i, "tag16h5", {}) for i in range(a.count * 4)]

    with ProcessPoolExecutor(a.jobs) as ex:
        for name, jobs in sets.items():
            if only and name not in only:
                continue
            os.makedirs(os.path.join(a.out, name), exist_ok=True)
            res = list(ex.map(job, jobs, chunksize=8))
            write_set(a.out, name, res)
            print(f"{name}: {len(res)} frames", flush=True)


if __name__ == "__main__":
    main()
