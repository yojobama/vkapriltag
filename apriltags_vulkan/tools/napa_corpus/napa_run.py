"""usage: napa_run.py ROOT OUT_DIR [--min-tag-px N] [--reps N] [--gap-ms MS] -- NAPA over the corpus: dumps in vk_eval's format, per-frame timings."""
import argparse, json, os, statistics, sys, time
import numpy as np
from napa_vision import NapaDetector

def read_pgm(path):
    b = open(path, "rb").read()
    parts, pos = [], 0
    while len(parts) < 4:
        while b[pos:pos + 1].isspace(): pos += 1
        if b[pos:pos + 1] == b"#":
            while b[pos:pos + 1] != b"\n": pos += 1
            continue
        s = pos
        while not b[pos:pos + 1].isspace(): pos += 1
        parts.append(b[s:pos])
    pos += 1
    w, h = int(parts[1]), int(parts[2])
    return np.frombuffer(b, dtype=np.uint8, count=w * h, offset=pos).reshape(h, w)

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("out")
ap.add_argument("--min-tag-px", type=int, default=32)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--only")
ap.add_argument("--gap-ms", type=float, default=0.0, help="sleep before each frame, to time at a camera rate")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
timings = {}
for s in sorted(os.listdir(a.root)):
    lst = os.path.join(a.root, s, "list.txt")
    if not os.path.exists(lst) or (a.only and s != a.only): continue
    det = NapaDetector(family="tag16h5" if s == "family16h5" else "tag36h11", min_tag_px=a.min_tag_px)
    out, times = [], []
    for i, line in enumerate(l.strip() for l in open(lst) if l.strip()):
        img = read_pgm(line if os.path.exists(line) else os.path.join(a.root, s, os.path.basename(line)))
        res = None; ts = []
        for r in range(a.reps):
            if a.gap_ms > 0: time.sleep(a.gap_ms / 1e3)
            t = time.perf_counter(); res = det.detect(img); ts.append(time.perf_counter() - t)
        times.append((img.shape[1], img.shape[0], statistics.median(ts) * 1e3, min(ts) * 1e3))
        out.append(f"F {i} {len(res)}")
        for t in res:
            # NAPA puts (0, 0) at the first pixel's centre and winds the other way; the corpus truth uses the top-left corner.
            c = np.asarray(t.corners, dtype=float).reshape(4, 2)[[1, 0, 3, 2]] + 0.5; ctr = np.asarray(t.center, dtype=float) + 0.5
            out.append(f"{t.tag_id} {getattr(t, 'hamming', 0)} {getattr(t, 'decision_margin', 0.0):.3f} {ctr[0]:.6f} {ctr[1]:.6f} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in c))
    open(os.path.join(a.out, s + ".txt"), "w").write("\n".join(out) + "\n")
    timings[s] = times
json.dump(timings, open(os.path.join(a.out, "timings.json"), "w"))
print("done", {k: round(statistics.median(t[2] for t in v), 3) for k, v in timings.items()})
