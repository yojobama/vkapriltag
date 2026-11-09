"""Scores detection dumps against the corpus ground truth.

usage: score_corpus.py ROOT DUMP_DIR [--offset DX DY] [--json OUT]
Reads ROOT/<scenario>/gt.json and list.txt, and DUMP_DIR/<scenario>.txt (the "F <n> <count>" dump
format of apriltag_vulkan_validate --dump-detections and vk_eval). Per scenario: tags, found,
recall, false detections (wrong IDs and detections with no tag), the median, 95th percentile and
maximum worst-corner error in pixels, the mean signed corner offset, and tags found at the wrong
winding. A detection matches a truth tag with the same ID and a corner error below a quarter of
the tag's side. --offset is subtracted from detected corners (libapriltag's pixel convention).
"""
import argparse
import json
import math
import os


def load_dump(path):
    frames, cur = {}, None
    for line in open(path):
        f = line.split()
        if not f or f[0] == "#":
            continue
        if f[0] == "F":
            cur = int(f[1])
            frames[cur] = []
        else:
            v = [float(x) for x in f]
            frames[cur].append({"id": int(v[0]), "ham": int(v[1]), "corners": [(v[5 + 2 * k], v[6 + 2 * k]) for k in range(4)]})
    return frames


def pct(v, q):
    if not v:
        return float("nan")
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def match(truth, det, off):
    """(worst-corner error, rotation, signed mean offset) of the best winding of det onto truth."""
    best = None
    for rot in range(4):
        errs, sx, sy = [], 0.0, 0.0
        for k in range(4):
            tx, ty = truth[k]
            dx, dy = det[(k + rot) % 4]
            dx, dy = dx - off[0], dy - off[1]
            errs.append(math.hypot(dx - tx, dy - ty))
            sx += dx - tx
            sy += dy - ty
        r = (max(errs), rot, (sx / 4, sy / 4))
        if best is None or r[0] < best[0]:
            best = r
    return best


def score_set(root, name, dump, off):
    gt = json.load(open(os.path.join(root, name, "gt.json")))
    names = [os.path.basename(l.strip()) for l in open(os.path.join(root, name, "list.txt")) if l.strip()]
    frames = load_dump(dump)
    tags = found = false = wrong_rot = 0
    errs, offs, by_size = [], [], {}
    for i, n in enumerate(names):
        dets = list(frames.get(i, []))
        for t in gt[n]["tags"]:
            tags += 1
            side = t["size_px"]
            bucket = by_size.setdefault(int(side // 6 * 6), [0, 0])
            bucket[0] += 1
            cand = [(match(t["corners"], d["corners"], off), d) for d in dets if d["id"] == t["id"]]
            cand = [c for c in cand if c[0][0] < 0.25 * side]
            if not cand:
                continue
            (err, rot, so), d = min(cand, key=lambda c: c[0][0])
            dets.remove(d)
            found += 1
            bucket[1] += 1
            errs.append(err)
            offs.append(so)
            wrong_rot += rot != 0
        false += len(dets)
    n = len(offs)
    return {"frames": len(names), "tags": tags, "found": found, "recall": found / tags if tags else float("nan"),
            "false": false, "err_median": pct(errs, 0.5), "err_p95": pct(errs, 0.95), "err_max": max(errs) if errs else float("nan"),
            "offset": [sum(o[0] for o in offs) / n, sum(o[1] for o in offs) / n] if n else [float("nan")] * 2,
            "wrong_winding": wrong_rot, "by_size": {str(k): v for k, v in sorted(by_size.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("dumps")
    ap.add_argument("--offset", type=float, nargs=2, default=[0.0, 0.0])
    ap.add_argument("--json")
    a = ap.parse_args()
    sets = sorted(d for d in os.listdir(a.root) if os.path.exists(os.path.join(a.root, d, "gt.json")))
    out = {}
    print(f"{'scenario':16} {'tags':>5} {'found':>5} {'recall':>7} {'false':>5} {'med px':>7} {'p95 px':>7} {'max px':>7} {'offset':>14} {'winding':>7}")
    for s in sets:
        p = os.path.join(a.dumps, s + ".txt")
        if not os.path.exists(p):
            continue
        r = out[s] = score_set(a.root, s, p, a.offset)
        print(f"{s:16} {r['tags']:5d} {r['found']:5d} {r['recall']*100:6.1f}% {r['false']:5d} {r['err_median']:7.3f} {r['err_p95']:7.3f} "
              f"{r['err_max']:7.3f} {r['offset'][0]:+6.3f},{r['offset'][1]:+6.3f} {r['wrong_winding']:7d}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
