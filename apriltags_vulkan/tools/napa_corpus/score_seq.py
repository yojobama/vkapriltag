"""Scores sequence dumps (track_eval) against ground truth, with per-frame detail.

usage: score_seq.py ROOT DUMP_DIR [DUMP_DIR ...]
For each DUMP_DIR/<seq>.txt prints: recall, false detections, wrong IDs (detections whose ID is not a
ground-truth tag of the frame, so tags partly outside the frame show up equally in every mode), the
median / p95 / max worst-corner error, the share of frames per path marker, and the discovery delay:
for each run of consecutive frames in which a tag is in the ground truth (and that starts after frame
0), the frames before its first matching detection (mean, max, number of runs, runs never found).
"""
import json
import os
import sys

import score_corpus as sc


def load_paths(path):
    paths, cur = {}, None
    for line in open(path):
        f = line.split()
        if f and f[0] == "F":
            cur = int(f[1])
        elif f and f[0] == "#" and f[1] == "path" and cur is not None:
            paths[cur] = int(f[2])
    return paths


def score_seq(root, name, dump):
    gt = json.load(open(os.path.join(root, name, "gt.json")))
    names = [os.path.basename(l.strip()) for l in open(os.path.join(root, name, "list.txt")) if l.strip()]
    frames = sc.load_dump(dump)
    paths = load_paths(dump)
    tags = found = wrong_id = false = 0
    errs = []
    hit = {}  # (id, frame) -> matched
    for i, n in enumerate(names):
        dets = list(frames.get(i, []))
        truth_ids = {t["id"] for t in gt[n]["tags"]}
        wrong_id += sum(1 for d in dets if d["id"] not in truth_ids)
        for t in gt[n]["tags"]:
            tags += 1
            side = t["size_px"]
            cand = [(sc.match(t["corners"], d["corners"], (0, 0)), d) for d in dets if d["id"] == t["id"]]
            cand = [c for c in cand if c[0][0] < 0.25 * side]
            hit[(t["id"], i)] = bool(cand)
            if not cand:
                continue
            (err, _, _), d = min(cand, key=lambda c: c[0][0])
            dets.remove(d)
            found += 1
            errs.append(err)
        false += len(dets)
    delays, never = [], 0
    for tid in {k[0] for k in hit}:
        start, first = None, None
        for i in range(len(names) + 1):
            present = (tid, i) in hit
            if present and start is None:
                start, first = i, None
            if present and first is None and hit[(tid, i)]:
                first = i - start
            if not present and start is not None:
                if first is None:
                    never += 1
                elif start > 0:
                    delays.append(first)
                start = None
    share = [0, 0, 0, 0]
    for p in paths.values():
        share[p] += 1
    tot = max(1, sum(share))
    return {"tags": tags, "found": found, "false": false, "wrong_id": wrong_id,
            "med": sc.pct(errs, 0.5), "p95": sc.pct(errs, 0.95), "max": max(errs) if errs else float("nan"),
            "share": [100.0 * s / tot for s in share],
            "delay_mean": sum(delays) / len(delays) if delays else float("nan"),
            "delay_max": max(delays) if delays else float("nan"), "delays": len(delays), "never": never}


def main():
    root = sys.argv[1]
    print(f"{'dir':10} {'seq':20} {'recall':>7} {'false':>5} {'wrongid':>7} {'med':>6} {'p95':>6} {'max':>6} {'full%':>6} {'trk%':>6} {'cap%':>5} {'dly':>5} {'dmax':>4} {'n':>3} {'never':>5}")
    for d in sys.argv[2:]:
        for f in sorted(os.listdir(d)):
            if not f.endswith(".txt"):
                continue
            seq = f[:-4]
            r = score_seq(root, seq, os.path.join(d, f))
            print(f"{os.path.basename(d.rstrip('/')):10} {seq:20} {100.0 * r['found'] / max(1, r['tags']):6.1f}% {r['false']:5d} {r['wrong_id']:7d} "
                  f"{r['med']:6.3f} {r['p95']:6.3f} {r['max']:6.3f} {r['share'][0]:6.1f} {r['share'][1] + r['share'][2]:6.1f} {r['share'][3]:5.1f} "
                  f"{r['delay_mean']:5.2f} {r['delay_max']:4.0f} {r['delays']:3d} {r['never']:5d}")


if __name__ == "__main__":
    main()
