"""usage: napa_summary.py TIMINGS_JSON [CORPUS DUMP_DIR] -- median / p95 ms over all frames and per frame size; with CORPUS and DUMP_DIR also overall recall and pooled error."""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def timing(path):
    t = json.load(open(path))
    groups = {}
    for frames in t.values():
        for w, h, med, _ in frames:
            groups.setdefault("all", []).append(med)
            groups.setdefault(f"{w}x{h}", []).append(med)
    for k, v in sorted(groups.items()):
        print(f"{k:>10} {len(v):6d} frames  median {statistics.median(v):.3f}  p95 {pct(v, 0.95):.3f} ms")


def accuracy(root, dumps):
    import score_corpus
    errs = []
    real = score_corpus.pct
    score_corpus.pct = lambda v, q: (errs.extend(v) if q == 0.5 else None, real(v, q))[1]
    tags = found = false = 0
    for s in sorted(d for d in os.listdir(root) if os.path.exists(os.path.join(root, d, "gt.json"))):
        p = os.path.join(dumps, s + ".txt")
        if os.path.exists(p):
            r = score_corpus.score_set(root, s, p, [0.0, 0.0])
            tags, found, false = tags + r["tags"], found + r["found"], false + r["false"]
    print(f"tags {tags} found {found} ({100 * found / tags:.1f}%) false {false} error median {pct(errs, 0.5):.3f} p95 {pct(errs, 0.95):.3f} px")


if __name__ == "__main__":
    timing(sys.argv[1])
    if len(sys.argv) == 4:
        accuracy(sys.argv[2], sys.argv[3])
