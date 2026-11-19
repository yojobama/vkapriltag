"""Scores a corpus's detection dumps and prints the headline numbers against a stored baseline.

usage: eval_report.py CORPUS DUMP_DIR [--label NAME] [--baseline FILE] [--time-ms MS] [--update-baseline]
                      [--strict]
Per scenario and overall: recall, false detections, median and 95th percentile worst-corner error
(px), and (with --time-ms) the mean per-frame time. The baseline JSON maps a label to the same
numbers; differences from it are printed, and with --strict a lower recall, more false detections
or a worse p95 error (beyond --err-slack px) exits non-zero. --update-baseline stores this run.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "napa_corpus"))
import score_corpus as sc


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))] if v else float("nan")


def score(root, dumps):
    pooled = []
    real = sc.pct
    sc.pct = lambda v, q: (pooled.extend(v) if q == 0.5 else None, real(v, q))[1]
    rows, tags, found, false = {}, 0, 0, 0
    for s in sorted(d for d in os.listdir(root) if os.path.exists(os.path.join(root, d, "gt.json"))):
        p = os.path.join(dumps, s + ".txt")
        if not os.path.exists(p):
            continue
        r = sc.score_set(root, s, p, [0.0, 0.0])
        rows[s] = {k: r[k] for k in ("tags", "found", "recall", "false", "err_median", "err_p95")}
        tags, found, false = tags + r["tags"], found + r["found"], false + r["false"]
    sc.pct = real
    overall = {"tags": tags, "found": found, "recall": found / tags if tags else float("nan"), "false": false,
               "err_median": pct(pooled, 0.5), "err_p95": pct(pooled, 0.95)}
    return overall, rows


def fmt(x, spec):
    return "   n/a" if x != x else format(x, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("dumps")
    ap.add_argument("--label", default="run")
    ap.add_argument("--baseline", default="")
    ap.add_argument("--time-ms", type=float, default=float("nan"))
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--err-slack", type=float, default=0.02)
    a = ap.parse_args()

    overall, rows = score(a.corpus, a.dumps)
    overall["time_ms"] = a.time_ms
    base = {}
    if a.baseline and os.path.exists(a.baseline):
        base = json.load(open(a.baseline)).get(a.label, {})
    ref = base.get("overall")
    ref_rows = base.get("scenarios", {})

    print(f"{'scenario':18} {'tags':>5} {'found':>5} {'recall':>7} {'false':>5} {'med px':>7} {'p95 px':>7}  vs baseline")
    for s, r in list(rows.items()) + [("OVERALL", overall)]:
        b = ref if s == "OVERALL" else ref_rows.get(s)
        note = ""
        if b:
            d = (r["found"] - b["found"], r["false"] - b["false"], r["err_median"] - b["err_median"], r["err_p95"] - b["err_p95"])
            note = f"found {d[0]:+d}, false {d[1]:+d}, med {d[2]:+.3f}, p95 {d[3]:+.3f}" if any(x == x and x != 0 for x in d) else "same"
        print(f"{s:18} {r['tags']:5d} {r['found']:5d} {r['recall'] * 100:6.1f}% {r['false']:5d} {fmt(r['err_median'], '7.3f')} {fmt(r['err_p95'], '7.3f')}  {note}")
    if a.time_ms == a.time_ms:
        extra = f" (baseline {ref['time_ms']:.3f} ms, {a.time_ms / ref['time_ms'] - 1:+.1%})" if ref and ref.get("time_ms") == ref.get("time_ms") and ref.get("time_ms") else ""
        print(f"time: {a.time_ms:.3f} ms per frame, mean over scenarios{extra}")

    if a.update_baseline:
        data = json.load(open(a.baseline)) if a.baseline and os.path.exists(a.baseline) else {}
        data[a.label] = {"overall": overall, "scenarios": rows}
        json.dump(data, open(a.baseline, "w"), indent=1, sort_keys=True, allow_nan=True)
        open(a.baseline, "a").write("\n")
        print(f"baseline '{a.label}' written to {a.baseline}")
    elif ref is None and a.baseline:
        print(f"(no baseline for label '{a.label}' in {a.baseline})")
    regress = ref is not None and (overall["found"] < ref["found"] or overall["false"] > ref["false"] or
                                   overall["err_p95"] > ref["err_p95"] + a.err_slack)
    if regress:
        print("REGRESSION against the baseline")
    return 1 if regress and a.strict else 0


if __name__ == "__main__":
    sys.exit(main())
