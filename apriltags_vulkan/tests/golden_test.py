"""Golden-detection test: runs vk_eval over the committed corpus subset for each profile in
profiles.json and compares the dumps with the committed goldens.

usage: golden_test.py --vk-eval EXE --corpus DIR --golden DIR [--profiles a,b] [--update] [--out DIR]
       [--tolerance PX] [--noise-frames N]

Comparison per frame: the detections must have the same IDs (as a multiset), the same Hamming
distance, and every corner within --tolerance pixels (default 0.02, covering the sub-0.01 px
corner differences between GPUs and drivers). Frames with a different ID set or a corner beyond
the tolerance are counted; up to --noise-frames of them per profile (default 2) are allowed, for
the one-line run-to-run noise of a marginal detection. --update rewrites the goldens instead.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def load_dump(path):
    frames, cur = {}, None
    for line in open(path):
        f = line.split()
        if not f or f[0] == "#":
            continue
        if f[0] == "F":
            cur = int(f[1])
            frames[cur] = []
        elif cur is not None:
            v = [float(x) for x in f]
            frames[cur].append({"id": int(v[0]), "ham": int(v[1]), "margin": v[2], "c": v[3:5], "corners": v[5:13]})
    return frames


def compare_frame(got, want, tol):
    """Returns a message if the detections differ beyond the tolerance, else None."""
    ids_got = sorted(d["id"] for d in got)
    ids_want = sorted(d["id"] for d in want)
    if ids_got != ids_want:
        return f"ids {ids_got} != golden {ids_want}"
    # Detections are written sorted by id then centre; the same ID twice is matched in that order.
    worst = 0.0
    for g, w in zip(sorted(got, key=lambda d: (d["id"], d["c"][0], d["c"][1])),
                    sorted(want, key=lambda d: (d["id"], d["c"][0], d["c"][1]))):
        if g["ham"] != w["ham"]:
            return f"id {g['id']}: hamming {g['ham']} != golden {w['ham']}"
        worst = max(worst, max(abs(a - b) for a, b in zip(g["corners"], w["corners"])))
    if worst > tol:
        return f"corner differs by {worst:.4f} px (tolerance {tol})"
    return None


def scenarios(corpus):
    return sorted(d for d in os.listdir(corpus) if os.path.exists(os.path.join(corpus, d, "list.txt")))


def run_profile(exe, corpus, profile, out_dir, extra=()):
    env = dict(os.environ)
    env.update(profile["env"])
    for scenario in scenarios(corpus):
        d = os.path.join(corpus, scenario)
        names = [l.strip() for l in open(os.path.join(d, "list.txt")) if l.strip()]
        lst = os.path.join(out_dir, scenario + ".list")
        with open(lst, "w") as f:
            f.write("".join(os.path.abspath(os.path.join(d, n)).replace("\\", "/") + "\n" for n in names))
        cmd = [exe, "--list", lst, "--out", os.path.join(out_dir, scenario + ".txt"), "--iterations", "1",
               "--warmup", "0", "--decimation", str(profile["decimation"])]
        if scenario == "family16h5":
            cmd += ["--family", "tag16h5"]
        cmd += list(extra)
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{scenario}: vk_eval failed ({r.returncode})\n{r.stdout}{r.stderr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vk-eval", required=True)
    ap.add_argument("--corpus", default=os.path.join(HERE, "corpus"))
    ap.add_argument("--golden", default=os.path.join(HERE, "golden"))
    ap.add_argument("--profiles", default="")
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--vk-eval-arg", action="append", default=[], help="an extra argument for vk_eval (repeatable)")
    ap.add_argument("--out", default="")
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--noise-frames", type=int, default=2)
    a = ap.parse_args()

    profiles = json.load(open(os.path.join(HERE, "profiles.json")))
    names = [p for p in a.profiles.split(",") if p] or list(profiles)
    failed = 0
    for name in names:
        t0 = time.time()
        out_dir = os.path.join(a.out, name) if a.out else tempfile.mkdtemp(prefix=f"golden-{name}-")
        os.makedirs(out_dir, exist_ok=True)
        run_profile(a.vk_eval, a.corpus, profiles[name], out_dir, a.vk_eval_arg)
        gold_dir = os.path.join(a.golden, name)
        if a.update:
            os.makedirs(gold_dir, exist_ok=True)
            for s in scenarios(a.corpus):
                data = open(os.path.join(out_dir, s + ".txt"), "rb").read()
                open(os.path.join(gold_dir, s + ".txt"), "wb").write(data.replace(b"\r\n", b"\n"))
            print(f"{name}: goldens written ({time.time() - t0:.1f} s)")
            continue
        diffs = []
        for s in scenarios(a.corpus):
            got = load_dump(os.path.join(out_dir, s + ".txt"))
            want = load_dump(os.path.join(gold_dir, s + ".txt"))
            if sorted(got) != sorted(want):
                diffs.append(f"{s}: frame numbers differ")
                continue
            for i in sorted(want):
                msg = compare_frame(got[i], want[i], a.tolerance)
                if msg:
                    diffs.append(f"{s} frame {i}: {msg}")
        status = "ok" if len(diffs) <= a.noise_frames else "FAIL"
        print(f"{name}: {len(diffs)} differing frame(s), allowed {a.noise_frames}: {status} ({time.time() - t0:.1f} s)")
        for d in diffs:
            print("   ", d)
        if status == "FAIL":
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
