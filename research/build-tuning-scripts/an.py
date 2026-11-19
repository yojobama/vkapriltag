"""usage: an.py OUTDIR base name1 name2 ...  -- median over rounds of corpus median/p95, gpu, cpu and four scenarios, change vs base, dump equality vs base."""
import os, re, statistics, sys, filecmp

out, base, names = sys.argv[1], sys.argv[2], sys.argv[3:]
SC = ["one-tag", "dense-tags", "jpeg", "busy-background"]


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def load(d):
    allt, gpu, cpu, sc = [], [], [], {}
    for f in os.listdir(d):
        if not f.endswith(".log") or f == "run.log":
            continue
        s = f[:-4]
        for line in open(os.path.join(d, f), errors="replace"):
            m = re.match(r"TIME (\S+) gpu=([\d.]+) cpu=([\d.]+) total=([\d.]+)", line)
            if m:
                allt.append(float(m.group(4))); gpu.append(float(m.group(2))); cpu.append(float(m.group(3)))
                sc.setdefault(s, []).append(float(m.group(4)))
    return dict(med=statistics.median(allt), p95=pct(allt, 0.95), gpu=statistics.median(gpu), cpu=statistics.median(cpu),
                **{s: statistics.median(sc[s]) for s in SC if s in sc})


def rounds(n):
    r = 1
    res = []
    while os.path.isdir(os.path.join(out, f"{n}_{r}")):
        res.append(load(os.path.join(out, f"{n}_{r}"))); r += 1
    return res


def agg(n):
    rs = rounds(n)
    return {k: statistics.median([r[k] for r in rs]) for k in rs[0]}, len(rs)


def same(a, b):
    ndiff = 0; tot = 0
    for f in sorted(os.listdir(a)):
        if f.endswith(".txt"):
            tot += 1
            if not filecmp.cmp(os.path.join(a, f), os.path.join(b, f) if os.path.exists(os.path.join(b, f)) else os.devnull, shallow=False):
                ndiff += 1
    return f"{tot - ndiff}/{tot}"


B, nb = agg(base)
keys = ["med", "p95", "gpu", "cpu"] + SC
print(f"{'config':14s} n " + " ".join(f"{k:>16s}" for k in keys) + "  dumps=base")
for n in [base] + names:
    a, nr = agg(n)
    cells = []
    for k in keys:
        if n == base:
            cells.append(f"{a[k]:16.4f}")
        else:
            cells.append(f"{a[k]:.4f} {100 * (a[k] / B[k] - 1):+5.1f}%")
    eq = same(os.path.join(out, f"{n}_1"), os.path.join(out, f"{base}_1")) if n != base else (same(os.path.join(out, f"{base}_2"), os.path.join(out, f"{base}_1")) if nb > 1 else "-")
    print(f"{n:14s} {nr} " + " ".join(f"{c:>16s}" for c in cells) + "  " + eq)
