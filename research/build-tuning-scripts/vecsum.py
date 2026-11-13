import re, sys, collections, os
R = r"C:\Users\yojob\Projects\vkapriltag\.claude\worktrees\agent-a1cbf2804a671d5fc\apriltags_vulkan\library\src"
targets = {
    "RefineEdges": ("RefineEdges.cpp", ["FitSide", "ProfilePass", "LevelPass", "ClipBias"]),
    "QuadDecode": ("gpu/QuadDecode.cpp", ["FitLine", "FitQuadForBlob", "FitQuadHull"]),
    "OwnDecode": ("OwnDecode.cpp", ["SampleFamily", "ComputeHomography"]),
    "BorderSample": ("BorderSample.cpp", ["ReadBorder"]),
    "PolishFit": ("PolishFit.cpp", ["BuildScan", "BuildFit", "ComputeBox", "Freeze", "ModelPass", "Evaluate", "Optimise", "Dot4", "FitLevels", "MedianAbsResidual", "PolishDetection"]),
}
tag = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
for k, (f, names) in targets.items():
    L = open(os.path.join(R, *f.split("/")), encoding="utf8", errors="replace").read().split("\n")
    msgs = collections.defaultdict(set)
    opt = collections.defaultdict(set)
    for l in open(os.path.join(here, "vec", tag, k + ".txt"), errors="replace"):
        m = re.match(r".*?/?([\w.]+):(\d+):(\d+): (missed|optimized): (.*)", l.rstrip())
        if not m or m.group(1) != f.split("/")[-1]:
            continue
        ln = int(m.group(2)); t = m.group(5)
        if m.group(4) == "optimized" and "vectorized" in t:
            opt[ln].add(t)
        elif "not vectorized:" in t:
            msgs[ln].add(t.replace("not vectorized: ", ""))
    for n in names:
        st = [i for i, x in enumerate(L, 1) if re.match(r"[A-Za-z].*\b" + n + r"\(", x) and ";" not in x]
        for s in st:
            e = next(i for i in range(s, len(L) + 1) if L[i - 1].startswith("}"))
            print(f"== {k}::{n} lines {s}-{e}")
            for ln in sorted(opt):
                if s <= ln <= e:
                    print("  VEC  ", ln, sorted(opt[ln])[0])
            for ln in sorted(msgs):
                if s <= ln <= e:
                    print("  MISS ", ln, "; ".join(sorted(msgs[ln]))[:150], "|", L[ln - 1].strip()[:50])
