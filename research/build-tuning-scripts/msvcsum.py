import re, os, collections
S = r"C:\Users\yojob\AppData\Local\Temp\claude\34b51eecd9123eb4\0a677b25-5396-4a18-8327-0da17e9c41ad\scratchpad\bt"
names = ["FitSide", "ProfilePass", "LevelPass", "ClipBias", "FitLine", "FitQuadForBlob", "FitQuadHull", "SampleFamily", "ComputeHomography", "ReadBorder",
         "BuildScan", "BuildFit", "ModelPass", "Evaluate", "Optimise", "Dot4", "FitLevels", "Freeze", "ComputeBox"]
for f in ["RefineEdges", "QuadDecode", "OwnDecode", "BorderSample", "PolishFit"]:
    cur = None
    d = collections.defaultdict(lambda: [0, collections.Counter()])
    for l in open(os.path.join(S, "msvc_" + f + ".txt"), errors="replace"):
        m = re.match(r"--- Analyzing function: .*?(\w+)\(", l)
        if m:
            cur = m.group(1) if m.group(1) in names else None
            continue
        if cur is None:
            continue
        if "C5001" in l:
            d[cur][0] += 1
        m = re.search(r"C5002: loop not vectorized due to reason '(\d+)'", l)
        if m:
            d[cur][1][m.group(1)] += 1
    for n in names:
        if n in d:
            print(f, n, "vectorized loops:", d[n][0], "not:", dict(d[n][1]))
