import re, subprocess, os, sys
S = r"C:\Users\yojob\AppData\Local\Temp\claude\34b51eecd9123eb4\0a677b25-5396-4a18-8327-0da17e9c41ad\scratchpad"
cmds = open(S + r"\bt\cl_cmds.txt").read().splitlines()
B = r"C:\Users\yojob\Projects\ext\avk-compare\build-build-tuning-base"
extra = sys.argv[1:]
for f in ["RefineEdges", "QuadDecode", "OwnDecode", "BorderSample", "PolishFit"]:
    c = next(x for x in cmds if ("\\" + f + ".cpp") in x)
    c = re.sub(r" /showIncludes| /FS| /Fo\S+| /Fd\S+", "", c)
    c = c.replace(" -c ", " /Fo" + os.environ["TEMP"] + "\\vv.obj /Qvec-report:2 -c " + " ".join(extra) + " ")
    r = subprocess.run(["cmd", "/c", S + r"\vs.bat " + c], cwd=B, capture_output=True, text=True)
    open(S + r"\bt\msvc_" + f + ".txt", "w").write(r.stdout + r.stderr)
    print(f, len((r.stdout + r.stderr).splitlines()))
