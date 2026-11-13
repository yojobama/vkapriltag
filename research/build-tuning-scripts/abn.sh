#!/bin/bash
# usage: abn.sh DEC OUTDIR ROUNDS name=exe ...   (first is the base); interleaved, rotating order; outputs OUTDIR/<name>_<round>
S=/c/Users/yojob/AppData/Local/Temp/claude/34b51eecd9123eb4/0a677b25-5396-4a18-8327-0da17e9c41ad/scratchpad
A=/c/Users/yojob/Projects/ext/avk-compare; T=/c/Users/yojob/Projects/vkapriltag/apriltags_vulkan/tools/napa_corpus
D=$1; O=$2; R=$3; shift 3; mkdir -p $O
V=("$@"); n=${#V[@]}
for r in $(seq $R); do
  for k in $(seq 0 $((n-1))); do
    i=$(( (k + r - 1) % n )); name=${V[$i]%%=*}; exe=${V[$i]#*=}
    rm -rf $O/${name}_$r
    bash $T/run_corpus.sh $exe $A/napa-corpus $O/${name}_$r --decimation $D --iterations 5 >/dev/null 2>&1
  done
done
