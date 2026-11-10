#!/bin/bash
# usage: run_sequences.sh TRACK_EVAL.EXE SEQ_ROOT OUT_DIR [extra track_eval args...]
# Runs every sequence in SEQ_ROOT through track_eval (tracking off and on, interleaved) and writes
# OUT_DIR/full/<seq>.txt, OUT_DIR/track/<seq>.txt (score with score_corpus.py) and OUT_DIR/summary.txt.
exe=$1
root=$2
out=$3
shift 3
mkdir -p "$out/full" "$out/track"
: > "$out/summary.txt"
for d in "$root"/*/; do
  s=$(basename "$d")
  [ -f "$d/list.txt" ] || continue
  "$exe" --list "$d/list.txt" --out-prefix "$out/$s" "$@" | sed "s/^SUMMARY/SUMMARY seq=$s/" >> "$out/summary.txt"
  mv "$out/${s}_full.txt" "$out/full/$s.txt"
  mv "$out/${s}_track.txt" "$out/track/$s.txt"
done
cat "$out/summary.txt"
