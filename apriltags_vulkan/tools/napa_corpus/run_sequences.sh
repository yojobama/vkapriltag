#!/bin/bash
# usage: MODES=full,key,async run_sequences.sh TRACK_EVAL.EXE SEQ_ROOT OUT_DIR [extra track_eval args...]
# Runs every sequence in SEQ_ROOT through track_eval (the modes interleaved) and writes
# OUT_DIR/<mode>/<seq>.txt (score with score_seq.py) and OUT_DIR/summary.txt.
exe=$1
root=$2
out=$3
shift 3
modes=${MODES:-full,key,async}
mkdir -p "$out"
: > "$out/summary.txt"
for m in ${modes//,/ }; do mkdir -p "$out/$m"; done
for d in "$root"/*/; do
  s=$(basename "$d")
  [ -f "$d/list.txt" ] || continue
  "$exe" --list "$d/list.txt" --out-prefix "$out/$s" --modes "$modes" "$@" | grep '^SUMMARY' | sed "s/^SUMMARY/SUMMARY seq=$s/" >> "$out/summary.txt"
  for m in ${modes//,/ }; do
    mv "$out/${s}_$m.txt" "$out/$m/$s.txt"
  done
done
cat "$out/summary.txt"
