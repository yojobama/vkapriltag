#!/bin/bash
# usage: run_corpus.sh VK_EVAL ROOT OUT_DIR [vk_eval options...]
# Runs vk_eval (see ext/avk-compare) over every scenario's list.txt and writes OUT_DIR/<scenario>.txt.
# The tag16h5 set needs a vk_eval that takes --family; it is run with it and the others without.
E=$1; R=$2; O=$3; shift 3; mkdir -p "$O"
for d in "$R"/*/; do
  s=$(basename "$d"); [ -f "$d/list.txt" ] || continue
  fam=""; [ "$s" = family16h5 ] && fam="--family tag16h5"
  "$E" --list "$d/list.txt" --out "$O/$s.txt" --iterations 1 --warmup 0 $fam "$@" > "$O/$s.log" 2>&1 || echo "$s: failed"
done
