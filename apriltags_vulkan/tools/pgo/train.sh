#!/bin/bash
# usage: train.sh VK_EVAL TRACK_EVAL CORPUS_ROOT SEQ_ROOT
# Runs the profile-guided optimisation training set with an instrumented build (VKAPRILTAG_PGO=GENERATE):
# four scenarios at decimation 1 and 2, then every sequence through track_eval (modes full,key).
set -e
vk_eval=$1; track_eval=$2; corpus=$3; seqs=$4
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
for dec in 1 2; do
  for s in one-tag dense-tags jpeg busy-background; do
    "$vk_eval" --list "$corpus/$s/list.txt" --out "$work/$s.txt" --iterations 3 --warmup 1 --decimation $dec > "$work/$s.log"
  done
done
MODES=full,key bash "$here/../napa_corpus/run_sequences.sh" "$track_eval" "$seqs" "$work/seq" > /dev/null
echo "training finished"
