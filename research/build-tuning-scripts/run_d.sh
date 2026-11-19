#!/bin/bash
# usage: run_d.sh TAG DEC ROUNDS name ... (names are build-build-tuning-<name>; base always first)
S=/c/Users/yojob/AppData/Local/Temp/claude/34b51eecd9123eb4/0a677b25-5396-4a18-8327-0da17e9c41ad/scratchpad; A=/c/Users/yojob/Projects/ext/avk-compare
TAG=$1; D=$2; R=$3; shift 3
args=(base=/c/Users/yojob/Projects/ext/avk-compare/build-build-tuning-base/tools/vk_eval.exe); for n in "$@"; do args+=($n=/c/Users/yojob/Projects/ext/avk-compare/build-build-tuning-$n/tools/vk_eval.exe); done
bash $S/benchlock.sh bash $S/bt/abn.sh $D $A/bt-$TAG-d$D $R "${args[@]}"
