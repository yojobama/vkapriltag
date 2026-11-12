#!/bin/bash
# usage: eval.sh [-e VK_EVAL] [-c CORPUS] [-o OUT_DIR] [-b BASELINE_JSON] [-l LABEL] [-u] [-s] [-- vk_eval options]
# Runs every scenario of a corpus (default: the committed subset in tests/corpus) through vk_eval,
# scores the dumps with napa_corpus/score_corpus.py and prints recall, false detections, corner
# error median/p95 and time against the baseline JSON (default tests/eval_baseline.json).
#   -u  store this run as the baseline for LABEL      -s  exit non-zero on a regression
# Options after -- go to vk_eval, e.g. `-- --decimation 1`; the default is --decimation 2.
# Environment switches (APRILTAG_VK_*) pass through. LABEL defaults to d<decimation>.
HERE=$(cd "$(dirname "$0")" && pwd)
EXE=${VK_EVAL:-}; CORPUS="$HERE/../tests/corpus"; OUT=""; BASE="$HERE/../tests/eval_baseline.json"; LABEL=""; EXTRA=()
while [ $# -gt 0 ]; do
  case $1 in
    -e) EXE=$2; shift 2;; -c) CORPUS=$2; shift 2;; -o) OUT=$2; shift 2;; -b) BASE=$2; shift 2;;
    -l) LABEL=$2; shift 2;; -u) EXTRA_REPORT="$EXTRA_REPORT --update-baseline"; shift;;
    -s) EXTRA_REPORT="$EXTRA_REPORT --strict"; shift;; --) shift; break;; *) echo "unknown option $1" >&2; exit 2;;
  esac
done
VKOPTS=("$@")
[ -n "$EXE" ] || { echo "need -e VK_EVAL (or set VK_EVAL)" >&2; exit 2; }
DEC=""; for ((i = 0; i < ${#VKOPTS[@]}; i++)); do [ "${VKOPTS[i]}" = --decimation ] && DEC=${VKOPTS[i+1]}; done
[ -n "$DEC" ] || { DEC=2; VKOPTS+=(--decimation 2); }
[ -n "$LABEL" ] || LABEL=d$DEC
[ -n "$OUT" ] || OUT=$(mktemp -d)
mkdir -p "$OUT"
total=0; n=0
for d in "$CORPUS"/*/; do
  s=$(basename "$d"); [ -f "$d/list.txt" ] || continue
  # Lists may hold bare file names; vk_eval needs paths that resolve from here.
  while IFS= read -r f; do f=${f%$'\r'}; [ -n "$f" ] || continue
    case $f in /*|?:*) echo "$f";; *) echo "$(cd "$d" && pwd -W 2>/dev/null || pwd)/$f";; esac
  done < "$d/list.txt" > "$OUT/$s.list"
  fam=(); [ "$s" = family16h5 ] && fam=(--family tag16h5)
  LIST="$OUT/$s.list"; [ -n "$(command -v cygpath)" ] && LIST=$(cygpath -m "$LIST")
  "$EXE" --list "$LIST" --out "$(cygpath -m "$OUT/$s.txt" 2>/dev/null || echo "$OUT/$s.txt")" --iterations 3 --warmup 1 "${fam[@]}" "${VKOPTS[@]}" > "$OUT/$s.log" 2>&1 || echo "$s: vk_eval failed" >&2
  t=$(sed -n 's/.*median_total=\([0-9.]*\) ms.*/\1/p' "$OUT/$s.log" | tail -n 1)
  [ -n "$t" ] && { total=$(awk -v a="$total" -v b="$t" 'BEGIN{print a+b}'); n=$((n+1)); }
done
TIME=nan; [ $n -gt 0 ] && TIME=$(awk -v a="$total" -v n="$n" 'BEGIN{printf "%.4f", a/n}')
python "$HERE/eval_report.py" "$CORPUS" "$OUT" --label "$LABEL" --baseline "$BASE" --time-ms "$TIME" $EXTRA_REPORT
