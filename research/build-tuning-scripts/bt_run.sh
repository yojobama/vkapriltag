#!/bin/sh
# usage: bt_run.sh OUT DEC ROUNDS name ...   (base first); runs ~/w/build-tuning/b-<name>/tools/vk_eval, interleaved, rotating order
O=$1; D=$2; R=$3; shift 3
W=$HOME/w/build-tuning; C=$HOME/w/napa-corpus; mkdir -p $W/lists $W/out/$O
for d in $C/*/; do s=$(basename $d); [ -f $W/lists/$s.list ] || ls $d*.pgm | grep -v 1920x1080 > $W/lists/$s.list; done
n=$#; r=1
while [ $r -le $R ]; do
  k=0
  while [ $k -lt $n ]; do
    i=$(( (k + r - 1) % n + 1 )); name=$(eval echo \${$i})
    od=$W/out/$O/${name}_$r; rm -rf $od; mkdir -p $od
    for l in $W/lists/*.list; do s=$(basename $l .list); fam=""; [ $s = family16h5 ] && fam="--family tag16h5"
      $PIN $W/b-$name/tools/vk_eval --list $l --out $od/$s.txt --iterations 5 --warmup 0 --decimation $D $fam > $od/$s.log 2>&1 || echo "$s failed"
    done
    k=$((k+1))
  done
  r=$((r+1))
done
echo done $O
