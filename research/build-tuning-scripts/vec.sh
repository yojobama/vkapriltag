#!/bin/sh
# usage: vec.sh TAG EXTRA_FLAGS...  -> ~/w/build-tuning/vec/TAG/<file>.txt
T=$1; shift
cd ~/w/build-tuning/b-base
mkdir -p ../vec/$T
for f in RefineEdges QuadDecode OwnDecode BorderSample PolishFit; do
  p=$(ls ../src/apriltags_vulkan/library/src/$f.cpp ../src/apriltags_vulkan/library/src/gpu/$f.cpp 2>/dev/null | head -1)
  cmd=$(ninja -t commands library/libvkapriltag.a | grep "/$f.cpp" | head -1)
  cmd=$(echo "$cmd" | sed 's/ -o [^ ]*//; s/ -MD -MT [^ ]*//; s/ -MF [^ ]*//')
  sh -c "$cmd $* -O3 -fopt-info-vec-missed -fopt-info-vec-optimized -c -o /dev/null" > ../vec/$T/$f.txt 2>&1
  echo $f $(wc -l < ../vec/$T/$f.txt)
done
