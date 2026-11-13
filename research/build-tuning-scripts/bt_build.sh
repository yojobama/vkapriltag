#!/bin/sh
# usage: bt_build.sh NAME [cmake args...]  builds ~/w/build-tuning/src into ~/w/build-tuning/b-NAME
N=$1; shift
cd ~/w/build-tuning
cmake -S src/apriltags_vulkan -B b-$N -G Ninja -DCMAKE_BUILD_TYPE=Release -DVKAPRILTAG_BUILD_APPS=OFF "$@" > b-$N.cfg.log 2>&1 || { tail -20 b-$N.cfg.log; exit 1; }
cmake --build b-$N > b-$N.log 2>&1 || { tail -30 b-$N.log; exit 1; }
echo built $N
