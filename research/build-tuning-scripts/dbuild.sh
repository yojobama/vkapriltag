#!/bin/bash
# usage: dbuild.sh NAME [cmake args...]  -> $A/build-build-tuning-NAME
S=/c/Users/yojob/AppData/Local/Temp/claude/34b51eecd9123eb4/0a677b25-5396-4a18-8327-0da17e9c41ad/scratchpad
A=/c/Users/yojob/Projects/ext/avk-compare
W=/c/Users/yojob/Projects/vkapriltag/.claude/worktrees/agent-a1cbf2804a671d5fc
N=$1; shift; B=$A/build-build-tuning-$N
cmd //c "$(cygpath -w $S/vs.bat) cmake -S $(cygpath -w $W/apriltags_vulkan) -B $(cygpath -w $B) -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl $*" > $S/bt/d_$N.log 2>&1
cmd //c "$(cygpath -w $S/vs.bat) cmake --build $(cygpath -w $B)" >> $S/bt/d_$N.log 2>&1
echo "done $N $(tail -1 $S/bt/d_$N.log)" >> $S/bt/dbuild.status
