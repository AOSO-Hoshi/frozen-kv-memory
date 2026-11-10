#!/usr/bin/env bash
# levers-table row "anchor budget 512 -> 1024" (4096 ctx, 8 hops, probe 128,
# default ckpt). First reproduce v1 with the v1 code, to prove the arguments
# are the ones that made +87.8/+96.6, then rerun with the fixed scoring.
cd /home/hoshi/Claude/memzip
until grep -q COMBO_DONE out/v2/progress.log; do sleep 30; done
source ~/ctfenv/bin/activate
for A in 512 1024; do
  for V in v1 v2; do
    F=hybrid.py; [ $V = v1 ] && F=hybrid.py.v1bak
    SECONDS=0
    python -u $F --k 64 --anchors $A --probe 128 --chunk 0 --hops 8 --n 10 \
      > out/v2/levers_${V}_a$A.log 2>&1 && echo "ok   levers_${V}_a$A.log ${SECONDS}s" || echo "FAIL levers_${V}_a$A.log"
  done
done >> out/v2/progress.log
echo LEVERS_DONE >> out/v2/progress.log
