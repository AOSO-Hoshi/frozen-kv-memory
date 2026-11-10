#!/usr/bin/env bash
# levers row "anchor budget", remeasured for v2: the v1 checkpoint behind
# +87.8/+96.6 no longer exists (the default ckpt now scores -2.3% as pure
# abstraction). Same settings as the 2x2 and the scale sweep, three seeds.
cd /home/hoshi/Claude/memzip
source ~/ctfenv/bin/activate
for A in 512 1024; do
  for S in 0 1 2; do
    SECONDS=0
    python -u hybrid.py --k 64 --anchors $A --probe 16 --chunk 0 --hops 8 --n 12 \
      --ckpt ckpt/memtok_k32_s${S}_deep_frozen.pt > out/v2/levers_a${A}_s$S.log 2>&1 \
      && echo "ok   levers_a${A}_s$S.log ${SECONDS}s" || echo "FAIL levers_a${A}_s$S.log"
  done
done >> out/v2/progress.log
echo LEVERS_DONE >> out/v2/progress.log
