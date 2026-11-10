#!/usr/bin/env bash
cd /home/hoshi/Claude/memzip
until grep -q LEVERS_DONE out/v2/progress.log; do sleep 30; done
source ~/ctfenv/bin/activate
for S in 0 1 2; do
  SECONDS=0
  python -u needle.py --n 10 --depths 0.1,0.3,0.5,0.7,0.9 --ctx 4096 --seed $S --block 1 \
    --ckpt ckpt/memtok_k32_s${S}_deep_frozen.pt > out/v2/needle_q_s$S.log 2>&1 \
    && echo "ok   needle_q_s$S.log ${SECONDS}s" || echo "FAIL needle_q_s$S.log"
done >> out/v2/progress.log
echo NEEDLEQ_DONE >> out/v2/progress.log
