#!/usr/bin/env bash
cd /home/hoshi/Claude/memzip
until grep -q ALL_DONE out/v2/progress.log; do sleep 30; done
source ~/ctfenv/bin/activate
for S in 0 1 2; do
  SECONDS=0
  python -u combo_cmp.py --n 12 --ckpt ckpt/memtok_k32_s${S}_deep_frozen.pt > out/v2/combo_s$S.log 2>&1 \
    && echo "ok   combo_s$S.log ${SECONDS}s" || echo "FAIL combo_s$S.log"
done >> out/v2/progress.log
echo COMBO_DONE >> out/v2/progress.log
