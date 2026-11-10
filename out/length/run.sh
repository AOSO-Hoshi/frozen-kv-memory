#!/usr/bin/env bash
# Does the room above a strong per-head selector reopen with context length?
# oracle+slots minus H2O+slots at 4096 is 2.3 points; if it stays that small at
# 8192 and 16384, a learned retriever (paper-2 direction A) has nothing to win.
cd /home/hoshi/Claude/memzip
source ~/ctfenv/bin/activate
for H in 16 32; do
  SECONDS=0
  python -u combo_cmp.py --n 12 --hops $H --ckpt ckpt/memtok_k32_s0_deep_frozen.pt \
    > out/length/combo_$((H*512))_s0.log 2>&1 \
    && echo "ok   combo_$((H*512))_s0 ${SECONDS}s" || echo "FAIL combo_$((H*512))_s0"
done >> out/length/progress.log
echo LENGTH_DONE >> out/length/progress.log
