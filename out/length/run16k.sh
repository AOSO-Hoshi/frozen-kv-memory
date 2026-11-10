#!/usr/bin/env bash
cd /home/hoshi/Claude/memzip
source ~/ctfenv/bin/activate
SECONDS=0
python -u combo_cmp.py --n 12 --hops 32 --ckpt ckpt/memtok_k32_s0_deep_frozen.pt \
  > out/length/combo_16384_s0.log 2>&1 \
  && echo "ok   combo_16384_s0 ${SECONDS}s" || echo "FAIL combo_16384_s0"
