#!/usr/bin/env bash
# Same headroom test with the per-head future-attention ceiling added. The
# shared-position token oracle collapsed at 16384 (+73.0), so it stopped being
# a ceiling there; this one is what learned per-head selectors train toward.
cd /home/hoshi/Claude/memzip
source ~/ctfenv/bin/activate
for H in 8 16 32; do
  SECONDS=0
  python -u combo_cmp.py --n 12 --hops $H --ckpt ckpt/memtok_k32_s0_deep_frozen.pt \
    > out/length/combo_$((H*512))_s0.log 2>&1 \
    && echo "ok   future_$((H*512)) ${SECONDS}s" || echo "FAIL future_$((H*512))"
done >> out/length/progress.log
echo FUTURE_DONE >> out/length/progress.log
