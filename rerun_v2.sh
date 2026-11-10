#!/usr/bin/env bash
# Paper v2: rerun every 0.6B result that goes through attention-based
# selection, after (1) the RoPE fix to hand-computed attention and (2) the
# per-layer, per-head H2O/SnapKV plus question-window SnapKV. Same arguments
# and per-seed checkpoints as the v1 runs. tests/test_attention_match.py gates it.
set -u
cd "$(dirname "$0")"
source ~/ctfenv/bin/activate
O=out/v2; mkdir -p $O
python tests/test_attention_match.py > $O/test.log 2>&1 || { echo "attention test FAILED"; exit 1; }
run() { local log=$1; shift; SECONDS=0; python -u "$@" > $O/$log 2>&1 \
          && echo "ok   $log ${SECONDS}s" || echo "FAIL $log"; }
for S in 0 1 2; do
  CK=ckpt/memtok_k32_s${S}_deep_frozen.pt
  run needle_b1_s$S.log needle.py --n 10 --depths 0.1,0.3,0.5,0.7,0.9 --ctx 4096 --seed $S --block 1 --ckpt $CK
  run baseline_b1_s$S.log baseline_cmp.py --n 12 --block 1 --ckpt $CK
done
for S in 0 1 2; do
  run hyb_s$S.log hybrid.py --k 64 --anchors 512 --probe 16 --chunk 0 --hops 16 --n 12 \
    --ckpt ckpt/memtok_k32_s${S}_deep_frozen.pt
done
for S in 0 1 2; do
  CK=ckpt/memtok_k32_s${S}_deep_frozen.pt
  run needle_b8_s$S.log needle.py --n 10 --depths 0.1,0.3,0.5,0.7,0.9 --ctx 4096 --seed $S --block 8 --ckpt $CK
  run baseline_b8_s$S.log baseline_cmp.py --n 12 --block 8 --ckpt $CK
done
echo ALL_DONE
