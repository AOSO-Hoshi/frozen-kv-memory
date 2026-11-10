#!/bin/bash
# Is the 10-point scale effect real, or is it the seed?
#
# The controlled row moved the deployable hybrid from +84.5% (0.6B) to +94.5%
# (4B). Reading that requires knowing the noise, and the only spread we have for
# the hybrid was measured at --pred-len 512. So: three seeds of 0.6B under the
# matched settings. That gives the noise scale at the cheap end; if 0.6B holds
# within a couple of points, the 4B figure is outside it and the effect stands.
#
# Three seeds of the larger models would be better and cost two more hours.
set -u
cd "$(dirname "$0")"
source ~/ctfenv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out/scale3

C="--no-lora --mem-depth --lr 5e-3 --k 32 --k-random 16,128 --pred-len 192 \
   --steps 1000 --chunks 400 --eval-n 24 --dtype bf16 --model ./qwen3-0.6b"

for S in 0 1 2; do
  echo "=== train 06b s$S  $(date +%H:%M:%S) ==="
  python -u train_memtok.py $C --seed $S > out/scale3/06b_s$S.log 2>&1 \
    || { echo "  FAILED"; tail -3 out/scale3/06b_s$S.log; continue; }
  grep -E "^\[1000\]" out/scale3/06b_s$S.log | tail -1
done

H="--k 64 --anchors 512 --probe 16 --chunk 0 --hops 8 --n 12"
for S in 0 1 2; do
  CK="ckpt/memtok_k32_s${S}_pl192_deep_frozen.pt"
  [ -f "$CK" ] || { echo "skip s$S (no $CK)"; continue; }
  echo "=== hybrid 06b s$S  $(date +%H:%M:%S) ==="
  python -u hybrid.py $H --ckpt "$CK" --model ./qwen3-0.6b \
    > out/scale3/hyb_06b_s$S.log 2>&1 \
    || { echo "  FAILED"; tail -3 out/scale3/hyb_06b_s$S.log; continue; }
  grep -E "LAST SEG|NO state" out/scale3/hyb_06b_s$S.log | tail -2
done
echo "=== done $(date +%H:%M:%S) ==="
