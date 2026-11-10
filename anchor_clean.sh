#!/bin/bash
# A 0.6B baseline on the scrubbed corpus.
#
# The rented box gets data_clean (real credentials do not leave this machine),
# while every local number so far was measured on data. Scrubbing shifted the
# gap from 0.768 to 0.756 nats -- 1.6%, worth about 1.3 recovery points, which
# is the same size as the seed noise. Too small to change any conclusion, too
# large to leave unstated.
#
# So rather than retrain every local scale point on the clean corpus (~1.5h,
# mostly 4B), measure one anchor here. The 8B point can then be read against a
# same-corpus baseline, and the two 0.6B versions bound the corpus effect.
set -u
cd "$(dirname "$0")"
source ~/ctfenv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MEMZIP_DATA=data_clean
mkdir -p out/clean

C="--no-lora --mem-depth --lr 5e-3 --k 32 --k-random 16,128 --pred-len 192 \
   --steps 1000 --chunks 400 --eval-n 24 --dtype bf16 --model ./qwen3-0.6b"

for S in 0 1; do
  echo "=== train 06b clean s$S  $(date +%H:%M:%S) ==="
  python -u train_memtok.py $C --seed $S > out/clean/06b_s$S.log 2>&1 \
    || { echo "  FAILED"; tail -3 out/clean/06b_s$S.log; continue; }
  grep -E "^\[1000\]" out/clean/06b_s$S.log | tail -1
done

H="--k 64 --anchors 512 --probe 16 --chunk 0 --hops 8 --n 12"
for S in 0 1; do
  CK="ckpt/memtok_k32_data_clean_s${S}_pl192_deep_frozen.pt"
  [ -f "$CK" ] || { echo "skip s$S (no $CK)"; ls ckpt/ | grep clean; continue; }
  echo "=== hybrid 06b clean s$S  $(date +%H:%M:%S) ==="
  python -u hybrid.py $H --ckpt "$CK" --model ./qwen3-0.6b \
    > out/clean/hyb_06b_s$S.log 2>&1 \
    || { echo "  FAILED"; tail -3 out/clean/hyb_06b_s$S.log; continue; }
  grep -E "state=LAST SEG only|NO state|ORACLE, state" out/clean/hyb_06b_s$S.log | tail -3
done
echo "=== done $(date +%H:%M:%S) ==="
