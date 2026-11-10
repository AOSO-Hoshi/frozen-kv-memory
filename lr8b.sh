#!/bin/bash
# Is 8B's drop a scale effect or a learning-rate effect?
#
# 8B trained at lr 5e-3 peaked at +92.3% around step 500 and then stepped down
# to +86% by step 800, which is what too large a learning rate looks like. That
# lr was tuned on 0.6B and never checked on anything bigger; 4B happens to be
# fine with it, so "bigger always needs smaller" is not the rule either.
#
# Only the deployable hybrid figure matters for the conclusion, and 8B currently
# sits at +83.3 against +90.2 at 1.7B. If a smaller lr recovers that, the scale
# row saturates; if it does not, 8B genuinely falls off and that is a finding.
#
# seed 1 keeps the checkpoint name distinct from the 5e-3 run, which is already
# measured -- two variables move at once, but the alternative is another 20
# minutes to separate them and the lr is the hypothesis under test.
set -u
cd "$(dirname "$0")"
export PATH=/root/miniconda3/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MEMZIP_DATA=data_clean
LOG=./out/cloud
mkdir -p "$LOG"

echo "=== train 8B at lr 2e-3  $(date +%H:%M:%S) ==="
python -u train_memtok.py --no-lora --mem-depth --lr 2e-3 --pred-len 192 \
  --steps 1000 --chunks 400 --eval-n 24 --k 32 --k-random 16,128 --dtype bf16 \
  --seed 1 --model ./qwen3-8b > "$LOG/qwen3-8b_lr2e-3.log" 2>&1 \
  || { echo FAILED; tail -3 "$LOG/qwen3-8b_lr2e-3.log"; exit 1; }
echo "--- curve ---"
grep -oE "^\[ *[0-9]+\].*recovery=[+-][0-9.]+%" "$LOG/qwen3-8b_lr2e-3.log" \
  | sed -E "s/.*\[ *([0-9]+)\].*recovery=([+-][0-9.]+)%/step \1: \2/" | tail -8

CK=ckpt/memtok_8b_k32_data_clean_s1_pl192_deep_frozen.pt
[ -f "$CK" ] || { echo "no checkpoint at $CK"; ls ckpt/ | grep 8b; exit 1; }
echo "=== hybrid  $(date +%H:%M:%S) ==="
python -u hybrid.py --k 64 --anchors 512 --probe 16 --chunk 0 --hops 8 --n 12 \
  --ckpt "$CK" --model ./qwen3-8b > "$LOG/hyb_qwen3-8b_lr2e-3.log" 2>&1 \
  || { echo FAILED; tail -3 "$LOG/hyb_qwen3-8b_lr2e-3.log"; exit 1; }
grep -E "state=LAST SEG only|NO state|ORACLE, state" "$LOG/hyb_qwen3-8b_lr2e-3.log" | tail -3
echo "=== all done $(date +%H:%M:%S) ==="
