#!/bin/bash
# A controlled scale row.
#
# The existing three points were never comparable: 1.7B ran 400 steps against
# 1000, and 4B ran --pred-len 192 against the 0.6B default of 512, which changes
# the gap itself (0.822 vs 0.492 nats) and so changes the denominator of every
# recovery figure. Matching upward is impossible -- 4B at pred-len 512 OOMs on
# 8GB -- so everything matches down to 192 instead, and 4B's existing run is
# reused unchanged.
#
# One confound cannot be removed: 4B only fits under NF4. So 0.6B is trained
# twice, bf16 and NF4, with everything else identical. That pair measures what
# quantising the frozen backbone costs the ratio, turning "the error should
# largely cancel" from an argument into a number.
set -u
cd "$(dirname "$0")"
source ~/ctfenv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out/scale3

C="--no-lora --mem-depth --lr 5e-3 --k 32 --k-random 16,128 --pred-len 192 \
   --steps 1000 --chunks 400 --eval-n 24"

run () {  # name, extra args
  local name="$1"; shift
  echo "=== $name  $(date +%H:%M:%S) ==="
  if ! python -u train_memtok.py $C "$@" > "out/scale3/$name.log" 2>&1; then
    echo "  FAILED:"; grep -iE "error|memory" "out/scale3/$name.log" | tail -3
    return 1
  fi
  grep -E "^\[1000\]" "out/scale3/$name.log" | tail -1
}

run 06b_bf16 --dtype bf16   --model ./qwen3-0.6b
run 06b_nf4  --load-4bit    --model ./qwen3-0.6b
run 17b_bf16 --dtype bf16   --model ./qwen3-1.7b

echo "=== checkpoints written ==="
ls -la ckpt/ | grep "pl192"
echo "=== done $(date +%H:%M:%S) ==="
