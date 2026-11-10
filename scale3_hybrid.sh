#!/bin/bash
# The hybrid architecture across a controlled scale row.
#
# Every module here was trained with identical hyperparameters and --pred-len
# 192; 4B reuses run4b.sh's checkpoint, which already matched those settings.
# 4096-token contexts (8 hops), the same configuration as the scale row of the
# paper's lever table.
#
# 0.6B appears twice, bf16 and NF4. That pair isolates the one confound that
# cannot be designed away -- 4B only fits quantised -- so the cost of NF4 to the
# recovery ratio is measured rather than assumed.
set -u
cd "$(dirname "$0")"
source ~/ctfenv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out/scale3

H="--k 64 --anchors 512 --probe 16 --chunk 0 --hops 8 --n 12"

run () {  # name, ckpt, model, [extra]
  local name="$1" ck="$2" mdl="$3"; shift 3
  if [ ! -f "$ck" ]; then echo "=== $name: MISSING $ck ==="; return 1; fi
  echo "=== $name  $(date +%H:%M:%S) ==="
  if ! python -u hybrid.py $H --ckpt "$ck" --model "$mdl" "$@" \
        > "out/scale3/hyb_$name.log" 2>&1; then
    echo "  FAILED:"; grep -iE "error|memory" "out/scale3/hyb_$name.log" | tail -3
    return 1
  fi
  grep -E "LAST SEG|NO state|ORACLE \+|SPECULATIVE" "out/scale3/hyb_$name.log" | tail -4
}

run 06b_bf16 ckpt/memtok_k32_pl192_deep_frozen.pt      ./qwen3-0.6b
run 06b_nf4  ckpt/memtok_k32_pl192_deep_nf4_frozen.pt  ./qwen3-0.6b --load-4bit
run 17b_bf16 ckpt/memtok_1.7b_k32_pl192_deep_frozen.pt ./qwen3-1.7b
run 4b_nf4   ckpt/memtok_4b_deep.pt                    ./qwen3-4b   --load-4bit

echo "=== done $(date +%H:%M:%S) ==="
