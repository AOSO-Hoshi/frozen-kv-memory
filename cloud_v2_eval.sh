#!/bin/bash
# Paper v2, rented GPU: re-evaluate the scale sweep after the scoring fix.
#
# Evaluation only -- nothing trains. Every checkpoint from the controlled sweep
# (bf16, data_clean, pred-len 192) is uploaded from the laptop; the backbones are
# fetched here. What changed since v1 is only how context positions are scored:
# queries are now rotated before being dotted against the cached keys
# (memtok.rope_q), and H2O/SnapKV select per layer and per KV head. The hybrid
# row of the scale table went through the broken scoring, so it is rerun; the
# two component rows did not (rarity anchors and pure abstraction) and are
# rerun anyway so that the table comes from one machine and one code version.
#
# Upload: *.py tests/ fetch_ms.sh cloud_v2_eval.sh data_clean/ and the six
# ckpts named in ck_for() and the 8B block. Never data/ -- it holds real
# credentials.
#
#   bash cloud_v2_eval.sh            # everything, ~1-2 h on one 48 GB card
#   MODELS="qwen3-14b" bash cloud_v2_eval.sh
set -u
cd "$(dirname "$0")"
# AutoDL: conda is not on PATH in a non-interactive shell
[ -x /root/miniconda3/bin/python ] && export PATH=/root/miniconda3/bin:$PATH
# ModelScope stages downloads in its cache; keep that on the data disk, since
# the system disk (30 GB) cannot hold 14B's shards twice
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-$PWD/.mscache}
LOG=${LOG:-./out/cloud_v2}
mkdir -p "$LOG"
export MEMZIP_DATA=data_clean
[ -d data_clean ] || { echo "no ./data_clean -- upload it first"; exit 1; }
[ -d data ] && { echo "./data exists on a rented box -- it must never be uploaded; shred it"; exit 1; }

python - <<'PY' || { echo 'MISSING DEPS'; exit 1; }
import importlib
for m in ("torch", "transformers", "modelscope"):
    mod = importlib.import_module(m)
    print(f"  {m:<14} {getattr(mod, '__version__', '?')}")
import torch
print(" ", torch.cuda.get_device_name(0),
      f"{torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GiB",
      "bf16" if torch.cuda.is_bf16_supported() else "NO BF16 -- stop")
PY

ALL="qwen3-0.6b qwen3-1.7b qwen3-4b qwen3-8b qwen3-14b"
MODELS=${MODELS:-$ALL}
ck_for() {                      # tag rule in train_memtok.py: 0.6b has no prefix
  case "$1" in
    qwen3-0.6b) echo "ckpt/memtok_k32_data_clean_pl192_deep_frozen.pt" ;;
    *)          echo "ckpt/memtok_${1#qwen3-}_k32_data_clean_pl192_deep_frozen.pt" ;;
  esac
}
for M in $MODELS; do
  [ -f "$(ck_for $M)" ] || { echo "missing $(ck_for $M) -- upload before starting"; exit 1; }
done

# Downloads run in one background chain so the card is never idle waiting on
# ModelScope; each finished model leaves a .ok marker the loop below waits on.
# The data disk (50 GB) cannot hold all five (57 GB), so each download waits
# for room and the eval loop deletes a model once it is done with it.
need_gb() { case "$1" in qwen3-0.6b) echo 2;; qwen3-1.7b) echo 5;; qwen3-4b) echo 9;;
            qwen3-8b) echo 17;; qwen3-14b) echo 31;; *) echo 35;; esac; }
free_gb() { df -BG --output=avail . | tail -1 | tr -dc 0-9; }
(
  for M in qwen3-0.6b $MODELS; do
    [ -f "$M/.ok" ] && continue
    # fetched elsewhere (e.g. into /dev/shm by a parallel job that touches .ok)
    [[ " ${SKIP_FETCH:-} " == *" $M "* ]] && continue
    until [ "$(free_gb)" -ge $(( $(need_gb $M) + 3 )) ]; do sleep 20; done
    ./fetch_ms.sh "$(echo ${M#qwen3-} | tr 'a-z' 'A-Z')" > "$LOG/fetch_$M.log" 2>&1 \
      && touch "$M/.ok" || echo "FETCH FAILED $M" >> "$LOG/fetch_failed"
    rm -rf "$MODELSCOPE_CACHE"
  done
) &

wait_for() {
  until [ -f "$1/.ok" ]; do
    grep -q "$1" "$LOG/fetch_failed" 2>/dev/null && return 1
    sleep 20
  done
}

# The scoring must match the model's own attention on this box too, before
# any hour is spent on numbers that depend on it.
wait_for qwen3-0.6b || { echo "0.6B fetch failed"; exit 1; }
python tests/test_attention_match.py > "$LOG/test.log" 2>&1 \
  || { echo "ATTENTION TEST FAILED -- stop"; cat "$LOG/test.log"; exit 1; }
tail -1 "$LOG/test.log"

H="--k 64 --anchors 512 --probe 16 --chunk 0 --hops 8 --n 12"
for M in $MODELS; do
  wait_for "$M" || { echo "skip $M (fetch failed)"; continue; }
  CK=$(ck_for $M)
  for job in hyb base combo; do
    case $job in
      hyb)   cmd="hybrid.py $H" ;;
      base)  cmd="baseline_cmp.py --n 12" ;;
      combo) cmd="combo_cmp.py --n 12" ;;
    esac
    SECONDS=0
    python -u $cmd --ckpt "$CK" --model "./$M" > "$LOG/${job}_$M.log" 2>&1 \
      && echo "ok   ${job}_$M ${SECONDS}s" \
      || { echo "FAIL ${job}_$M"; grep -iE "error|memory" "$LOG/${job}_$M.log" | tail -2; }
  done
  # 8B is needed once more below, at the second learning rate
  if [ "$M" = qwen3-8b ]; then
    python -u hybrid.py $H --ckpt ckpt/memtok_8b_k32_data_clean_s1_pl192_deep_frozen.pt \
      --model ./qwen3-8b > "$LOG/hyb_qwen3-8b_lr2e-3.log" 2>&1 \
      && echo "ok   hyb_qwen3-8b_lr2e-3" || echo "FAIL hyb_qwen3-8b_lr2e-3"
  fi
  rm -rf "$(readlink -f "./$M")" "./$M"   # room for the next download (and RAM)
done

echo "=== summary ==="
grep -H -E "LAST SEG only|NO state|anchors only|576 abstract \(compact" "$LOG"/hyb_*.log
grep -H -E "H2O \(per head\)|ours \(spec" "$LOG"/base_*.log
grep -H -E "slots" "$LOG"/combo_*.log
echo "=== done $(date +%H:%M:%S) -- download out/cloud_v2, then RELEASE the instance ==="
