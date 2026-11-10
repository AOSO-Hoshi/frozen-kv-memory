#!/bin/bash
# Rented-GPU run plan. Billed by the hour, so this is meant to be started once
# and left alone -- no interactive debugging on the clock.
#
# 2026-09-22 rewrite. The previous version would have wasted the session:
#   * it omitted --lr, and the default 1e-3 is the known bad point (-246%
#     against +66% at 5e-3) -- hours of compute for a dead number;
#   * it omitted --no-lora, so the backbone would have adapted and inflated the
#     lower bound, which is exactly the confound the paper is about;
#   * its step/chunk counts differed from the local runs, so the result would
#     not have been comparable to the scale row it is meant to extend.
# Everything below is pinned to the local controlled row (scale3.sh) instead.
# The point of renting is one more scale point, so matching matters more than
# training longer.
set -u
cd "$(dirname "$0")"
LOG=${LOG:-./out/cloud}
mkdir -p "$LOG" ckpt out

# The rented box only ever receives the scrubbed corpus -- real credentials stay
# on the laptop. This must be exported before anything trains, and it also lands
# in every checkpoint name (see the tag rule in train_memtok.py), which is why
# the hybrid section below rebuilds the same string rather than hardcoding it.
export MEMZIP_DATA=${MEMZIP_DATA:-data_clean}
[ -d "$MEMZIP_DATA" ] || { echo "no corpus dir ./$MEMZIP_DATA -- upload it first"; exit 1; }
DOMTAG=""; [ "$MEMZIP_DATA" = "data" ] || DOMTAG="_$MEMZIP_DATA"
echo "corpus: $MEMZIP_DATA ($(ls "$MEMZIP_DATA" | wc -l) files, $(du -sh "$MEMZIP_DATA" | cut -f1))"

echo "=== card check ==="
# Picks the dtype from what the card actually supports rather than from what it
# is called. A rented box may be Ampere (bf16 fine), Ada/Blackwell (fine), or
# Volta (fp16 only) -- and a vGPU slice does not advertise which. Guessing wrong
# either crashes on the first matmul or silently produces NaNs.
python - <<'PY' > .dtype_probe || { echo 'CARD CHECK FAILED -- stop, do not burn hours'; exit 1; }
import torch, sys
cc = torch.cuda.get_device_capability(0)
info = [f"torch {torch.__version__} | cuda {torch.version.cuda}",
        f"device {torch.cuda.get_device_name(0)} | sm_{cc[0]}{cc[1]}",
        f"vram {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB"]
(torch.zeros(8, device="cuda") + 1).sum().item()          # real kernel
dt = "fp16"
try:
    a = torch.randn(64, 64, dtype=torch.bfloat16, device="cuda")
    if torch.isfinite(a @ a).all() and torch.cuda.is_bf16_supported():
        dt = "bf16"
except Exception as e:
    info.append(f"bf16 unusable: {type(e).__name__}")
info.append(f"chosen dtype: {dt}")
print("\n".join(info), file=sys.stderr)
print(dt)
PY
cat .dtype_probe >/dev/null
DT=${DT:-$(cat .dtype_probe)}
echo "using dtype: $DT"

# Check imports before anything is billed. A missing package discovered forty
# minutes into training costs the whole session; discovered here it costs a pip
# install. peft is deliberately absent from this list -- the frozen path no
# longer imports it.
echo "=== deps ==="
python - <<'PY' || { echo 'MISSING DEPS -- fix before training'; exit 1; }
import importlib, sys
missing = []
for m in ("torch", "transformers"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:<14} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        missing.append(f"{m} ({e})")
for m, why in (("bitsandbytes", "needed only if bf16 OOMs and we retry under NF4"),):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:<14} {getattr(mod, '__version__', '?')}")
    except Exception:
        print(f"  {m:<14} MISSING -- {why}")
        print(f"  {'':<14} pip install bitsandbytes")
if missing:
    print("FATAL, missing required packages:", *missing, sep="\n  ")
    sys.exit(1)
PY

# Pinned to scale3.sh. Do not "improve" these on the clock: every one of them
# has to match the local row for the new point to mean anything.
#   --no-lora    freeze the backbone (the whole measurement depends on it)
#   --mem-depth  per-layer bias at memory positions
#   --lr 5e-3    the peak; 1e-3 is catastrophic and is the argparse default
#   --pred-len 192  matches the local row; changing it changes the gap itself
COMMON="--no-lora --mem-depth --lr 5e-3 --pred-len 192 \
        --steps 1000 --chunks 400 --eval-n 24 --k 32 --k-random 16,128 --dtype $DT"

# 1) smoke test: a few minutes, proves the whole path before committing hours.
#
# Two bugs lived here and both defeated the entire point of a smoke test:
#
#   * `python ... | tail -5 || exit` checks tail's exit code, never python's.
#     tail always succeeds, so the guard never fired. Redirect to a file and
#     test the command itself.
#   * it hardcoded 0.6B. On a box where only the target model was downloaded
#     the smoke failed for an irrelevant reason (no such directory) and, thanks
#     to the pipe, training started anyway. Smoke the model actually about to
#     be trained, or the test proves nothing about it.
SMOKE_M=""
for M in ${MODELS:-qwen3-8b qwen3-4b qwen3-1.7b qwen3-0.6b}; do
  [ -d "$M" ] && { SMOKE_M="$M"; break; }
done
[ -n "$SMOKE_M" ] || { echo "no model directory present -- nothing to do"; exit 1; }
echo "=== smoke on $SMOKE_M  $(date +%H:%M:%S) ==="
if ! python -u train_memtok.py $COMMON --steps 40 --chunks 80 --eval-n 12 \
     --model "./$SMOKE_M" > "$LOG/smoke.log" 2>&1; then
  echo 'SMOKE FAILED -- stop, do not burn hours:'; tail -8 "$LOG/smoke.log"; exit 1
fi
tail -3 "$LOG/smoke.log"

# 2) the scale axis -- the reason for renting. 8B is the new point; the smaller
#    ones are rerun only if this box's dtype differs from the local bf16/NF4,
#    since then the local numbers would not be comparable anyway.
# MODELS limits what runs. Default is everything present, but on a rented box
# the small points are usually already measured locally, and re-running them
# just spends money confirming them. e.g. MODELS="qwen3-8b qwen3-4b"
for M in ${MODELS:-qwen3-8b qwen3-4b qwen3-1.7b qwen3-0.6b}; do
  [ -d "$M" ] || { echo "skip $M (not downloaded; ./fetch_ms.sh ${M#qwen3-})"; continue; }
  echo "=== train $M  $(date +%H:%M:%S) ==="
  if python -u train_memtok.py $COMMON --model "./$M" > "$LOG/$M.log" 2>&1; then
    grep -E "^\[ *1000\]" "$LOG/$M.log" | tail -1
  elif grep -qi "out of memory" "$LOG/$M.log"; then
    # Fall back to NF4 rather than lose the point entirely. This does introduce
    # a confound, but it is a measured one: the local row trains 0.6B both ways
    # so the cost of quantisation to the ratio is known and can be subtracted.
    echo "  OOM in $DT -- retrying under NF4"
    python -u train_memtok.py $COMMON --load-4bit --model "./$M" \
      > "$LOG/$M.nf4.log" 2>&1 \
      && grep -E "^\[ *1000\]" "$LOG/$M.nf4.log" | tail -1 \
      || { echo "  FAILED under NF4 too:"; tail -3 "$LOG/$M.nf4.log"; }
  else
    echo "  FAILED:"; grep -iE "error" "$LOG/$M.log" | tail -3
  fi
done

# 3) hybrid on each trained module, in the same session. Running it later means
#    renting again.
echo "=== hybrid $(date +%H:%M:%S) ==="
H="--k 64 --anchors 512 --probe 16 --chunk 0 --hops 8 --n 12"
for M in ${MODELS:-qwen3-8b qwen3-4b qwen3-1.7b qwen3-0.6b}; do
  [ -d "$M" ] || continue
  # tag rule in train_memtok.py: 0.6b carries no model prefix
  case "$M" in
    qwen3-0.6b) CK="ckpt/memtok_k32${DOMTAG}_pl192_deep_frozen.pt" ;;
    *)          CK="ckpt/memtok_${M#qwen3-}_k32${DOMTAG}_pl192_deep_frozen.pt" ;;
  esac
  # a module trained under the NF4 fallback carries _nf4 and must be evaluated
  # the same way -- its slots live in the quantised model's key space
  Q=""
  if [ ! -f "$CK" ]; then
    CKQ="${CK%_deep_frozen.pt}_deep_nf4_frozen.pt"
    if [ -f "$CKQ" ]; then CK="$CKQ"; Q="--load-4bit"; fi
  fi
  [ -f "$CK" ] || { echo "skip $M hybrid (no checkpoint)"; continue; }
  echo "=== hybrid $M  $(date +%H:%M:%S) ==="
  python -u hybrid.py $H --ckpt "$CK" --model "./$M" $Q > "$LOG/hyb_$M.log" 2>&1 \
    || { echo "  FAILED:"; grep -iE "error|memory" "$LOG/hyb_$M.log" | tail -3; continue; }
  grep -E "LAST SEG|NO state|ORACLE \+" "$LOG/hyb_$M.log" | tail -3
done

echo "=== all done $(date +%H:%M:%S) ==="
grep -H -E "^\[ *1000\]" "$LOG"/qwen3-*.log 2>/dev/null
