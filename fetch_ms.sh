#!/bin/bash
# Pull a Qwen3 checkpoint with ModelScope's own client.
#
# Not curl against hf-mirror: its redirects break ranged requests, and a shard
# once arrived at 6417MB against an index total of 7.49GB for all three -- silent
# corruption that only shows up as a load error much later. ModelScope is slower
# from here but correct, and on a rented box inside CN it is usually faster too.
#
#   ./fetch_ms.sh 8B        -> ./qwen3-8b
set -u
SPEC=${1:?usage: fetch_ms.sh <size, e.g. 8B> [target dir]}
# optional target, e.g. /dev/shm/qwen3-14b to download into RAM alongside
# another download on disk
D=${2:-./qwen3-$(echo "$SPEC" | tr 'A-Z' 'a-z')}
cd "$(dirname "$0")"
source ~/ctfenv/bin/activate 2>/dev/null || true
python -c "import modelscope" 2>/dev/null || pip install -q modelscope

echo "=== Qwen/Qwen3-$SPEC -> $D  $(date +%H:%M:%S) ==="
python - <<PY
from modelscope import snapshot_download
p = snapshot_download("Qwen/Qwen3-$SPEC", local_dir="$D")
print("downloaded to", p)
PY

# A truncated shard loads fine as a file and fails only at inference, so check
# the sizes against the index before trusting it.
python - <<PY
import json, os, sys
d = "$D"
idx = os.path.join(d, "model.safetensors.index.json")
if not os.path.exists(idx):
    files = [f for f in os.listdir(d) if f.endswith(".safetensors")]
    print("single-shard:", files); sys.exit(0)
shards = sorted(set(json.load(open(idx))["weight_map"].values()))
bad = [s for s in shards if not os.path.exists(os.path.join(d, s))]
print(f"{len(shards)} shards, total {sum(os.path.getsize(os.path.join(d,s)) for s in shards if os.path.exists(os.path.join(d,s)))/2**30:.2f} GiB")
if bad:
    print("MISSING:", bad); sys.exit(1)
print("all shards present")
PY
echo "=== DONE $(date +%H:%M:%S) ==="
