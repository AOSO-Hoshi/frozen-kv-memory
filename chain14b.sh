#!/bin/bash
# Wait for the 14B download, then train and measure, with no one watching.
# Billed at 7/hour, so the gap between "download done" and "someone noticed"
# is real money.
#
# [f]etch_ms rather than fetch_ms: pgrep -f matches its own command line.
set -u
cd "$(dirname "$0")"
export PATH=/root/miniconda3/bin:$PATH
LOG=/root/autodl-tmp/fetch14b.log
while ! grep -q "=== DONE" "$LOG" 2>/dev/null; do
  if ! pgrep -f "[f]etch_ms" >/dev/null; then
    echo "download gone with no DONE marker -- stopping"; tail -5 "$LOG"; exit 1
  fi
  sleep 30
done
echo "=== download complete $(date +%H:%M:%S) ==="
du -sh qwen3-14b
MODELS="qwen3-14b" ./cloud_run.sh
