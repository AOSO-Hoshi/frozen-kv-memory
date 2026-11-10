# v2 rerun summary (0.6B, local 4060, after RoPE fix + per-head baselines)

Generated from out/v2/*_b1_s{0,1,2}.log. v1 numbers for comparison in parentheses.

## Needle (4096 ctx, 3 seeds x 5 depths x 10 = 150 trials, budget 576)

| condition | EM | recovery | per-seed EM |
|---|---|---|---|
| SnapKV, question in window, per head | 99.3% | +99.6% | 100/98/100 |
| oracle anchors + slots | 98.0% (98) | +94.4% | unchanged, not attention-based |
| StreamingLLM | 20.0% (20) | +15.4% | unchanged, not attention-based |
| ours anchors (question as query) | 13.3% (5) | +14.0% (+14.9) | 14/12/14 |
| ours hybrid (anchors + 64 slots) | 12.0% (3) | +13.2% (+14.3) | 16/10/10 |
| SnapKV, per head, no question | 8.7% | +12.7% | |
| SnapKV, one set (v1-style, fixed RoPE) | 4.7% (24, buggy) | +0.9% | |
| H2O, per head | 0% | +2.2% | |
| H2O, one set (v1-style, fixed RoPE) | 0% (5, buggy) | -6.4% | |
| 64 abstract slots | 0% (0) | -28.6% (-28.6) | unchanged |

By depth (EM, 3-seed mean): SnapKV-q 100/100/100/97/100; ours anchors 0/0/0/0/67;
StreamingLLM only at 0.9 (100). Query-agnostic eviction finds almost nothing below 0.9.

## Natural text (4096 ctx, 12 windows, budget 576)

| policy | recovery | v1 |
|---|---|---|
| ours (spec anchors + slots) | +88.6% (88.7/88.1/89.1) | +89.5 |
| ours, oracle anchors | +103.2% | +103.2 |
| H2O per head | +77.9% | (not run) |
| StreamingLLM | +55.8% | +55.8 |
| SnapKV per head | +54.2% | (not run) |
| SnapKV one set | +54.3% | +53.1 (buggy) |
| H2O one set | +48.3% | +50.5 (buggy) |

Lead over best baseline: v1 33.7 points -> v2 10.7 points.

## Strong selector + learned slots (combo_cmp.py, natural text, 4096, budget 576)

| condition | mean | per seed |
|---|---|---|
| 64 slots only (last segment) | +75.4% | +75.9 / +74.7 / +75.5 |
| H2O per head, 576 entries | +77.9% | +77.9 / +77.9 / +77.9 |
| H2O per head, 512 + 64 slots | +101.1% | +101.0 / +100.7 / +101.6 |
| SnapKV per head, 576 entries | +54.2% | +54.2 / +54.2 / +54.2 |
| SnapKV per head, 512 + 64 slots | +90.2% | +90.0 / +90.1 / +90.5 |

## Levers: anchor budget (hybrid.py, 4096, 8 hops, probe 16, n 12, LAST SEG state)

| anchors | mean | per seed |
|---|---|---|
| 512 | +89.2% | 88.8 / 89.2 / 89.7 |
| 1024 | +96.8% | 97.0 / 96.9 / 96.4 |

v1's +87.8 -> +96.6 came from a checkpoint that no longer exists (the current
default ckpt scores -2.3% as pure abstraction; logs in levers_badckpt/).

## Needle: strong selector + slots (needle_q_s*.log; all other conditions byte-identical to needle_b1_s*)

| condition | EM | recovery |
|---|---|---|
| SnapKV per head, question in window, 576 | 99.3% (100/98/100) | +99.6% |
| same, 512 + 64 slots | 96.7% (98/96/96) | +95.4% |

Slots cost ~3 EM / ~4 recovery on the needle, against +36.0 on natural text behind SnapKV.
