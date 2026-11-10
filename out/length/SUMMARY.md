# Selector headroom vs context length (paper-2 direction A), 2026-09-23

0.6B, seed 0 ckpt (memtok_k32_s0_deep_frozen.pt), data/, 12 windows, budget 576
(512 selected + 64 learned slots, last-segment state). `combo_cmp.py --hops 8/16/32`.

| | 4096 | 8192 | 16384 |
|---|---|---|---|
| future attention per head + slots (answer-aware ceiling) | +118.8 | +113.6 | +105.0 |
| token-match oracle, one shared set + slots | +103.3 | +97.2 | +73.0 |
| H2O per head 512 + slots | +101.0 | +87.2 | +70.6 |
| SnapKV per head 512 + slots | +90.0 | +83.1 | +70.4 |
| H2O per head 576 alone | +77.9 | +63.2 | +49.6 |
| 64 slots alone | +75.9 | +70.4 | +62.9 |
| gap (nats) | 0.974 | 1.076 | 1.282 |
| **headroom = future ceiling - H2O+slots** | **17.8** | **26.4** | **34.4** |
| headroom in nats | 0.173 | 0.284 | 0.441 |

- The shared-set token oracle is NOT a ceiling for per-head selectors: it
  collapses at 16k (+73.0, 512 shared positions cover 3% of the context).
  Earlier conclusion "only 2.3 points of room at 4096" was an artefact of it.
- Headroom grows with length in both units.
- From 8k on, 64 learned slots alone beat 576 H2O entries.
- Not yet ruled out: per-step exact top-k per head at decode time (Quest's
  upper bound, training-free, uses the true current query). If it reaches the
  ceiling, a learned selector buys only efficiency. That is the next test.
- Single seed; H2O and the oracles are deterministic, slots vary ~1 point.
- Logs: combo_{4096,8192,16384}_s0.log; v1_oracle/ has the runs without the
  per-head ceiling.
