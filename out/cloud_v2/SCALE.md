| condition | 0.6b | 1.7b | 4b | 8b | 14b |
|---|---|---|---|---|---|
| 576 slots, all abstraction | +55.2 | +54.0 | +60.6 | +59.6 | +54.5 |
| 512 rarity anchors alone | +20.9 | +26.9 | +26.1 | +26.9 | +30.2 |
| 512 speculative anchors, no state | +46.3 | +51.1 | +55.9 | +60.0 | +62.8 |
| 512 spec + 64 slots (rolling) | +78.5 | +87.3 | +89.1 | +85.3 | +78.8 |
| 512 spec + 64 slots (last seg) | +81.5 | +89.9 | +91.6 | +88.5 | +82.6 |
| 512 spec + RANDOM slots | -64.7 | +40.3 | +48.0 | +49.6 | +60.9 |
| 512 oracle + 64 slots (last seg) | +97.9 | +106.8 | +105.1 | +98.0 | +94.0 |
| StreamingLLM 576 | +55.7 | +57.9 | +57.3 | +58.8 | +60.9 |
| SnapKV per head 576 | +54.1 | +57.9 | +59.0 | +63.2 | +64.8 |
| H2O per head 576 | +77.8 | +80.2 | +80.7 | +80.6 | +81.5 |
| H2O one set 576 | +48.3 | +55.9 | +60.7 | +59.4 | +64.3 |
| ours (baseline_cmp) | +78.7 | +89.8 | +90.3 | +88.4 | +81.7 |
| 64 slots alone | +67.5 | +71.7 | +73.9 | +67.7 | +61.7 |
| H2O 512 + 64 slots | +96.3 | +103.0 | +104.1 | +98.0 | +94.0 |
| SnapKV 512 + 64 slots | +83.6 | +91.5 | +91.1 | +87.6 | +84.4 |
| gap (hyb, nats) | 0.978 | 0.962 | 0.979 | 0.968 | 1.026 |
| hybrid absolute nats (lower-CE) | 0.797 | 0.864 | 0.897 | 0.857 | 0.847 |

8B lr2e-3 hybrid last-seg: 84.9  no state: 54.6  random: 44.9

## derived (v2 cloud)

| | 0.6b | 1.7b | 4b | 8b | 14b | 8b_lr2e-3 |
|---|---|---|---|---|---|---|
| 576 slots all abstraction (accum pos) | +59.8 | +53.2 | +59.6 | +59.3 | +54.0 | +57.7 |
| state gain behind spec anchors (recovery) | +35.2 | +38.8 | +35.7 | +28.5 | +19.8 | +30.3 |
| state gain behind spec anchors (nats) | 0.344 | 0.373 | 0.349 | 0.276 | 0.203 | 0.293 |
| slots gain behind H2O (recovery) | +18.5 | +22.8 | +23.4 | +17.4 | +12.5 | - |
| slots gain behind H2O (nats) | 0.180 | 0.220 | 0.228 | 0.168 | 0.128 | - |
| slots gain behind SnapKV (recovery) | +29.5 | +33.6 | +32.1 | +24.4 | +19.6 | - |
| gap | 0.978 | 0.962 | 0.979 | 0.968 | 1.026 | 0.968 |
