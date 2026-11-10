# Zenodo 新版本（v2）填写内容

在 v1 记录页（https://doi.org/10.5281/zenodo.22897854）点 **New version**，
然后删掉旧 PDF、上传新的 `main.pdf`（建议改名成 `gao2026-compression-ceiling-v2.pdf`）。

## Version

2

## Publication date

2026-09-23

## Description（替换原来的，= v2 摘要）

Compression of a transformer's KV cache is usually reported as one number: the fraction of full-context quality retained. We ask what the missing fraction is made of. Training only 3.67M memory-slot parameters with the backbone frozen, we find that fraction is set by a single property of the task—how much of the answer can be copied from the context rather than inferred.

Splitting predicted tokens on that property separates them sharply: tokens that never appeared recover at +202%, above the verbatim cache they replace, while tokens seen exactly once reach +45%. Posed as a task—recover a random five-digit code hidden in the context—64 learned slots answer none of 150 trials and leave the model worse off than with no context at all, while 84 verbatim entries answer 98% when they are the right ones.

The same property says where each kind of memory pays. On held-out text, 64 learned slots are worth as much as 576 entries selected by H2O, and giving 64 of H2O's 576 entries to the slots recovers 101% of the gap, more than the full cache. On the needle task the slots are dead weight and everything rests on selection: SnapKV with the question in its window answers 99%, the same policies committing before the question arrives answer at most 20%, and our own retrieval, one set of positions shared by every layer, answers 12%. Across 0.6B to 14B each resource alone is flat, while what the slots add behind a fixed selection peaks at 1.7B–4B and halves by 14B.

We also quantify two measurement hazards: fine-tuning the backbone improves the no-context baseline by 0.58 nats, inflating every relative compression metric, and pooling scale points measured under different quantisation and corpora manufactured a clean monotone trend that a controlled sweep does not show. This version corrects a bug in version 1's attention scoring and every conclusion that rested on it.

## Version notes / Additional notes（如果有 "Notes" 字段就填这里；没有就附在 Description 最后）

Version 2 corrects a bug in version 1 and every conclusion that rested on it.

The bug: every policy that selects context by attention — our deployable retrieval and our implementations of H2O and SnapKV — recomputed attention by hand without applying the rotary position embedding to the query, while the cached keys carry it. The scores were therefore not the model's attention (on a test sequence, 1 of 32 top positions agreed). Version 2 rotates the query and checks hand-computed attention against the eager implementation at every layer before any experiment runs.

What changed:
- Baselines: H2O and SnapKV now select per layer and per KV head, as specified. v1's comparison (+89.5% vs a best baseline of +55.8%) becomes +88.6% vs H2O's +77.9%; H2O's selection with our learned slots behind it reaches +101.1%.
- Needle task: v1 said our method placed last because its query was a drafted guess. That was wrong independently of the bug. With correct scoring, the outcome is decided by whether the selector sees the question and selects per head (SnapKV with the question in its window: 99%). v1's Figure 2 is withdrawn.
- Scale: re-evaluated on one card. The peak remains at 1.7B–4B, but 14B no longer falls below 0.6B. The scale result is restated as what the learned state adds behind a fixed selection, which peaks at 1.7B–4B and roughly halves by 14B behind three selectors.
- Decomposition (2x2): deployable row +31.4/+71.9/+75.4 → +32.6/+74.7/+77.8; conclusions unchanged.
- Unaffected: the copyability decomposition, oracle selection, abstract-state results, and the fine-tuning confound.

The full list is in the section "Changes from version 1" at the end of the paper.

AI assistance disclosure: All experiments, analysis, debugging (including identifying the attention-scoring bug) and scientific conclusions are the author's own work. An AI writing assistant (Anthropic's Claude) was used to help draft and edit the English prose.
