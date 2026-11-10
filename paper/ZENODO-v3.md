# Zenodo 新版本（v3）填写内容

v3 不改任何结果、表格和图。只做两件事：源码地址换成 source.hoshichan.moe（v1/v2 里的 GitHub 仓库已经打不开），
重写文末的 AI 协助声明。

在 v2 记录页（https://doi.org/10.5281/zenodo.22913648）点 **New version**，然后：

- 删掉旧 PDF，上传新的 `main.pdf`（建议改名 `gao2026-compression-ceiling-v3.pdf`）
- 再上传一份源码包，这样代码跟着 DOI 永久存档：从 https://source.hoshichan.moe/frozen-kv-memory/ 下载当前的
  `frozen-kv-memory-<提交号>.tar.gz`（必须是包含 v3 源码的那次提交）

## Version

3

## Publication date

发布当天的日期（论文首页印的是 2026-11-10）

## Description（替换原来的，= v3 摘要，只有最后一句和 v2 不同）

Compression of a transformer's KV cache is usually reported as one number: the fraction of full-context quality retained. We ask what the missing fraction is made of. Training only 3.67M memory-slot parameters with the backbone frozen, we find that fraction is set by a single property of the task—how much of the answer can be copied from the context rather than inferred.

Splitting predicted tokens on that property separates them sharply: tokens that never appeared recover at +202%, above the verbatim cache they replace, while tokens seen exactly once reach +45%. Posed as a task—recover a random five-digit code hidden in the context—64 learned slots answer none of 150 trials and leave the model worse off than with no context at all, while 84 verbatim entries answer 98% when they are the right ones.

The same property says where each kind of memory pays. On held-out text, 64 learned slots are worth as much as 576 entries selected by H2O, and giving 64 of H2O's 576 entries to the slots recovers 101% of the gap, more than the full cache. On the needle task the slots are dead weight and everything rests on selection: SnapKV with the question in its window answers 99%, the same policies committing before the question arrives answer at most 20%, and our own retrieval, one set of positions shared by every layer, answers 12%. Across 0.6B to 14B each resource alone is flat, while what the slots add behind a fixed selection peaks at 1.7B–4B and halves by 14B.

We also quantify two measurement hazards: fine-tuning the backbone improves the no-context baseline by 0.58 nats, inflating every relative compression metric, and pooling scale points measured under different quantisation and corpora manufactured a clean monotone trend that a controlled sweep does not show. This text corrects a bug in the first release's attention scoring and every conclusion that rested on it.

## Related works

| Relation | Identifier | Scheme | Resource type |
|---|---|---|---|
| Is supplemented by | https://source.hoshichan.moe/frozen-kv-memory/ | URL | Software |
| Is supplemented by | https://github.com/AOSO-Hoshi/frozen-kv-memory | URL | Software |

（第二行是镜像，可以不加。旧的 `github.com/HoshiGT/...` 那一行删掉。）

## Additional notes（替换 v2 的整段）

Version 3 changes no result, table or figure. It gives the code a new address, https://source.hoshichan.moe/frozen-kv-memory/, because the repository that versions 1 and 2 link to is no longer reachable, and it rewrites the statement on the use of AI assistance, which in version 2 did not describe the division of work accurately.

Use of AI assistance: This paper was produced by the author working with AI assistants, and their part in it was large: Anthropic's Claude wrote most of the code, ran the experiments, analysed the results and drafted the text, and xAI's Grok carried out the literature search. The author set the direction, chose the experiments, reviewed the results and the text, and takes full responsibility for the paper.

Version 2 corrected a bug in version 1's attention scoring and every conclusion that rested on it; the full list is in the section "Changes from version 1" at the end of the paper.

## 顺手改一下 v2 那条记录（Edit，不是 New version）

v2 记录页的 Additional notes 末尾有一段旧的 "AI assistance disclosure"。把那一段换成：

The statement on AI assistance printed in this version does not describe the division of work accurately; it is replaced in version 3.
