# 第二篇方向的前人工作（2026-09-23）

对照 `HANDOFF-paper2-prior-art.md`。排除 `github.com/HoshiGT/frozen-kv-memory` 和 Zenodo `10.5281/zenodo.22897854`。每条 arXiv 的标题都对过 abs 页，对不上的不收。

已知近邻（交接里点过名，这里不展开）：Self-Pruned KV `2605.14037`（学的是写入，主干一起训）、SpeCache `2503.16163`（预取，不训练）、Quest `2406.10774`、InfLLM `2402.04617`（按块打分，不训练）。下面调回侧比 Quest / SpeCache 更近，监督侧比它们更近。

重合程度按交接的三档：相同 / 部分 / 不同。三个方向都没有标成「相同」的。

分类口径：调回 = 按当前查询决定读哪些已有 KV；驱逐 = 决定哪些 KV 留下、其余丢掉；写入 = 决定要不要进缓存。

---

## Q1 学出来的 KV 选择器

监督来自下游 loss、注意力蒸馏，或未来注意力。没有一条用的是「知道答案时长什么样」的那次选择。

| ID | 名称 | 一句话 | 相关 | 重合 | 开源或部署 |
|---|---|---|---|---|---|
| [2609.13141](https://arxiv.org/abs/2609.13141) | SAS | 主干冻结，只训 AttnGate；按查询对历史块做 Top-K；训练时把连续分数灌进注意力，用语言模型 loss 更新选择器。对照 SeerAttention-R 的注意力蒸馏，紧预算下下游 loss 更好 | A | 部分，调回侧最近 | 摘要未给代码 |
| [2410.13276](https://arxiv.org/abs/2410.13276) | SeerAttention | 可学习的块级门，原权重冻结，自蒸馏去拟合稠密注意力；推理时按门控选块 | A | 部分 | [microsoft/SeerAttention](https://github.com/microsoft/SeerAttention) |
| [2506.08889](https://arxiv.org/abs/2506.08889) | SeerAttention-R | 同一套门，去掉 query pooling 以适配逐 token 解码；插件，不改原参数；0.4B token 训完，推理模型上近乎无损 | A | 部分 | 同上仓库 |
| [2510.24606](https://arxiv.org/abs/2510.24606) | DHSA | 主干冻结。边界预测器的软标签来自稠密注意力的局部质量差；分块后再按块相似度扩成 token 索引 | A、B | 部分 | [xiongsiheng/DHSA](https://github.com/xiongsiheng/DHSA) |
| [2606.04511](https://arxiv.org/abs/2606.04511) | SparDA | 多一个 Forecast 投影，预测下一层要的 KV 块并预取；只训 Forecast（<0.5% 参数），监督是原选择器的注意力分布 | A | 部分，比 SpeCache 多了训练 | 摘要给了代码链接 |
| [2609.20734](https://arxiv.org/abs/2609.20734) | On-Demand Attention | 只训一个 recall head，预训练权重不动，历史 KV 全留着；预测「这一步全局读值不值得」，值得才读全部 | A、B | 部分。学的是何时调回，调回的是整段历史 | vLLM 里做了条件执行；摘要未给独立仓库 |
| [2607.05061](https://arxiv.org/abs/2607.05061) | KVpop | 固定预算的驱逐。打分器直接监督留或丢，目标是未来注意力质量，不物化稠密注意力图；打分可以推迟几步，用到近未来上下文 | A | 部分。监督最贴「未来注意力」，动作是驱逐 | 摘要未给代码 |
| [2602.03203](https://arxiv.org/abs/2602.03203) | ForesightKV | Golden Eviction 用未来注意力找出该丢的 KV，再以成对排序损失蒸馏，并用 GRPO 补低熵 token 上的损失 | A | 部分。oracle 是未来注意力，动作是驱逐 | 摘要未给代码 |
| [2602.10238](https://arxiv.org/abs/2602.10238) | Learning to Evict / KVP | 每个头一个小 RL agent，只看 K/V，奖励来自未来效用；不改 LLM。评了 RULER，也评了多轮对话 OASST2-4k | A | 部分。驱逐 | [apple/ml-learning-to-evict](https://github.com/apple/ml-learning-to-evict)，ICML 2026 |
| [2603.10899](https://arxiv.org/abs/2603.10899) | LookaheadKV | 可学习的 lookahead token + LoRA，预测未来重要性再驱逐，不先生成草稿 | A、B | 部分。驱逐；「草稿当查询」被它绕开了 | [SamsungLabs/LookaheadKV](https://github.com/SamsungLabs/LookaheadKV)，ICLR 2026 |
| [2601.17668](https://arxiv.org/abs/2601.17668) | Fast KVzip | 冻结权重的 LLM 上加 sink-attention 门，用重建目标挑该留的 KV；训练只跑前向，不反传 LLM | A | 部分。驱逐，监督是重建 | 开源，接在 [snu-mllab/KVzip](https://github.com/snu-mllab/kvzip) |
| [2410.01805](https://arxiv.org/abs/2410.01805) | Locret | 冻结主干，可学习 retaining head 给 KV 单元打因果重要性，按块 prefill 驱逐 | A | 部分。驱逐 | 摘要未给仓库 |
| [2605.06676](https://arxiv.org/abs/2605.06676) | LKV | 头预算和 token 选择一起学，自蒸馏，教师冻结；token 选择与查询无关 | A | 部分。驱逐 | 摘要未给代码 |
| [2502.11089](https://arxiv.org/abs/2502.11089) | NSA | 预训练就带上三路：压缩、选择、滑窗。选择分和模型一起训，主干不冻结 | A、Q4 | 部分 | 论文摘要未给仓库。同谱系的产品压缩见 V4.1 |
| [2502.13189](https://arxiv.org/abs/2502.13189) | MoBA | 查询自己学着路由到 KV 块；门是无参数 top-k，模型一起训 | A、Q4 | 部分 | 已用于 Kimi 长上下文；[MoonshotAI/MoBA](https://github.com/MoonshotAI/MoBA) |
| [2505.00315](https://arxiv.org/abs/2505.00315) | MoSA | expert-choice 路由，每个头动态选 token，和模型一起训。实验规模小于上面几篇 | A | 部分 | 摘要未给代码 |
| [2506.17121](https://arxiv.org/abs/2506.17121) | PruLong | 端到端学哪些头保留全量 KV、哪些头可以丢。头级，不是按查询选 token | A、B | 部分 | 摘要未给代码 |
| [2410.10819](https://arxiv.org/abs/2410.10819) | DuoAttention | 用合成数据做一次优化，把头分成 retrieval / streaming。分完就固定，不随查询变 | A、B | 部分 | [mit-han-lab/duo-attention](https://github.com/mit-han-lab/duo-attention) |
| [2508.17032](https://arxiv.org/abs/2508.17032) | Learned Structure in Cartridges | 机制观察：Cartridge 的 key 像稳定的检索路由，压缩主要发生在 value | A | 部分。索引长在 KV 自己身上，没有外挂选择器，也没有答案 oracle 标签 | 研讨会论文 |
| [2609.03430](https://arxiv.org/abs/2609.03430) | Random Attention | 推理轨迹上按头随机留 KV，打平 SnapKV、R-KV 一类不训练的驱逐。论文自己写了：没和需要训练的选择器比 | A | 不同。提醒：推理任务上「更聪明的选择」空间可能很窄 | [SalesforceAIResearch/Random-Attention](https://github.com/SalesforceAIResearch/Random-Attention) |
| [2505.13109](https://arxiv.org/abs/2505.13109) | FreeKV | 训练免费的 KV 调回系统：投机检索 + CPU/GPU 混合布局。2025–2026 这条线的系统代表 | A | 不同。比 Quest 新，仍然不训练 | 摘要给了代码 |

RocketKV `2502.14051` 的 Exact-TopK 是稀疏注意力的上界，不是训练标签。第一篇调研里已经记过。

---

## Q2 按任务或查询换策略

没有找到「先判断这是问答还是闲聊，再在分散 token 和连续段之间切换」的工作。近的是：这一步要不要读、这个头用稀疏还是全量、压缩时用哪一种代理查询。

| ID | 名称 | 一句话 | 相关 | 重合 | 开源或部署 |
|---|---|---|---|---|---|
| [2609.20734](https://arxiv.org/abs/2609.20734) | On-Demand Attention | 生成过程中按预测收益在局部注意力和一次全局读之间切。见 Q1 | B、A | 部分 | 见 Q1 |
| [2601.17367](https://arxiv.org/abs/2601.17367) | Elastic Attention | 预训练模型上加一个轻量 Attention Router，按输入把每个头派到全量或稀疏。约 12 小时训练 | B | 部分。切的是头的计算模式，输入自适应 | 摘要未给代码 |
| [2609.20888](https://arxiv.org/abs/2609.20888) | Elastic Threshold Attention | 从 query 预测动态阈值：难的检索/推理步留接近稠密的上下文，寻常 token 剪掉。1.45B 上和模型一起训 | B | 部分。粒度是解码步，模型很小，主干不冻结 | 摘要未给代码 |
| [2608.00902](https://arxiv.org/abs/2608.00902) | Practical Online KV Cache Compaction | 智能体在线压缩。比较边界查询、重复 prefill、推迟到未来生成再当查询。立刻压经常伤质量，用未来查询能补回一截。训练免费 | B | 部分。直接比过「以后生成的内容当查询」 | 摘要未给代码 |
| [2603.10899](https://arxiv.org/abs/2603.10899) | LookaheadKV | 用学出来的未来 token 代替显式草稿。见 Q1 | B | 部分 | 开源 |
| [2407.09450](https://arxiv.org/abs/2407.09450) | EM-LLM | 不微调。用贝叶斯惊讶分事件，检索时同时用相似度和时间上相邻的段。LongBench 上胜过 RAG，也能在千万 token 上检索 | B、C | 部分。连续段和分散相似度都用了，没有按任务切换 | 论文，未见产品 |
| [2412.14838](https://arxiv.org/abs/2412.14838) | DynamicKV | 训练免费。按层调整保留 token 数，信号来自最近 token 的注意力，作者称为 task-aware | B | 部分。预算随任务变，策略族不变 | 摘要未给代码 |
| [2501.15113](https://arxiv.org/abs/2501.15113) | Task-KV | 按任务语义把头发成异质头 / 非异质头，异质头给满 KV | B | 部分。头预算随任务变 | 摘要未给代码 |
| [2503.04973](https://arxiv.org/abs/2503.04973) | Beyond RAG: Task-Aware KV Cache Compression | 按任务（不是按每条查询）预先压一份 KV，和 RAG、与任务无关的压缩比。LongBench v2 上压 30× 仍高于 RAG | B、C | 部分。任务条件的压缩，查询到来前就压完 | 摘要未给代码 |
| [2510.24606](https://arxiv.org/abs/2510.24606) | DHSA | 学出来的块边界 + 块内 token。见 Q1 | B | 部分 | 开源 |

FastGen `2310.01801` 按头混合 local / special-token / 全量，策略来自离线画像，不随当前请求改。第一篇调研里已有。

---

## Q3 落地，以及 KV 和文本记忆的比较

陪伴或个人助理产品里，用 KV 当长期生活记忆的，没找到。找到的是服务系统：把多轮对话的 KV 存下来避免重算。文本层记忆（MemGPT / Letta、mem0、MemoryBank）按交接排除。

| ID 或链接 | 名称 | 一句话 | 相关 | 重合 | 开源或部署 |
|---|---|---|---|---|---|
| [2407.00079](https://arxiv.org/abs/2407.00079) | Mooncake | Kimi 的服务系统。跨请求的 KV 池，预填充/解码分离，DRAM 和 SSD 分层。有基于预测的提前拒绝。内容不做语义压缩或挑选 | C | 部分。KV 跨会话复用，选择的是放哪一层 | 已部署；[kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake) |
| [https://github.com/LMCache/LMCache](https://github.com/LMCache/LMCache) | LMCache | KV 管理层：CPU、磁盘、Redis、S3、Mooncake 都能当后端，引擎崩溃 KV 还在。非前缀复用靠 CacheBlend | C | 部分 | 开源，接入 vLLM 生产栈 |
| [2405.16444](https://arxiv.org/abs/2405.16444) | CacheBlend | RAG 里非前缀文本的 KV 直接复用，只重算一小部分 token 把交叉注意力补上。选择的是「哪些 token 重算」 | C | 部分。有选择，目的是拼缓存 | EuroSys 2025；进了 LMCache |
| [https://www.usenix.org/conference/atc24/presentation/gao-bin-cost](https://www.usenix.org/conference/atc24/presentation/gao-bin-cost) | CachedAttention | 多轮对话把历史 KV 放进分层存储，下一轮读回来。上下文溢出时截断，并解开位置编码。后续 FAST 2026 论文写明这套实现闭源 | C | 部分。持久化，截断是按窗口，不是按答案挑 | 会议系统，闭源。华为云作者 |
| [2310.07240](https://arxiv.org/abs/2310.07240) | CacheGen | 把 KV 编成码流传输，带宽变了就改压缩档。压缩是编码，不是挑记忆 | C | 不同。交接点名的这一类，补在这里 | 论文系统 |
| [2609.16215](https://arxiv.org/abs/2609.16215) | Where Should the KV Cache Live? | 模拟器里比较聊天、智能体、文档问答的 KV 该放 GPU、CPU 还是 SSD。收益主要来自分层容量 | C | 不同。放置策略，不压缩内容 | 模拟，未部署 |
| [2506.06266](https://arxiv.org/abs/2506.06266) | Cartridges | 离线把语料蒸馏成一小段可训练 KV，推理时当前缀加载。自合成问答 + 上下文蒸馏。记忆质量对齐 ICL，显存约 38.6× 更小 | C | 部分。KV 级文档记忆，对象是语料，主干侧是往 KV 里训 | [HazyResearch/cartridges](https://github.com/HazyResearch/cartridges) |
| [2606.04557](https://arxiv.org/abs/2606.04557) | Cartridges at Scale | 每篇文档一个 cartridge，再用检索决定加载哪几个。和常规 RAG 比，准确率打平或更好，prompt token 少 3–4× | C | 部分。KV 记忆 + 文本检索选筒 | 摘要未给代码 |
| [2412.15605](https://arxiv.org/abs/2412.15605) | Don't Do RAG (CAG) | 知识库不大时，整库预计算 KV，跳过实时检索。和稀疏/稠密 RAG 比了 BERTScore | C | 部分。直接比过 KV 缓存和 RAG | 论文 |
| [2503.04973](https://arxiv.org/abs/2503.04973) | Beyond RAG | 任务条件的 KV 压缩对 RAG。见 Q2 | C、B | 部分 | 论文 |
| [2609.17346](https://arxiv.org/abs/2609.17346) | Where Should a Document Live | 同一批知识任务上比：原文上下文、KV（Cartridges / Compaction）、微调进参数。oracle 设定下 Cartridges 几乎每个预算都最准 | C | 部分。KV 表征对参数、对全文 | 论文，2026-09-15 |
| [2407.09450](https://arxiv.org/abs/2407.09450) | EM-LLM | KV 分段检索，论文里赢了 RAG。见 Q2。不是产品 | C、B | 部分 | 论文 |

---

## Q4 MoE，以及 Qwen3.6-35B-A3B

没找到「同一个驱逐器在 MoE 和稠密模型上行为不同」的正面实验。找到的是：MoE 模型把 KV 的形状在训练时就改了，驱逐文献的假设（每层都有全量 KV）对不上。

| ID | 名称 | 一句话 | 相关 | 重合 | 开源或部署 |
|---|---|---|---|---|---|
| [2405.04434](https://arxiv.org/abs/2405.04434) | DeepSeek-V2 | MoE + MLA。KV 压进低秩潜变量，相对 DeepSeek 67B 的 KV 减少 93.3%，和模型一起训 | Q4 | 部分。架构级压缩，已部署的 MoE 做法 | 模型已发布 |
| [2609.19969](https://arxiv.org/abs/2609.19969) | DeepSeek-V4.1-Flash | 552B MoE。CSA2 跨层复用 KV，再加 FP4。常驻 HBM 的全局 KV 写到约 890 字节/token，约为 V4-Flash 的 1/4 | Q4 | 部分。产品架构里的 KV 压缩，选择发生在稀疏注意力里 | 模型论文，2026-09-17 |
| [2502.11089](https://arxiv.org/abs/2502.11089) | NSA | 压缩支路加选择支路，和主干一起预训练。见 Q1 | Q4、A | 部分 | 见 Q1 |
| [2502.13189](https://arxiv.org/abs/2502.13189) | MoBA | Kimi（MoE 服务）里用过的块路由注意力。见 Q1 | Q4、A | 部分 | 已部署，开源 |
| [2511.00819](https://arxiv.org/abs/2511.00819) | NSA + 潜变量注意力 | 滑窗支路换成 MLA，压缩和选择支路用分组潜变量注意力。相对 NSA，KV 显存再降 50% | Q4 | 部分。把「选择」和「潜变量压缩」接到一起 | 摘要未给代码 |
| [2608.23834](https://arxiv.org/abs/2608.23834) | Minima-KV | Qwen3.6-27B（稠密、混合注意力）上，近期页 FP8、旧页 TQ3，页不删。单卡 96GB 上记账约 18.3 KiB/token。对象是混合模型里剩下的全注意力层 | Q4、C | 部分。同一家族的 KV 压缩，模型是 27B 稠密，35B-A3B 上没看到同样实验 | 公司部署记账；摘要未给代码 |
| [2609.18063](https://arxiv.org/abs/2609.18063) | The Other Half of the Memory Wall | 35B 级 MoE 从 SSD 供专家，训练路由预测。点名 Qwen3.6 的 softmax-top-k 路由。KV 被当作已经被 MLA / 稀疏注意力 / 线性状态处理过的那一半 | C、Q4 | 部分。8GB 上先撞上的是专家权重 | 摘要未给代码 |
| [2609.20734](https://arxiv.org/abs/2609.20734) | On-Demand Attention | 实验包含混合注意力主干（Qwen、Gemma 一族）。摘要没写是不是 35B-A3B | Q4、A、B | 部分 | 见 Q1 |

官方模型卡 [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)（2026-04 发布说明）：40 层，排成 10 组「3× Gated DeltaNet + 1× Gated Attention」，再接 MoE。全注意力 16 个 Q 头、2 个 KV 头、头维 256。KV 只在这 10 层出现。30 层 DeltaNet 是固定大小的循环状态。256 个专家，每 token 8 个路由专家加 1 个共享专家。上下文 262144，可扩到约 1M。

---

## 未收进表

- 二手页面上的 “Judge Q”（2025-09，可训练 soft query 做驱逐）没有对上的 arXiv，丢掉。
- OracleKV：ICLR 2026 匿名评审稿，[openreview](https://openreview.net/pdf?id=k3IAjIsfyw)。在上下文末尾拼一段「未来问题的表面统计」，再跑现成驱逐器。未正式发表，也不是训练选择器。
- Pensieve（EuroSys 2025）被后续 USENIX 论文和 CachedAttention、Mooncake 并列，当作多轮 KV 分层缓存。这次没打开原文，不单列。

---

## 总判断

A 的宽问题已经有人做了，2026 年还很挤。冻结主干、只训一个轻量模块、用下游 loss 或注意力蒸馏来决定读哪些块：SAS、SeerAttention、SeerAttention-R、DHSA、SparDA。用未来注意力当监督：KVpop、ForesightKV、KVP、LookaheadKV，这四条的动作是驱逐，丢了就没有。和模型一起训选择：NSA、MoBA，而且 MoBA 已经在 Kimi 里。还没看到人做完的，是交接里那条窄定义：监督标签来自「知道答案时该调哪 512 条原文 KV」，选择器从卸载的原文 KV 里按查询调回，压缩槽只负责记住位置。Cartridges 的机制笔记（key 像路由、value 存内容）说明「索引可以长在 KV 里」，他们训的是缓存本身。Random Attention 则提醒，推理轨迹上随机留和启发式驱逐差不多，选择器的空间更可能在 needle 这种价值集中的任务上，这和第一篇的排序翻转是同一件事。

B 还有一块，但是要收窄。没有人先分类「问答还是续写」再换检索几何。已经有人做的是相邻动作：这一步要不要全局读（ODA）、这个头用稀疏还是全量（Elastic Attention）、难步多留上下文（ETA）、压缩时用边界、重复 prefill，还是未来生成当查询（2608.00902）、事件边界再加时间相邻段（EM-LLM、DHSA）。草稿当查询这条，LookaheadKV 用学出来的未来 token 替代了显式草稿，2608.00902 则在智能体上量过「等未来查询再压」有多值钱。剩下的空间是：在冻结主干上，用第一篇那种价值集中 / 价值弥散的划分，显式切换「分散 token」和「连续段」，以及「用户原话」和「草稿」。

C 作为「陪伴产品里的 KV 生活记忆」是空的；作为「KV 跨轮留下去」是满的。Mooncake、LMCache、CacheBlend、CachedAttention 都在做持久化和复用，选择发生在存储层或重算哪些 token，不发生在「这段人生要不要记住」。和文本 RAG 的正面比较已经有 CAG、Beyond RAG、Cartridges、2609.17346，结论随预算和任务变，没有一边倒。落到 Qwen3.6-35B-A3B 上，题还会改：官方结构里只有 10/40 层有 KV，另外 30 层是 Gated DeltaNet 的循环状态；8GB 上先放不下的是专家权重（2609.18063），不是一份稠密模型那种随上下文线性涨的 KV。Minima-KV 压的是同家族 27B 稠密模型剩下的全注意力 KV，35B-A3B 上没有对上的实验。若还做 C，对象应是那 10 层的 KV 加上 30 层的循环状态，而不是把第一篇的槽位原样套到每一层。
