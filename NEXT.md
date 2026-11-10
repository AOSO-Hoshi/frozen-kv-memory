# TODO

昨天那份已全部做完（预算杠杆、天花板成因、混合架构、检索策略）。这是 2026-09-21 下午起的新队列。

## 当前最好的配置

```bash
python hybrid.py --k 64 --anchors 512 --probe 128 --hops 8
# 7:1 压缩，speculative 检索 +86.8%，纯抽象只有 +69.4%
```

架构：64 抽象槽常驻 + 原文 KV offload + 按 speculative query 调入 512 条。

## 已完成（2026-09-21）

**上午—下午**
1. ~~anchor-aware 训练~~ → 负面，分工是自发的
2. ~~1.7B / 4B 跨规模~~ → 三个规模点全部持平，规模线结案
3. ~~打包~~ → `memory.py` + `README-usage.md`
4. ~~报告页~~ → v4（已修正 64× 的错误说法）

**傍晚—晚上**
5. ~~有效窗口~~ → 只有一段；滚动在稀释，`memory.py` 已改成一次压缩
6. ~~分层 / 主题召回~~ → 都不如不加
7. ~~跨域~~ → 状态域特定（对话 +53.5% / 小说 −54.4%），重训 12 分钟可解
8. ~~三种子方差~~ → k16 极差 51 点，混合架构 1.5–2.3 点；结论已按方差重新分类
9. ~~文献调研~~ → 交给 Grok，`docs/LITERATURE-*.md`，Q1/Q2 仍空、Q3 需降级
10. ~~FreeToken 接口调查~~ → `docs/freetoken-integration.md`，缺一个 embedding 入口

## 下一步

**A. 论文 —— 草稿已写并编译（2026-09-22，`paper/`，13 页）**

`pdflatex main && bibtex main && pdflatex main x2`，texlive 已装好。
`\TODO` / `\CHECK` 全清零，0 error / 0 undefined / 0 overfull。
细节见 `paper/README.md`。

**写草稿 + 补实验一共推翻了六条自己的旧结论**（详见 README 第四点五节）：
1. 加法分解表混了两组实验 → 补跑 2×2 缺的那格（oracle × last-seg = **+98.0%**）
2. `+231%` 是单次高估 → 3 seed 是 **+202%**，且极差 11.3（小分母放大噪声）
3. "4B 步数不同所以不可比" → **步数是对的**，真因是 pred-len 和 NF4
4. "三个杠杆都推不动天花板" → **规模推得动**（+82.3→+94.5，噪声仅 1.0），
   但**不推动检索**（+50.6→+54.1）→ 成了 Contribution 2 的独立证据
5. "精确回忆不可压缩" → **顺序是任务属性，水平不是**；最难档位提升最多（+25.7）
6. "1.7B 差是因为只训了 400 步" → 训满照样 −101.5%，**两个种子一致**，
   是这个规模点在多跳下真的发散（单跳正常 +86.7%）

下一步仍是补实验而不是补文字：gap 大的 benchmark、baseline、8B。

主贡献现在是三个诊断（原两个 + 规模分离），不是新方法：
- **(a)** 按 token 可复制性拆解天花板
- **(b)** 检索 / 压缩 / oracle 的加法分解

(c) 冻结主干的测量论点**要降级并正面引用**
CCM [2312.03414]（观察过同一现象，归因为训练过拟合）和
No Mean Feat [2510.20797]（用了几乎相同的归一化公式）。
用"据我们所知"，不用"首次"。

**不要**写"我们独立想到、实现中才发现前人工作"——时间上我们在后面，
这种声明会被读成辩解，而且诊断类工作本来就不需要主张方法首创权。
那个过程属于 README 和 git 历史，不属于论文。

~~gap 大的任务~~ → 2026-09-22 已做 `needle.py`（合成，gap 2.52 nats vs 对话 0.95）。
抽象状态 EM 0/150 且比无上下文还差；oracle 98% vs 可部署 5%。
**但那是自制合成任务**，LongBench / RULER 仍然缺——它们测的是真实任务
那种"部分可复制"的混合，而我们只验证了两个端点。

~~H2O / SnapKV baseline~~ → 2026-09-22 已做 `baseline_cmp.py`：
同样 576 预算，StreamingLLM +55.8 / SnapKV +53.1 / H2O +50.5 vs **ours +89.5**。
★ 最简单的 StreamingLLM 反而最强，印证"价值弥散"。
★ 比较不对等（我们训 3.67M/12 分钟），论文已明写。

**会议投稿：封存**（2026-09-22，Hoshi 决定）。arXiv endorsement 仍卡着，
不影响论文本身；想起来再说。注意若将来投 ACL 系列，它们有**匿名期**规定，
截稿前一段时间不能新发 preprint，放 arXiv 的时机要错开。

**现在剩下的缺口**（按优先级）：
1. **LongBench / RULER** —— needle 是自制合成任务，只验证了"全可复制"这个端点；
   真实任务是"部分可复制"的混合，两种 regime 的边界在哪没测
2. **needle 上跑 baseline 对比** —— 预测 StreamingLLM 会垫底（针不在最近），
   如果成立就把"两种 regime"从观察变成可预测的规律，本地就能做
3. 更大的 dense 规模点（倒 U 的右端在哪），或 MoE（要另租卡）

~~8B 规模点~~ → 2026-09-22 已做（租卡 4080S，四点受控 sweep）。

**可选（不急）**：Qwen3-14B 填 8B 之后的空白，判断倒 U 是继续跌还是回升。
约束：32G 卡只能跑 NF4（bf16 28GB + 激活会爆），而 0.6–8B 全是 bf16，
**跑 NF4 就得配一个 NF4 锚点**否则引入混淆；
autodl-tmp 50G 装不下 14B（28GB），要先删旧模型。
27B / 35B-A3B **不适合做这条线**——跨模型族 / MoE 稀疏激活，会把架构差异混进已经非单调的曲线里。
35B-A3B 属于 Airi 落地那条线（见 B），不是规模线。

**租卡准备已就绪**（2026-09-22）：`cloud_run.sh` 重写，参数钉死在 `scale3.sh` 上，
修掉了"没写 --lr 所以用默认 1e-3 坏点"和"没写 --no-lora"两个会烧钱的坑；
bf16 OOM 自动降级 NF4（代价已量化 1.2 点）；训完同场跑 hybrid。
下载用 `fetch_ms.sh <size>`（ModelScope + 分片校验）。
`hybrid.py` / `diagnose.py` 都已支持 `--load-4bit`。

**B. Airi 落地**：等 10 月中加内存，按 `docs/freetoken-integration.md` 验三个接口，
通过再租 96G 训 35B 记忆模块。**先验接口再训模型。**

**C. 工程余量**：offload KV 的 4bit 量化（896M→224M）、检索结果跨 token 缓存。
质量已基本榨干，剩下的都是延迟和内存。

## 环境备忘

- `source ~/ctfenv/bin/activate`
- 仓库 https://github.com/HoshiGT/frozen-kv-memory **已公开**（2026-09-21）——推送前检查 data/ 没进去
- `recur.py` 已废弃，位置约定有 bug，用 `train_recur.py`
- ★ 位置要累积（`accum=True`），不要重置——值 4 个点
- ★ 任何只保留部分 KV 的方案都必须带 attention sink，否则 −400%

## 第二篇（幻想中，2026-09-23）

三个候选方向写在 `docs/HANDOFF-paper2-prior-art.md`：
A 学出来的检索器（用 oracle 选择当监督，填"选择是瓶颈"那 22.5 点）、
B 按场景自适应检索、C Airi 落地（35B-A3B，KV 层长期记忆）。
已交给 Grok 查前人工作，结果回来后抽查 ID **和标题**。

## ★★ 论文 v2 修改清单（2026-09-23 重写，取代上一版）

### 起因：打分 bug（比"span 切碎"严重得多）

所有手算注意力的检索（`query_anchors` / speculative / H2O / SnapKV / `memory.py._score`）
都**没给 query 加 RoPE**，而 cache 里的 key 是转过的。和模型真实注意力比：
top-32 重合 **1/32**，修好后 32/32、误差 0。`tests/test_attention_match.py` 钉住。
修复：`memtok.rope_q()`，三处调用。另：SnapKV 观察窗口 v1 是单独跑的（没见过上下文），
改成从整段前向里取，和原论文 prefill 一致。

v1 的"span 切碎 → block=8 修复"是在**坏打分**上观察到的，修好后是否还成立要看新数据。
v1 还有两处与 bug 无关的事实错：
- §4.1 说 SnapKV 窗口"在这个任务上就是问题"——**错**，我们的 SnapKV 从没见过问题
  （观察窗口是 haystack 末尾 32 token）；needle 上 ours 用的才是问题
- 摘要 "5% when a drafted query does"——needle 上用的是问题不是 draft
- Limitations 第二条 "4B 需要 NF4 / 8B 放不下" 是租卡前的旧文，受控扫描全是 bf16（已改）
- `tab:needle` "512 anchors, oracle retrieval 98%" **标错**：oracle 实际是 needle 自身 16 token
  + sink 4 + 64 槽 = 84 条，不是 512；正文"同样 512 条预算 oracle 98% vs 注意力 5%"也要改
- 规模表 README 写"同机"不准：14B 是第二次租的 96G 卡上训的

### 受影响的结果（要重跑）

本地 0.6B（`rerun_v2.sh` → `out/v2/`）：
- `tab:needle` 可部署两行 + 按深度分析（"只在 0.3/0.9 找到"）
- `tab:baselines`（H2O/SnapKV/ours；StreamingLLM 不看注意力，不变）
- `tab:inversion` + Figure 2 + 整段机制解释
- `tab:twobytwo` speculative 一行（oracle 行按 token 匹配，不受影响）
- "与 oracle 只重合 8.5% 却拿到 3/4 价值"（价值弥散论）
- 随机槽 −45~−97%（固定的是 speculative 锚点）
- `tab:levers` anchor 数 512→1024（+87.8→+96.6，4096/8 跳，需另跑）

要租卡（受控扫描 = 同机 bf16 + data_clean，只评估不训练）：
- `tab:scale` 的 hybrid 行 + gain 行 + nats 行、Figure 1、"倒 U"结论
- 顺带：scale 表 "512 anchors alone" 用的是**稀有度锚点**，hybrid 用的是 speculative 锚点，
  "gain over better component" 混了两种锚点——v2 改用同锚点的 NO state 行

不受影响：§3 可复制性（纯抽象）、§6 LoRA 分母、方差表前四行、oracle 所有数字、
纯抽象的 scale 行、levers 前两行。

### 代码
- `--block` 默认值**先不改**，等新数据说话
- `topical.py` 同样有 bug，结论没进论文，已加注释不重跑
- README / README-usage.md 里 speculative 相关数字全部要加更正说明

### 进度（2026-09-23 傍晚）：**v2 已完成，等 Hoshi 发 Zenodo**

- 本地 0.6B 全部重跑（`out/v2/SUMMARY.md`），云端 4090 48G 重评规模五点（`out/cloud_v2/SCALE.md`）
- 论文 0 个 TODO，18 页，编译干净，PDF 扫描干净（语料 8-gram 0 命中、无凭据）
- 摘要 1881 字符（arXiv 上限 1920）
- Zenodo 填写内容：`paper/ZENODO-v2.md`（New version → 换 PDF → Version 2 → 描述 + 版本说明）
- 发布后：把 v2 DOI 填进 `paper/README.md` 和仓库 README 顶部
- 仓库改动**全部未 commit**，推之前再扫一遍（别带 data/、data_clean/）

### 发版
重编译 + 渲染检查 + 扫 PDF → Zenodo **新版本**，版本说明写清：打分 bug、影响范围、
哪些结论变了。v1 DOI 永久保留。

## 第二篇：方向 A 的空间测试（2026-09-23 晚，`out/length/SUMMARY.md`）

Grok 调研：`docs/PAPER2-PRIOR-ART.md`（抽查 12 条 ID+标题全对）。A 宽题目很挤，窄题目
（答案 oracle 监督 + 从卸载 KV 调回 + 槽）空着。

**空间 = 逐头未来注意力 oracle + 槽 − 逐头 H2O + 槽**：4096 17.8 / 8192 26.4 / 16384 34.4 点
（0.17 / 0.28 / 0.44 nats），随长度单调变大 → A 还活着，长上下文更值得。
★ 共用一组位置的 token oracle 不是逐头选择器的天花板（16k 塌到 73），别再用它量空间。

**下一步（决定 A 的生死）**：解码时每步按头精确 top-k（Quest 的上界，免训练、用真实当前 query）
能拿到多少？吃掉大部分空间 → 学选择器只剩效率意义；吃不掉 → A 成立。
B（按任务切检索几何）动机随 bug 消失，暂搁。C 的对象要换（35B-A3B 只有 10/40 层有 KV）。
