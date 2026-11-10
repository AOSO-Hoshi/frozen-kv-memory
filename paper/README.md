# 论文

**v2 已发布（2026-09-23）**：DOI [10.5281/zenodo.22913648](https://doi.org/10.5281/zenodo.22913648)。
修正 v1 的注意力打分 bug（手算注意力没给 query 加 RoPE）及其影响的全部结论，18 页。
改动清单在论文末尾 "Changes from version 1"，数据在 `../out/v2/`、`../out/cloud_v2/`。
v1 源码留档为 `main.v1.tex`。

**Concept DOI（所有版本，永远指向最新）**：[10.5281/zenodo.22897853](https://doi.org/10.5281/zenodo.22897853)
—— 主页 www.hoshichan.moe 的「论文」卡片挂的就是这个（2026-09-26 加上）。

**v1 已发布到 Zenodo**：DOI [10.5281/zenodo.22897854](https://doi.org/10.5281/zenodo.22897854)
（2026-09-22，CC BY 4.0）。**v1 有 bug，请引用 v2 或 concept DOI（"所有版本"）；下面的 BibTeX 用的是 concept DOI。** arXiv 仍缺 endorsement。

引用格式：

```bibtex
@misc{gao2026compression,
  title  = {What Context Compression Cannot Buy: Diagnosing the Recall Ceiling
            with a Frozen Backbone},
  author = {Gao, Xin},
  year   = {2026},
  doi    = {10.5281/zenodo.22897853},
  url    = {https://doi.org/10.5281/zenodo.22897853}
}
```


`main.tex` + `refs.bib`。**Version 2，2026-09-23，编译后 18 页**（v1 是 17 页）。

标题：*What Context Compression Cannot Buy: Diagnosing the Recall Ceiling with a
Frozen Backbone*

## 怎么编译

**本地已装好**（2026-09-22，`texlive-latex-recommended` +
`texlive-fonts-recommended` + `texlive-bibtex-extra`）：

```bash
cd ~/Claude/memzip/paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

当前产物（v2）：**18 页，0 error、0 undefined、0 overfull box**，0 个 `\TODO`。
`main.pdf` 有意提交进仓库，这样不装 TeX 也能读。

Overleaf 同样可用（只用了标准包，拖 `main.tex` + `refs.bib` 进去即可）。

### ★ 编译才抓得到的三个坑（自检脚本抓不到）

1. **18 处 `\citet`/`\citep` 但没加载 `natbib`** —— 用 `plainnat` 样式必须加
2. **microtype 的 font expansion 在默认 CM 位图字体上直接 fatal** ——
   加 `\usepackage{lmodern}` + `[T1]{fontenc}` 解决
3. **`refs.bib` 的 `note=` 字段会被排进参考文献** ——
   内部核验标注（"[V] checked against arXiv API"）差点印进论文，已移除

## 状态：全文有数据，`\TODO` / `\CHECK` 已清零

| 章节 | 状态 |
|---|---|
| Abstract | 完成 |
| 1 Introduction | 完成，三条贡献定位收敛 |
| 2 Setup | 完成 |
| 3 天花板由精确回忆构成（贡献 a） | **完成，3 seed** |
| 4 检索/压缩分解（贡献 b） | **完成，3 seed，2×2 已闭合** |
| 5 三个杠杆 | 完成，每行配置已核对并标注 |
| 6 两个测量陷阱 | 完成 |
| 7 Related work | 完成，六段 |
| 8 Limitations | 完成，五条 |

自检脚本（label/ref、cite/bib、括号、$、环境、表格列数）全过。

## 今天写草稿暴露出的三个证据问题（都已修正）

写论文的过程确实会自己告诉你哪里证据不够：

1. **加法分解表混了两组实验**。`+19.0`（完美选择的增益）来自 4096 上下文的
   单次实验，而 `31.4` / `44.0` 来自 8192 的 3-seed。补跑了 2×2 缺的那一格
   （oracle × 只压最近一段 = **+98.0%**），现在两个方向都能固定一个因子读另一个。
2. **`+231%` 是单次测量的高估**。3 seed 均值是 **+202%**，极差 11.3 点。
   而且这一组极差最大不是巧合——novel 组的 gap 只有 0.213 nats，
   小分母放大噪声。论文改成先报绝对 nats，recovery 当辅助。
3. **"0.6B → 1.7B → 4B 全部持平"超出证据**。混合架构只验了两个点，
   4B 的 ckpt 训练步数不同，不是受控对比。已从论文撤下并在 Limitations 里点名。

## 引用

29 条，**全部在 2026-09-22 用 arXiv API 核过 ID / 标题 / 第一作者**。

- Grok 调研给的 20 个 ID **全部真实存在，没有编造**
- 但有 3 个标题是论文内部的一句话而非真标题，已修正
  （`2412.17483` 不叫 "Lost by the Boundary"；
  `2510.20797` 真名 "No Mean Feat" 而非 "Simple Context Compression"；
  `2502.01941` 不叫 ShotKV）
- 作者列表还都是 `and others`，**投稿前要补全**

## 署名（已定，2026-09-22）

```
Xin Gao
hoshi@stu.xaut.edu.cn
```

**不写 affiliation** —— Hoshi 的决定。arXiv 投稿时 affiliation 可以留空，
独立研究者是合法身份，用学校邮箱但不标学校也没有问题。
提交时这三行要和 arXiv 元数据里填的一致。

## 投稿前还缺的实验（Limitations 里都已诚实写明）

- gap 大的 benchmark（LongBench / RULER）——现在全是 perplexity
- H2O / SnapKV baseline
- 8B 规模点（要租卡）
- arXiv cs.CL endorsement（**先写好再找人**；arXiv 免费无版面费）
