# 交接：补全 24 条参考文献的作者列表

给接手的人/模型。**这是纯机械核对活，但有一个绝不能犯的错：编造作者名。**

## 背景

一篇关于 KV cache 压缩的论文要投 arXiv，参考文献里 24 条的 author 字段目前是
`姓, 名 and others` 的形式（第一作者是对的，其余被省略）。渲染出来是
"Zhang et al."，技术上能过，但对被引的人不礼貌，也显得赶工。

需要把每条补成完整作者列表。

## 需要什么

对下面每个 arXiv ID，给出**该论文的完整作者列表**，按 arXiv 摘要页上的顺序。

| arXiv ID | 当前第一作者（已核对，可用来确认找对了论文） |
|---|---|
| 2306.14048 | Zhang, Zhenyu |
| 2404.14469 | Li, Yuhong |
| 2406.02069 | Cai, Zefan |
| 2307.06945 | Ge, Tao |
| 2408.03094 | Li, Zongqian |
| 2407.09252 | Rau, David |
| 2502.06139 | An, Sumin |
| 2503.16163 | Jie, Shibo |
| 2608.30252 | Yuan, Tong |
| 2312.03414 | Kim, Jang-Hyun |
| 2510.20797 | Feldman, Yair |
| 2510.20535 | Pilchen, Hippolyte |
| 2406.11238 | Hu, Yutong |
| 2412.17483 | Deng, Chenlong |
| 2609.19877 | Chen, Guanhua |
| 2606.10677 | Ji, Suozhao |
| 2406.10774 | Tang, Jiaming |
| 2402.04617 | Xiao, Chaojun |
| 2607.05399 | Agrawal, Nikita |
| 2502.01941 | Liu, Xiang |
| 2402.02750 | Liu, Zirui |
| 2110.07602 | Liu, Xiao |
| 2505.09388 | Yang, An |
| 2605.14037 | Szilvasy, Gergely |

## 输出格式

每条一行，BibTeX 的 author 字段格式（`姓, 名` 用 ` and ` 连接）：

```
2306.14048 | Zhang, Zhenyu and Sheng, Ying and Zhou, Tianyi and ...
2404.14469 | Li, Yuhong and Huang, Yue and ...
```

细节：

- **顺序必须和 arXiv 页面一致**，不要按字母重排
- 姓名里有连字符、重音符号（Müller）、单名的，照抄，不要转写
- 作者超过 15 人的，列前 15 个然后写 `and others`，并在该行末尾注明总人数
- 机构名、邮箱、脚注都不要

## ★ 最重要的一条

**查不到就写"查不到"，绝对不要根据论文主题推测作者。**

一个编造的作者名进了论文的参考文献，比"et al."严重得多——那是把不存在的
署名安在别人的工作上。如果某条：

- arXiv 页面打不开 → 写 `UNAVAILABLE`
- 找到的论文标题和第一作者对不上 → 写 `MISMATCH` 并附上你找到的标题
- 不确定某个名字的拼写 → 保留你看到的原样，并标注 `[uncertain]`

宁可交回 24 条里的 20 条，也不要 24 条里混 1 条假的。

## 怎么验证你找对了

上表的"当前第一作者"已经用 arXiv API 核对过。如果你查到的第一作者和它不一致，
说明找错论文了（或者我们的记录有误），请指出来而不是直接覆盖。

批量查询可以用：
```
http://export.arxiv.org/api/query?id_list=2306.14048,2404.14469,...
```
一次能查十几个，返回的 Atom XML 里每个 `<entry>` 的 `<author><name>` 就是要的东西，
顺序即原始顺序。这比逐个打开网页可靠，也不会看错。
