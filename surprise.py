#!/usr/bin/env python3
"""Write a segment into weights, then see if the next segment got easier.

The backbone stays frozen. A rank-4 LoRA on q_proj and v_proj is the only
plastic part, and it is reset for every window. That is the daydream with the
training wheels off: Titans trains the update rule, this is just Adam.

Surprise is the frozen model's own per-token loss. Each step keeps the top
quarter of that loss under the weights as they stand now, and steps only there.
The control does the same number of steps on a different window. If the control
helps this window's continuation as much as the real write, the weights learned
the dialect, not the segment.

Scores are absolute CE on the continuation, with no KV loaded. Recovery is
printed only as a translation. The slots row is the hippocampal reference.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from memtok import MemoryTokens, compress
from memory import kv_only
from train import load_chunks, seg_b_loss

HERE = Path(__file__).resolve().parent


class LoRAWrap(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = alpha / rank
        self.enabled = True
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if not self.enabled:
            return y
        update = (x.float() @ self.A.T) @ self.B.T
        return y + (update * self.scale).to(y.dtype)


def attach(model, rank: int, alpha: float) -> list[LoRAWrap]:
    wraps = []
    for layer in model.model.layers:
        attn = layer.self_attn
        for name in ("q_proj", "v_proj"):
            wrap = LoRAWrap(getattr(attn, name), rank, alpha)
            setattr(attn, name, wrap)
            wraps.append(wrap)
    return wraps


def set_on(wraps, flag: bool) -> None:
    for w in wraps:
        w.enabled = flag


def reset(wraps, seeds: list[torch.Tensor]) -> None:
    for w, A0 in zip(wraps, seeds):
        w.A.data.copy_(A0)
        w.B.data.zero_()


def per_token_ce(model, ids: torch.Tensor) -> torch.Tensor:
    logits = model(ids, use_cache=False).logits
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)).float(),
        ids[:, 1:].reshape(-1),
        reduction="none",
    ).view(ids.shape[0], -1)


def fit(model, wraps, ids, steps: int, lr: float, surprise: bool) -> float:
    params = [p for w in wraps for p in (w.A, w.B)]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    set_on(wraps, True)
    model.train()
    last = 0.0
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        mask = None
        if surprise:
            with torch.no_grad():
                ce = per_token_ce(model, ids)
                k = max(1, int(0.25 * ce.shape[-1]))
                keep = ce.topk(k, dim=-1).indices
                mask = torch.zeros_like(ce)
                mask.scatter_(-1, keep, 1.0)
        ce = per_token_ce(model, ids)
        loss = (ce * mask).sum() / mask.sum() if mask is not None else ce.mean()
        loss.backward()
        opt.step()
        last = float(loss.detach())
    model.eval()
    return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=int, default=512)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 1e-3])
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--ckpt", default=str(HERE / "ckpt" / "memtok_k32_deep_frozen.pt"))
    ap.add_argument("--model", default=str(HERE / "qwen3-0.6b"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = model.config
    wraps = attach(model, a.rank, a.alpha)
    for w in wraps:
        w.to(dev)
    seeds = [w.A.detach().clone() for w in wraps]
    n_par = sum(p.numel() for w in wraps for p in (w.A, w.B))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    sd = torch.load(a.ckpt, map_location=dev)
    sd = sd["mem"] if "mem" in sd else sd
    mem = MemoryTokens(sd["emb"].shape[0], cfg.hidden_size,
                       depth=sd["deep"].shape[0] if "deep" in sd else 0).to(dev)
    mem.load_state_dict(sd)

    data = load_chunks(tok, a.seg, a.n)
    print(f"段 {a.seg} | n={len(data)} | LoRA {n_par/1e6:.2f}M "
          f"rank={a.rank} | steps={a.steps} | 惊喜保留最高 25% 损失\n")

    keys = ["lower", "upper", "slots"]
    for lr in a.lrs:
        tag = f"{lr:.0e}"
        keys += [f"surp_{tag}", f"dense_{tag}", f"ctrl_{tag}", f"surpA_{tag}"]
    tot = dict.fromkeys(keys, 0.0)

    for i in range(len(data)):
        w = data[i:i + 1].to(dev)
        other = data[(i + 1) % len(data):(i + 1) % len(data) + 1].to(dev)
        A, B = w[:, :a.seg], w[:, a.seg:]
        Ao = other[:, :a.seg]

        set_on(wraps, False)
        model.eval()
        with torch.no_grad():
            tot["lower"] += seg_b_loss(model, cfg, B, None, a.seg).item()
            full = kv_only(model, A)
            tot["upper"] += seg_b_loss(model, cfg, B, full, a.seg).item()
            st = compress(model, mem, A, past=None, past_len=0, grad=False, k=a.k)
            tot["slots"] += seg_b_loss(model, cfg, B, st, a.seg + a.k).item()
            del full, st

        for lr in a.lrs:
            tag = f"{lr:.0e}"
            reset(wraps, seeds)
            fit(model, wraps, A, a.steps, lr, surprise=True)
            with torch.no_grad():
                tot[f"surpA_{tag}"] += seg_b_loss(model, cfg, A, None, 0).item()
                tot[f"surp_{tag}"] += seg_b_loss(model, cfg, B, None, a.seg).item()

            reset(wraps, seeds)
            fit(model, wraps, A, a.steps, lr, surprise=False)
            with torch.no_grad():
                tot[f"dense_{tag}"] += seg_b_loss(model, cfg, B, None, a.seg).item()

            reset(wraps, seeds)
            fit(model, wraps, Ao, a.steps, lr, surprise=True)
            with torch.no_grad():
                tot[f"ctrl_{tag}"] += seg_b_loss(model, cfg, B, None, a.seg).item()

        set_on(wraps, False)
        torch.cuda.empty_cache()
        print(f"  sample {i + 1}/{len(data)}", flush=True)

    n = len(data)
    for k in tot:
        tot[k] /= n
    gap = tot["lower"] - tot["upper"]
    print(f"\nlower={tot['lower']:.4f}  upper={tot['upper']:.4f}  gap={gap:.4f}")
    print("续写 B 的绝对 CE。权重条件都没有加载 KV。\n")
    print(f"  {'条件':<32} {'CE':>8} {'比 lower':>10}")
    print("  " + "-" * 54)
    rows = [("lower", "什么都不给"), ("upper", "A 的全部 KV"),
            ("slots", f"{a.k} 槽（海马体）")]
    for lr in a.lrs:
        tag = f"{lr:.0e}"
        rows += [
            (f"surp_{tag}", f"惊喜写入 lr={tag}"),
            (f"dense_{tag}", f"整段写入 lr={tag}"),
            (f"ctrl_{tag}", f"写别人的段 lr={tag}"),
        ]
    for key, label in rows:
        ce = tot[key]
        print(f"  {label:<32} {ce:8.4f} {tot['lower'] - ce:>+10.4f}")
    print("\n写入之后，A 自己的 CE（看有没有背下来）")
    for lr in a.lrs:
        tag = f"{lr:.0e}"
        print(f"  lr={tag}  A 的 CE={tot[f'surpA_{tag}']:.4f}")


if __name__ == "__main__":
    main()
