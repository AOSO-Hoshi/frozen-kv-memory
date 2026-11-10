#!/usr/bin/env python3
"""Fuse several abstract states into one, instead of concatenating them.

Concatenation puts every state's pseudo-KV into one softmax. Similar keys then
mix each other's values, and recovery falls (topical.py: five blocks +56.6%,
the latest segment alone +70.7%). A fused state keeps the same k slots as a
single memory, so nothing extra enters that softmax.

Slots line up across states: slot i is always the KV of the same learned memory
token. Keys do not. Each state was rotated at the position it was written, and
averaging rotated keys blends two phases into a vector that matches neither
(compress.py, the v0 failure). Legal mixing un-rotates to phase 0, mixes, then
re-rotates the one result at the position the reader expects.

Conditions, all scored on the same tail, one sample at a time:

  recent          last segment only                         k slots
  concat          every block plus recent, side by side     (blocks+1)*k
  mean_rot        average K and V in the rotated space      k
  mean_align      un-rotate, average, re-rotate             k
  mean_rescale    mean_align, then put key norms back       k
  gather          keep recent's key, mix values by cosine   k
  rewrite         memory tokens read the pile and write one k
  fold            same write, two states at a time          k
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from memtok import MemoryTokens, compress, deep_prefix
from memory import kv_only
from train import cache_from, load_chunks, seg_b_loss

HERE = Path(__file__).resolve().parent


def _rotary(model):
    m = model
    for _ in range(4):
        if hasattr(m, "rotary_emb"):
            return m.rotary_emb
        m = getattr(m, "model", None) or getattr(m, "base_model", None)
    raise AttributeError("no rotary_emb")


def rope_factors(model, pos: torch.Tensor):
    """pos [n] -> cos, sin [n, head_dim], the same tables the model applies."""
    from transformers.models.qwen3.modeling_qwen3 import rotate_half
    dummy = torch.zeros(1, device=pos.device, dtype=torch.float32)
    cos, sin = _rotary(model)(dummy, pos.view(1, -1))
    return cos[0].float(), sin[0].float(), rotate_half


def unrotate(model, K: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """K [H, k, D] rotated at `pos` [k] -> phase-0 keys."""
    cos, sin, rotate_half = rope_factors(model, pos)
    Kf = K.float()
    return Kf * cos - rotate_half(Kf) * sin


def rotate(model, K: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    cos, sin, rotate_half = rope_factors(model, pos)
    Kf = K.float()
    return Kf * cos + rotate_half(Kf) * sin


def check_roundtrip(model, k: int = 8) -> float:
    """unrotate(rotate(x)) must be x. RoPE is orthogonal; a bug here voids the mix."""
    H, D = 2, model.config.head_dim
    pos = torch.arange(100, 100 + k, device=next(model.parameters()).device)
    x = torch.randn(H, k, D, device=pos.device)
    err = (unrotate(model, rotate(model, x, pos), pos) - x).abs().max().item()
    return err


def cat_states(states):
    return [tuple(torch.cat([s[l][j] for s in states], dim=2) for j in range(2))
            for l in range(len(states[0]))]


def slot_pos(pos0: int, k: int, device) -> torch.Tensor:
    return torch.arange(pos0, pos0 + k, device=device)


def stack_layer(states, li: int):
    K = torch.stack([s[li][0][0].float() for s in states], 0)  # [M, H, k, D]
    V = torch.stack([s[li][1][0].float() for s in states], 0)
    return K, V


def merge_mean(model, states, positions, write_pos: int, align: bool, rescale: bool):
    """One k-slot state. `align` mixes keys at phase 0; otherwise in rotated space."""
    dt = states[0][0][0].dtype
    dev = states[0][0][0].device
    k = states[0][0][0].shape[2]
    out = []
    for li in range(len(states[0])):
        K, V = stack_layer(states, li)
        if align:
            Ku = torch.stack([
                unrotate(model, K[m], slot_pos(positions[m], k, dev))
                for m in range(K.shape[0])
            ], 0)
            Km = Ku.mean(0)
            if rescale:
                n_in = Ku.norm(dim=-1).mean(0).clamp_min(1e-6)
                n_out = Km.norm(dim=-1).clamp_min(1e-6)
                Km = Km / n_out.unsqueeze(-1) * n_in.unsqueeze(-1)
            Km = rotate(model, Km, slot_pos(write_pos, k, dev))
        else:
            Km = K.mean(0)
        Vm = V.mean(0)
        out.append((Km[None].to(dt), Vm[None].to(dt)))
    return out


def merge_gather(model, states, positions, write_pos: int):
    """Keep the last state's key. Mix values by per-slot cosine in phase 0.

    Returns the state and the mean weight that landed on the last memory, so a
    collapse back to 'recent only' is visible.
    """
    dt = states[0][0][0].dtype
    dev = states[0][0][0].device
    k = states[0][0][0].shape[2]
    out = []
    weight_on_last = []
    for li in range(len(states[0])):
        K, V = stack_layer(states, li)
        Ku = torch.stack([
            unrotate(model, K[m], slot_pos(positions[m], k, dev))
            for m in range(K.shape[0])
        ], 0)
        q = Ku[-1]
        qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        kn = Ku / Ku.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        sim = (kn * qn).sum(-1)                     # [M, H, k]
        w = sim.softmax(dim=0)
        Vm = (w.unsqueeze(-1) * V).sum(0)
        Kr = rotate(model, q, slot_pos(write_pos, k, dev))
        out.append((Kr[None].to(dt), Vm[None].to(dt)))
        weight_on_last.append(w[-1].mean().item())
    return out, sum(weight_on_last) / len(weight_on_last)


def rewrite(model, mem, cfg, states, write_pos: int, k: int):
    """Memory tokens attend to `states` and their own KV becomes the new state.

    No new text. This is the compressor's write, applied to a pile of memories
    it was not trained on.
    """
    dev = states[0][0][0].device
    dt = states[0][0][0].dtype
    past = cat_states(states)
    x = mem.emb[:k].unsqueeze(0).to(dt)
    pos = torch.arange(write_pos, write_pos + k, device=dev)
    plen = past[0][0].shape[2]
    with deep_prefix(model, mem, k):
        out = model(
            inputs_embeds=x,
            position_ids=pos.view(1, -1),
            use_cache=True,
            past_key_values=cache_from(past, cfg),
            attention_mask=torch.ones(1, plen + k, device=dev, dtype=torch.long),
            cache_position=torch.arange(plen, plen + k, device=dev),
        )
    return [(layer.keys[:, :, -k:], layer.values[:, :, -k:])
            for layer in out.past_key_values.layers]


def fold(model, mem, cfg, states, write_pos: int, k: int):
    """Integrate left to right: each step writes one state from two."""
    running = states[0]
    for nxt in states[1:]:
        running = rewrite(model, mem, cfg, [running, nxt], write_pos, k)
    return running


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=int, default=512)
    ap.add_argument("--hops", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--ckpt", default=str(HERE / "ckpt" / "memtok_k32_deep_frozen.pt"))
    ap.add_argument("--model", default=str(HERE / "qwen3-0.6b"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda"
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16).to(dev).eval()
    cfg = model.config
    err = check_roundtrip(model)
    print(f"RoPE 往返误差 {err:.2e}")
    if err > 1e-3:
        raise SystemExit("RoPE 逆变换对不上，合成结果不能用")

    sd = torch.load(a.ckpt, map_location=dev)
    sd = sd["mem"] if "mem" in sd else sd
    mem = MemoryTokens(sd["emb"].shape[0], cfg.hidden_size,
                       depth=sd["deep"].shape[0] if "deep" in sd else 0).to(dev)
    mem.load_state_dict(sd)

    cut = a.seg * a.hops
    past_len_tok = cut - a.seg
    blk = past_len_tok // a.blocks
    if blk < a.seg or blk % a.seg != 0:
        raise SystemExit(f"每块 {blk} token，需要是 {a.seg} 的正整数倍")
    data = load_chunks(tok, cut + a.seg, a.n)
    print(f"上下文 {cut} | 远期 {past_len_tok} 切成 {a.blocks} 块（每块 {blk}）")
    print(f"合成目标是 {a.k} 槽，拼接则是 {(a.blocks + 1) * a.k} 槽 | n={len(data)}\n")

    def roll(w, start, end):
        st = None
        pos0 = start
        for s0 in range(start, end, a.seg):
            s1 = min(s0 + a.seg, end)
            st = compress(model, mem, w[:, s0:s1], past=st, past_len=s0,
                          grad=False, k=a.k)
            pos0 = s0 + (s1 - s0)
        return st, pos0

    names = ["upper", "lower", "recent", "concat",
             "mean_rot", "mean_align", "mean_rescale", "gather",
             "rewrite", "fold"]
    tot = dict.fromkeys(names, 0.0)
    gather_w = 0.0
    for i in range(len(data)):
        w = data[i:i + 1].to(dev)
        tail = w[:, cut:cut + a.seg]
        full = kv_only(model, w[:, :cut])
        tot["upper"] += seg_b_loss(model, cfg, tail, full, cut).item()
        tot["lower"] += seg_b_loss(model, cfg, tail, None, cut).item()

        recent, recent_pos = roll(w, cut - a.seg, cut)
        blocks = [roll(w, b * blk, (b + 1) * blk) for b in range(a.blocks)]
        states = [st for st, _ in blocks] + [recent]
        positions = [p for _, p in blocks] + [recent_pos]
        write_pos = recent_pos
        read_at = write_pos + a.k

        tot["recent"] += seg_b_loss(model, cfg, tail, recent, read_at).item()
        tot["concat"] += seg_b_loss(
            model, cfg, tail, cat_states(states), read_at).item()
        tot["mean_rot"] += seg_b_loss(
            model, cfg, tail,
            merge_mean(model, states, positions, write_pos, align=False, rescale=False),
            read_at).item()
        tot["mean_align"] += seg_b_loss(
            model, cfg, tail,
            merge_mean(model, states, positions, write_pos, align=True, rescale=False),
            read_at).item()
        tot["mean_rescale"] += seg_b_loss(
            model, cfg, tail,
            merge_mean(model, states, positions, write_pos, align=True, rescale=True),
            read_at).item()
        gathered, w_last = merge_gather(model, states, positions, write_pos)
        gather_w += w_last
        tot["gather"] += seg_b_loss(model, cfg, tail, gathered, read_at).item()
        tot["rewrite"] += seg_b_loss(
            model, cfg, tail, rewrite(model, mem, cfg, states, write_pos, a.k),
            read_at).item()
        tot["fold"] += seg_b_loss(
            model, cfg, tail, fold(model, mem, cfg, states, write_pos, a.k),
            read_at).item()
        del full
        torch.cuda.empty_cache()
        print(f"  sample {i + 1}/{len(data)}", flush=True)

    n = len(data)
    for key in tot:
        tot[key] /= n
    gap = tot["lower"] - tot["upper"]
    print(f"\nupper CE={tot['upper']:.4f}  lower CE={tot['lower']:.4f}  gap={gap:.4f}")
    print(f"gather 落在最近一份记忆上的平均权重：{gather_w / n:.2f}\n")
    rows = (
        ("recent", "只有最近一段", 1),
        ("concat", f"{a.blocks} 块 + 最近，拼接", a.blocks + 1),
        ("mean_rot", "旋转空间里直接平均", 1),
        ("mean_align", "对齐相位后平均", 1),
        ("mean_rescale", "对齐平均，并补回 key 范数", 1),
        ("gather", "留下最近的 key，按余弦混 value", 1),
        ("rewrite", "记忆槽读完全部，写回一份", 1),
        ("fold", "两两写回，从左折到右", 1),
    )
    print(f"  {'条件':<28} {'槽':>6} {'CE':>8} {'recovery':>10}")
    print("  " + "-" * 56)
    for key, label, nslot in rows:
        ce = tot[key]
        print(f"  {label:<28} {a.k * nslot:>6} {ce:8.4f} {(tot['lower'] - ce) / gap:>+9.1%}")


if __name__ == "__main__":
    main()
