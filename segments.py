#!/usr/bin/env python3
"""Do segment understandings work when they are never put in one softmax?

merge.py mixed six states either by concatenation or by averaging. Both ask
the final 512 tokens to use every segment at once. This script asks a different
question: each block is already its own state. Read it alone.

  local   the 512 tokens right after that block. Upper is that block's own KV,
          rotated at the block's real positions.
  remote  the same state, scored on the final tail, still alone.
  ensemble  six separate forwards, averaged in probability, not in KV.
  oracle  per sample, the single state with the lowest CE on the final tail.

Local recovery says whether a segment understanding is intact. Remote recovery
says whether it still matters at the end. The ensemble says whether keeping
them out of one softmax lets them add.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from memtok import MemoryTokens, compress
from memory import kv_only
from train import load_chunks, seg_b_loss

HERE = Path(__file__).resolve().parent


def logits_of(model, cfg, ids_b, past_pairs, seg: int):
    B, S = ids_b.shape
    pos = torch.arange(seg, seg + S, device=ids_b.device).unsqueeze(0).expand(B, -1)
    kw = {}
    if past_pairs is not None:
        from train import cache_from
        plen = past_pairs[0][0].shape[2]
        kw["past_key_values"] = cache_from(past_pairs, cfg)
        kw["attention_mask"] = torch.ones(B, plen + S, device=ids_b.device, dtype=torch.long)
        kw["cache_position"] = torch.arange(plen, plen + S, device=ids_b.device)
    return model(ids_b, position_ids=pos, use_cache=False, **kw).logits


def ce_of(logits, ids_b) -> float:
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)).float(),
        ids_b[:, 1:].reshape(-1),
    ).item()


def kv_at(model, ids, pos0: int):
    """KV rotated at absolute positions pos0 .. pos0+len-1, not at 0."""
    B, S = ids.shape
    pos = torch.arange(pos0, pos0 + S, device=ids.device).unsqueeze(0).expand(B, -1)
    base = getattr(model, "model", model)
    out = base(input_ids=ids, position_ids=pos, use_cache=True)
    return [(layer.keys, layer.values) for layer in out.past_key_values.layers]


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
    sd = torch.load(a.ckpt, map_location=dev)
    sd = sd["mem"] if "mem" in sd else sd
    mem = MemoryTokens(sd["emb"].shape[0], cfg.hidden_size,
                       depth=sd["deep"].shape[0] if "deep" in sd else 0).to(dev)
    mem.load_state_dict(sd)

    cut = a.seg * a.hops
    blk = (cut - a.seg) // a.blocks
    if blk < a.seg or blk % a.seg != 0:
        raise SystemExit(f"每块 {blk} token，需要是 {a.seg} 的正整数倍")
    data = load_chunks(tok, cut + a.seg, a.n)
    print(f"上下文 {cut} | {a.blocks} 块 × {blk} | 每块自己的下一段 {a.seg} | n={len(data)}\n")

    def roll(w, start, end):
        st = None
        pos0 = start
        for s0 in range(start, end, a.seg):
            s1 = min(s0 + a.seg, end)
            st = compress(model, mem, w[:, s0:s1], past=st, past_len=s0,
                          grad=False, k=a.k)
            pos0 = s0 + (s1 - s0)
        return st, pos0

    nB = a.blocks
    loc = {k: [0.0] * nB for k in ("upper", "lower", "state")}
    rem = {k: [0.0] * nB for k in ("state",)}
    fin = dict.fromkeys(["upper", "lower", "recent", "concat", "ensemble",
                         "ensemble_old", "oracle", "oracle_old"], 0.0)

    for i in range(len(data)):
        w = data[i:i + 1].to(dev)
        tail = w[:, cut:cut + a.seg]
        full = kv_only(model, w[:, :cut])
        fin["upper"] += seg_b_loss(model, cfg, tail, full, cut).item()
        fin["lower"] += seg_b_loss(model, cfg, tail, None, cut).item()

        pieces = []
        for b in range(nB):
            start, end = b * blk, (b + 1) * blk
            st, pos0 = roll(w, start, end)
            local = w[:, end:end + a.seg]
            upper = kv_at(model, w[:, start:end], start)
            loc["upper"][b] += seg_b_loss(model, cfg, local, upper, end).item()
            loc["lower"][b] += seg_b_loss(model, cfg, local, None, end).item()
            loc["state"][b] += seg_b_loss(model, cfg, local, st, pos0 + a.k).item()
            rem["state"][b] += seg_b_loss(model, cfg, tail, st, cut + a.k).item()
            pieces.append(st)
            del upper

        recent, recent_pos = roll(w, cut - a.seg, cut)
        fin["recent"] += seg_b_loss(
            model, cfg, tail, recent, recent_pos + a.k).item()

        from merge import cat_states
        fin["concat"] += seg_b_loss(
            model, cfg, tail, cat_states(pieces + [recent]), cut + a.k).item()

        # separate forwards: probabilities averaged, KV never shares a softmax
        states = pieces + [recent]
        acc = None
        ces = []
        for st in states:
            lg = logits_of(model, cfg, tail, st, cut + a.k)
            ces.append(ce_of(lg, tail))
            prob = lg[:, :-1].float().softmax(-1)
            acc = prob if acc is None else acc + prob
            del lg, prob
        mix = (acc / len(states)).clamp_min(1e-20).log()
        fin["ensemble"] += (-mix.gather(-1, tail[:, 1:, None])).mean().item()
        lg_r = logits_of(model, cfg, tail, recent, cut + a.k)
        acc_old = acc - lg_r[:, :-1].float().softmax(-1)
        mix_old = (acc_old / nB).clamp_min(1e-20).log()
        fin["ensemble_old"] += (-mix_old.gather(-1, tail[:, 1:, None])).mean().item()
        del lg_r, acc, acc_old

        fin["oracle"] += min(ces)
        fin["oracle_old"] += min(ces[:-1])
        del full
        torch.cuda.empty_cache()
        print(f"  sample {i + 1}/{len(data)}", flush=True)

    n = len(data)
    for key in loc:
        loc[key] = [v / n for v in loc[key]]
    rem["state"] = [v / n for v in rem["state"]]
    for key in fin:
        fin[key] /= n
    gap = fin["lower"] - fin["upper"]

    print(f"\n每段自己的下一段（单独读，upper 是这一段自己的 KV）")
    print(f"  {'段':<6} {'upper':>8} {'lower':>8} {'state':>8} {'recovery':>10}")
    print("  " + "-" * 46)
    for b in range(nB):
        g = loc["lower"][b] - loc["upper"][b]
        rec = (loc["lower"][b] - loc["state"][b]) / g if g else float("nan")
        print(f"  {b:<6} {loc['upper'][b]:8.4f} {loc['lower'][b]:8.4f} "
              f"{loc['state'][b]:8.4f} {rec:>+9.1%}")

    print(f"\n同一份状态，去预测最后一段（仍然单独读）")
    print(f"  最后一段 upper={fin['upper']:.4f}  lower={fin['lower']:.4f}  gap={gap:.4f}")
    print(f"  {'段':<6} {'CE':>8} {'recovery':>10}")
    for b in range(nB):
        print(f"  {b:<6} {rem['state'][b]:8.4f} {(fin['lower'] - rem['state'][b]) / gap:>+9.1%}")
    print(f"  {'最近':<6} {fin['recent']:8.4f} {(fin['lower'] - fin['recent']) / gap:>+9.1%}")

    print(f"\n最后一段上，几种读法")
    print(f"  {'条件':<28} {'CE':>8} {'recovery':>10}")
    print("  " + "-" * 50)
    for key, label in (
            ("recent", "只读最近一段"),
            ("concat", "六份放进同一次注意力"),
            ("ensemble_old", "五段各自算，概率平均"),
            ("ensemble", "六份各自算，概率平均"),
            ("oracle_old", "每条样本选最好的旧段"),
            ("oracle", "每条样本选最好的一份"),
    ):
        ce = fin[key]
        print(f"  {label:<28} {ce:8.4f} {(fin['lower'] - ce) / gap:>+9.1%}")


if __name__ == "__main__":
    main()
