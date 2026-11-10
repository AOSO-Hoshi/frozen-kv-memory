#!/usr/bin/env python3
"""Does the learned state still add anything behind a strong selector?

v1 of the paper claimed that at a fixed budget a small learned component buys
far more than reallocating the same entries. That was measured with a weak
selector on both sides (last layer only, one position set shared by every
layer and head, and -- until the RoPE fix -- a query that was not rotated).
With H2O and SnapKV run per layer and per KV head, the training-free policies
alone come within a few points of ours. So the claim has to be re-asked with
the strong selector held fixed:

    selector(576 entries)          vs   selector(512 entries) + 64 learned slots

The difference between the two rows is the learned state's worth at an equal
budget, under a selector nobody would call a strawman.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from baseline_cmp import future_heads, gather_heads, h2o_heads, hidden_inputs, snapkv_heads
from hybrid import oracle_anchors
from memory import kv_only
from memtok import MemoryTokens, compress
from train import load_chunks, seg_b_loss

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=int, default=512)
    ap.add_argument("--hops", type=int, default=8)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--anchors", type=int, default=512)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--ckpt", default=str(HERE / "ckpt" / "memtok_k32_s0_deep_frozen.pt"))
    ap.add_argument("--model", default=str(HERE / "qwen3-0.6b"))
    args = ap.parse_args()
    budget = args.anchors + args.k

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to(dev).eval()
    cfg = model.config
    for p in model.parameters():
        p.requires_grad_(False)

    sd = torch.load(args.ckpt)
    sd = sd["mem"] if "mem" in sd else sd
    depth = sd["deep"].shape[0] if "deep" in sd else 0
    mem = MemoryTokens(sd["emb"].shape[0], cfg.hidden_size, depth=depth).to(dev).eval()
    mem.load_state_dict(sd)

    cut = args.seg * args.hops
    data = load_chunks(tok, cut + args.seg, args.n)[: args.n]

    rows = ["upper", "lower", "abstract",
            "h2o", "h2o+slots", "snapkv", "snapkv+slots", "oracle+slots",
            "future+slots"]
    tot = {r: 0.0 for r in rows}
    n_win = 0
    with torch.no_grad():
        for i in range(len(data)):
            w = data[i : i + 1].to(dev)
            ctx, tail = w[:, :cut], w[:, cut : cut + args.seg]
            full = kv_only(model, ctx)
            tot["upper"] += seg_b_loss(model, cfg, tail, full, cut).item()
            tot["lower"] += seg_b_loss(model, cfg, tail, None, 0).item()

            # the last-segment state, as everywhere else in the paper
            st = compress(model, mem, ctx[:, cut - args.seg:], past=None,
                          past_len=cut - args.seg, grad=False, k=args.k)
            tot["abstract"] += seg_b_loss(model, cfg, tail, st, cut + args.k).item()

            hs = hidden_inputs(model, ctx)
            for name, fn in (("h2o", h2o_heads), ("snapkv", snapkv_heads)):
                alone = gather_heads(full, fn(model, cfg, full, ctx, budget, hs=hs))
                tot[name] += seg_b_loss(model, cfg, tail, alone, cut).item()
                sel = gather_heads(full, fn(model, cfg, full, ctx, args.anchors, hs=hs))
                mix = [(torch.cat([a, s], 2), torch.cat([b, t], 2))
                       for (a, b), (s, t) in zip(sel, st)]
                tot[name + "+slots"] += seg_b_loss(model, cfg, tail, mix, cut + args.k).item()
            del hs
            # the ceiling on any selector at this budget: the 512 entries the
            # continuation actually reuses, with the same slots behind them.
            # How far the strong selector sits under it is the room left for a
            # better one -- the question a learned retriever would have to answer.
            oidx = oracle_anchors(ctx, tail, args.anchors, 4)
            osel = [(kk[:, :, oidx], vv[:, :, oidx]) for kk, vv in full]
            omix = [(torch.cat([a, s], 2), torch.cat([b, t], 2))
                    for (a, b), (s, t) in zip(osel, st)]
            tot["oracle+slots"] += seg_b_loss(model, cfg, tail, omix, cut + args.k).item()
            # the per-head version of the same ceiling: the continuation's own
            # attention, layer by layer and head by head
            fsel = gather_heads(full, future_heads(model, cfg, full, ctx, tail, args.anchors))
            fmix = [(torch.cat([a, s], 2), torch.cat([b, t], 2))
                    for (a, b), (s, t) in zip(fsel, st)]
            tot["future+slots"] += seg_b_loss(model, cfg, tail, fmix, cut + args.k).item()
            n_win += 1

    for r in tot:
        tot[r] /= n_win
    gap = tot["lower"] - tot["upper"]
    print(f"\n{n_win} windows | {cut} tokens | budget {budget} entries")
    print(f"upper {tot['upper']:.4f}  lower {tot['lower']:.4f}  gap {gap:.4f} nats\n")
    print(f"  {'condition':<34} {'CE':>8} {'recovery':>10}")
    print("  " + "-" * 54)
    for r, label in (("abstract", f"{args.k} slots only (last segment)"),
                     ("h2o", f"H2O per head, {budget} entries"),
                     ("h2o+slots", f"H2O per head, {args.anchors} + {args.k} slots"),
                     ("snapkv", f"SnapKV per head, {budget} entries"),
                     ("snapkv+slots", f"SnapKV per head, {args.anchors} + {args.k} slots"),
                     ("oracle+slots", f"oracle, {args.anchors} + {args.k} slots"),
                     ("future+slots", f"future attn per head, {args.anchors} + {args.k}")):
        print(f"  {label:<34} {tot[r]:8.4f} {(tot['lower'] - tot[r]) / gap:+9.1%}")


if __name__ == "__main__":
    main()
