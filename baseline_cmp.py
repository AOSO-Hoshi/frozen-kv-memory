#!/usr/bin/env python3
"""Training-free KV selection baselines at the budget this paper spends.

The paper's hybrid keeps 512 verbatim entries plus 64 learned slots. The obvious
question is whether the learned part earns its place: three well-known
training-free policies choose which KV entries to keep, and at 576 entries they
are spending exactly the same memory.

Implemented from the mechanism each paper describes, not from a wrapper:

  StreamingLLM  a few initial "sink" positions plus the most recent window.
  H2O           recent window plus "heavy hitters" -- the positions that
                received the most accumulated attention over all queries.
  SnapKV        attention from a short observation window at the end of the
                prompt, max-pooled over neighbouring positions so that spans
                rather than isolated tokens are retrieved.

Two deliberate choices about fairness:

  * No sink is forced on H2O or SnapKV. Neither paper prescribes one, and both
    mechanisms tend to select the opening positions anyway (that is what the
    attention-sink phenomenon is). We report how many sink positions each policy
    actually chose, so a collapse from losing the sink is visible rather than
    assumed.
  * Attention scores are computed the same way for every policy, including ours
    -- the query projection of the last block against the cached keys. sdpa does
    not expose attention weights, and re-deriving them differently per baseline
    would make the comparison about implementations rather than mechanisms.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hybrid import query_anchors, speculate, oracle_anchors, rope_q
from memory import kv_only
from memtok import MemoryTokens, compress, _layers
from train import load_chunks, seg_b_loss

HERE = Path(__file__).resolve().parent


def _queries(model, cfg, ids: torch.Tensor, offset: int = 0):
    """Query vectors of the final block for every position in `ids`."""
    with torch.no_grad():
        out = model(ids, position_ids=torch.arange(
            offset, offset + ids.shape[1], device=ids.device).unsqueeze(0),
            output_hidden_states=True)
        h = out.hidden_states[-2][0]
    layer = _layers(model)[-1]
    attn = layer.self_attn
    q = attn.q_proj(layer.input_layernorm(h))
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    q = q.view(-1, cfg.num_attention_heads, hd)
    if hasattr(attn, "q_norm"):
        q = attn.q_norm(q)
    q = rope_q(model, q, torch.arange(offset, offset + ids.shape[1], device=ids.device))
    return q, hd


def _keys(cfg, full):
    k = full[-1][0][0]                                  # [kv_heads, ctx, hd]
    rep = cfg.num_attention_heads // k.shape[0]
    return k.repeat_interleave(rep, dim=0)


def accum_attention(model, cfg, full, ctx, chunk: int = 256):
    """Attention received by each key position, summed over all query positions.

    This is H2O's heavy-hitter statistic. Computed in query chunks because the
    full matrix is n^2 per head; causal masking is applied so a position is only
    credited by queries that could actually see it.
    """
    n = ctx.shape[1]
    q, hd = _queries(model, cfg, ctx)
    k = _keys(cfg, full)
    score = torch.zeros(n, device=ctx.device, dtype=torch.float32)
    pos = torch.arange(n, device=ctx.device)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        att = torch.einsum("thd,hcd->thc", q[a:b].float(), k.float()) / hd ** 0.5
        att = att.masked_fill(pos[None, None, :] > pos[a:b, None, None], float("-inf"))
        score += att.softmax(-1).sum(dim=(0, 1))
    return score


def streaming_idx(n: int, budget: int, sink: int, dev):
    """StreamingLLM: the opening sink plus the most recent window."""
    recent = max(0, budget - sink)
    return torch.cat([torch.arange(min(sink, n), device=dev),
                      torch.arange(max(0, n - recent), n, device=dev)]).unique()


def h2o_idx(model, cfg, full, ctx, budget: int, recent_frac: float = 0.5):
    """H2O: half the budget on the recent window, half on heavy hitters."""
    n = ctx.shape[1]
    dev = ctx.device
    n_recent = int(budget * recent_frac)
    recent = torch.arange(max(0, n - n_recent), n, device=dev)
    score = accum_attention(model, cfg, full, ctx)
    score[recent] = float("-inf")                       # already kept
    heavy = score.topk(min(budget - len(recent), n - len(recent))).indices
    return torch.cat([recent, heavy]).unique().sort().values


def snapkv_idx(model, cfg, full, ctx, budget: int, obs: int = 32, pool: int = 7):
    """SnapKV: observation window at the end of the prompt, max-pooled."""
    n = ctx.shape[1]
    dev = ctx.device
    obs = min(obs, n)
    # The window's queries are taken from a pass over the whole prompt, as in
    # SnapKV's prefill. v1 ran the window on its own, which handed SnapKV
    # queries that had never seen the context they were scoring.
    q, hd = _queries(model, cfg, ctx)
    q = q[n - obs:]
    k = _keys(cfg, full)
    att = torch.einsum("thd,hcd->thc", q.float(), k.float()) / hd ** 0.5
    pos = torch.arange(n, device=dev)
    qpos = torch.arange(n - obs, n, device=dev)
    att = att.masked_fill(pos[None, None, :] > qpos[:, None, None], float("-inf"))
    score = att.softmax(-1).sum(dim=(0, 1))
    # the observation window itself is always kept
    score[n - obs:] = float("-inf")
    # max-pool so that a wanted position drags its neighbours in
    sm = torch.nn.functional.max_pool1d(
        score[None, None, :], kernel_size=pool, stride=1, padding=pool // 2)[0, 0]
    keep = budget - obs
    top = sm.topk(min(max(0, keep), n - obs)).indices
    return torch.cat([top, torch.arange(n - obs, n, device=dev)]).unique().sort().values



# ---------------------------------------------------------------------------
# Per-layer, per-head versions (paper v2). The functions above pick one set of
# positions from the last layer and apply it to every layer and head, which is
# a simplification of both methods: SnapKV and H2O each decide per layer and per
# KV head, and at the same entry budget that freedom is worth a great deal on a
# task where one layer's heads find the needle and another's do not. These are
# the ones the v2 paper reports.
# ---------------------------------------------------------------------------

def hidden_inputs(model, ctx):
    """Input to every decoder layer for a full pass over `ctx`."""
    with torch.no_grad():
        # logits_to_keep=1: the logits are never used here, and at 16k tokens
        # the full [n, vocab] tensor alone is 5 GB
        out = model(ctx, output_hidden_states=True, logits_to_keep=1)
    return out.hidden_states                      # [emb, after L0, ...]; [i] feeds layer i


def layer_queries(model, cfg, h_in, i: int, pos: torch.Tensor):
    """Rotated queries of layer i from that layer's input hidden states [t, d]."""
    layer = _layers(model)[i]
    attn = layer.self_attn
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    q = attn.q_proj(layer.input_layernorm(h_in)).view(-1, cfg.num_attention_heads, hd)
    if hasattr(attn, "q_norm"):
        q = attn.q_norm(q)
    return rope_q(model, q, pos), hd


def _attn_rows(q, k, qpos, hd):
    """Causal softmax attention [H, t, n] of rotated queries over one layer's keys."""
    rep = q.shape[1] // k.shape[0]
    kk = k.repeat_interleave(rep, dim=0).float()
    a = torch.einsum("thd,hcd->htc", q.float(), kk) / hd ** 0.5
    n = k.shape[1]
    a = a.masked_fill(torch.arange(n, device=q.device)[None, None, :] > qpos[None, :, None],
                      float("-inf"))
    return a.softmax(-1)


def _to_kv_heads(score, kvh):
    """[H, n] summed within each GQA group -> [kv_heads, n]."""
    return score.view(kvh, -1, score.shape[-1]).sum(1)


def gather_heads(full, idx):
    """Cache holding, for each layer and KV head, only its own chosen positions."""
    out = []
    for (k, v), ix in zip(full, idx):
        g = ix[None, :, :, None].expand(k.shape[0], -1, -1, k.shape[-1])
        out.append((k.gather(2, g), v.gather(2, g)))
    return out


def snapkv_heads(model, cfg, full, ctx, budget: int, obs: int = 32, pool: int = 7,
                 hs=None):
    """SnapKV per layer and KV head: window votes, max-pooled, top-k plus window."""
    n = ctx.shape[1]
    dev = ctx.device
    obs = min(obs, n)
    hs = hidden_inputs(model, ctx) if hs is None else hs
    qpos = torch.arange(n - obs, n, device=dev)
    win = qpos.expand(full[0][0].shape[1], -1)
    idx = []
    for i, (k, _) in enumerate(full):
        q, hd = layer_queries(model, cfg, hs[i][0][n - obs:], i, qpos)
        a = _attn_rows(q, k[0], qpos, hd)
        score = _to_kv_heads(a.sum(1), k.shape[1])                     # [kvh, n]
        score = torch.nn.functional.max_pool1d(
            score[:, None, :], kernel_size=pool, stride=1, padding=pool // 2)[:, 0]
        score[:, n - obs:] = float("-inf")                              # kept anyway
        top = score.topk(min(max(0, budget - obs), n - obs), dim=-1).indices
        idx.append(torch.cat([top, win], -1).sort(-1).values)
    return idx


def snapkv_question(model, cfg, ctx, question, budget: int, pool: int = 7):
    """SnapKV as it is normally run: the question closes the prompt, so the
    observation window *is* the question. All `budget` entries go to context
    positions; the question itself is re-read at answer time anyway.
    """
    n = ctx.shape[1]
    seq = torch.cat([ctx, question], 1)
    hs = hidden_inputs(model, seq)
    kv = kv_only(model, seq)
    qpos = torch.arange(n, seq.shape[1], device=ctx.device)
    idx = []
    for i, (k, _) in enumerate(kv):
        q, hd = layer_queries(model, cfg, hs[i][0][n:], i, qpos)
        a = _attn_rows(q, k[0], qpos, hd)[:, :, :n]          # votes on context only
        score = _to_kv_heads(a.sum(1), k.shape[1])
        score = torch.nn.functional.max_pool1d(
            score[:, None, :], kernel_size=pool, stride=1, padding=pool // 2)[:, 0]
        idx.append(score.topk(min(budget, n), dim=-1).indices.sort(-1).values)
    return idx


def future_heads(model, cfg, full, ctx, tail, budget: int, chunk: int = 128):
    """Answer-aware ceiling for per-head selection: each layer and KV head keeps
    the context positions the true continuation attends to most.

    This is the "future attention" that learned evictors (KVpop, ForesightKV)
    are trained toward, so the distance from a training-free selector up to it
    is the most a learned per-head selector could win at this budget.
    """
    n = ctx.shape[1]
    seq = torch.cat([ctx, tail], 1)
    hs = hidden_inputs(model, seq)
    qpos = torch.arange(n, seq.shape[1], device=ctx.device)
    idx = []
    for i, (k, _) in enumerate(full):                  # context keys only
        score = torch.zeros(k.shape[1], n, device=ctx.device)
        for a0 in range(0, len(qpos), chunk):          # [H, t, n] is 0.5 GB at 16k
            qp = qpos[a0:a0 + chunk]
            q, hd = layer_queries(model, cfg, hs[i][0][n + a0:n + a0 + chunk], i, qp)
            score += _to_kv_heads(_attn_rows(q, k[0], qp, hd).sum(1), k.shape[1])
        idx.append(score.topk(min(budget, n), dim=-1).indices.sort(-1).values)
    return idx


def h2o_heads(model, cfg, full, ctx, budget: int, recent_frac: float = 0.5,
              chunk: int = 256, hs=None):
    """H2O per layer and KV head: recent window plus that head's heavy hitters."""
    n = ctx.shape[1]
    dev = ctx.device
    hs = hidden_inputs(model, ctx) if hs is None else hs
    n_recent = int(budget * recent_frac)
    kvh = full[0][0].shape[1]
    recent = torch.arange(max(0, n - n_recent), n, device=dev).expand(kvh, -1)
    idx = []
    for i, (k, _) in enumerate(full):
        score = torch.zeros(kvh, n, device=dev)
        for a0 in range(0, n, chunk):
            b0 = min(a0 + chunk, n)
            qpos = torch.arange(a0, b0, device=dev)
            q, hd = layer_queries(model, cfg, hs[i][0][a0:b0], i, qpos)
            score += _to_kv_heads(_attn_rows(q, k[0], qpos, hd).sum(1), kvh)
        score[:, n - n_recent:] = float("-inf")
        top = score.topk(min(budget - n_recent, n - n_recent), dim=-1).indices
        idx.append(torch.cat([top, recent], -1).sort(-1).values)
    return idx


def head_coverage(idx, lo: int, hi: int) -> float:
    """Mean over layers and KV heads of the fraction of [lo, hi) kept."""
    return torch.stack([((ix >= lo) & (ix < hi)).float().sum(-1) / (hi - lo)
                        for ix in idx]).mean().item()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=int, default=512)
    ap.add_argument("--hops", type=int, default=8)
    ap.add_argument("--k", type=int, default=64, help="abstract slots (ours)")
    ap.add_argument("--anchors", type=int, default=512, help="verbatim entries (ours)")
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--probe", type=int, default=16)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--score-mode", default="sumsoft")
    ap.add_argument("--block", type=int, default=1)
    ap.add_argument("--spec-rounds", type=int, default=1)
    ap.add_argument("--spec-samples", type=int, default=1)
    ap.add_argument("--spec-temp", type=float, default=1.0)
    ap.add_argument("--ckpt", default=str(HERE / "ckpt" / "memtok_k32_s0_deep_frozen.pt"))
    ap.add_argument("--model", default=str(HERE / "qwen3-0.6b"))
    args = ap.parse_args()
    budget = args.anchors + args.k          # what ours spends, total

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

    conds = ["upper", "lower", "streaming", "h2o", "snapkv", "h2o_1set", "snapkv_1set",
             "ours", "ours_oracle"]
    tot = {c: 0.0 for c in conds}
    sinkhit = {c: 0.0 for c in ("streaming", "h2o", "snapkv", "h2o_1set", "snapkv_1set", "ours")}
    n_win = 0

    with torch.no_grad():
        for i in range(len(data)):
            w = data[i : i + 1].to(dev)
            # load_chunks returns windows of 2*seg, so the tail must be cut
            # explicitly -- taking w[:, cut:] gives 5120 tokens here and the
            # logits for that alone are 3 GiB.
            ctx, tail = w[:, :cut], w[:, cut : cut + args.seg]
            full = kv_only(model, ctx)
            tot["upper"] += seg_b_loss(model, cfg, tail, full, cut).item()
            tot["lower"] += seg_b_loss(model, cfg, tail, None, 0).item()

            for name, idx in (("streaming", streaming_idx(cut, budget, args.sink, dev)),
                              ("h2o_1set", h2o_idx(model, cfg, full, ctx, budget)),
                              ("snapkv_1set", snapkv_idx(model, cfg, full, ctx, budget))):
                sel = [(kk[:, :, idx], vv[:, :, idx]) for kk, vv in full]
                tot[name] += seg_b_loss(model, cfg, tail, sel, cut).item()
                sinkhit[name] += int((idx < args.sink).sum())
            # per layer and KV head, as the methods are specified
            hs = hidden_inputs(model, ctx)
            for name, hidx in (("h2o", h2o_heads(model, cfg, full, ctx, budget, hs=hs)),
                               ("snapkv", snapkv_heads(model, cfg, full, ctx, budget, hs=hs))):
                tot[name] += seg_b_loss(model, cfg, tail, gather_heads(full, hidx), cut).item()
                sinkhit[name] += head_coverage(hidx, 0, args.sink) * args.sink
            del hs

            # ours: last-segment state behind speculatively retrieved anchors
            st = compress(model, mem, ctx[:, cut - args.seg:], past=None,
                          past_len=cut - args.seg, grad=False, k=args.k)
            base = [(torch.cat([kk[:, :, :args.sink], sk], 2),
                     torch.cat([vv[:, :, :args.sink], sv], 2))
                    for (kk, vv), (sk, sv) in zip(full, st)]
            guess = speculate(model, cfg, base, tail[:, :1], cut + args.k,
                              args.probe, args.spec_samples, args.spec_temp)
            pidx = query_anchors(model, cfg, full, cut, guess, args.anchors,
                                 args.sink, args)
            panc = [(kk[:, :, pidx], vv[:, :, pidx]) for kk, vv in full]
            mix = [(torch.cat([a, s], 2), torch.cat([b, t], 2))
                   for (a, b), (s, t) in zip(panc, st)]
            tot["ours"] += seg_b_loss(model, cfg, tail, mix, cut + args.k).item()
            sinkhit["ours"] += int((pidx < args.sink).sum())

            oidx = oracle_anchors(ctx, tail, args.anchors, args.sink)
            oanc = [(kk[:, :, oidx], vv[:, :, oidx]) for kk, vv in full]
            omix = [(torch.cat([a, s], 2), torch.cat([b, t], 2))
                    for (a, b), (s, t) in zip(oanc, st)]
            tot["ours_oracle"] += seg_b_loss(model, cfg, tail, omix, cut + args.k).item()
            n_win += 1

    for c in tot:
        tot[c] /= n_win
    gap = tot["lower"] - tot["upper"]
    print(f"\n{n_win} windows | {cut} tokens | budget {budget} entries "
          f"(ours: {args.anchors} verbatim + {args.k} learned)")
    print(f"upper {tot['upper']:.4f}  lower {tot['lower']:.4f}  gap {gap:.4f} nats\n")
    print(f"  {'policy':<24} {'CE':>8} {'recovery':>10} {'sink kept':>10}")
    print("  " + "-" * 56)
    for c, label in (("streaming", "StreamingLLM"), ("h2o", "H2O (per head)"),
                     ("snapkv", "SnapKV (per head)"), ("h2o_1set", "H2O (one set)"),
                     ("snapkv_1set", "SnapKV (one set)"), ("ours", "ours (spec + slots)"),
                     ("ours_oracle", "ours, oracle anchors")):
        sh = f"{sinkhit.get(c,0)/n_win:.1f}/{args.sink}" if c in sinkhit else "-"
        print(f"  {label:<24} {tot[c]:8.4f} {(tot['lower']-tot[c])/gap:+9.1%} {sh:>10}")


if __name__ == "__main__":
    main()
