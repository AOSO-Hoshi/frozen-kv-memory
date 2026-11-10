#!/usr/bin/env python3
"""A retrieval task where the answer can only be copied, never inferred.

Section 3 of the paper argues that the compression ceiling is made of precise
recall, using a per-token split of held-out continuations: tokens that appeared
in the context recover at +45%, tokens that never appeared at +202%. That is a
statistical claim about perplexity, and it is the form a reviewer is least
likely to find convincing.

This states the same claim as a task. Hide a uniformly random five-digit code in
a long context, ask for it back, and see which memory can produce it. The
copyability is total rather than statistical: the code is drawn at generation
time, so no prior over language helps. A model that did not read the needle
cannot beat chance, and a model that compressed the context into 64 abstract
slots can only answer if those slots stored the digits themselves.

The prediction, if Section 3 is right: the abstract state fails almost
completely here while recovering most of the loss on ordinary text, and exact
anchors are what closes the gap -- but only when retrieval actually finds the
needle, which is a selection problem and not a compression one.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from memtok import MemoryTokens, compress
from train import cache_from, load_chunks
# The same deployable retrieval the paper reports, rather than a second
# implementation: it scores context keys against the query's projected q
# manually, which also sidesteps sdpa refusing output_attentions.
from hybrid import query_anchors
# Same three training-free policies as Sec. 4.1, on a task whose value is
# concentrated in one span rather than spread over the context. If the two
# regimes are real, the ordering among them should invert here.
from baseline_cmp import (streaming_idx, h2o_idx, snapkv_idx, hidden_inputs,
                          h2o_heads, snapkv_heads, snapkv_question, gather_heads,
                          head_coverage)

HERE = Path(__file__).resolve().parent

# Deliberately mundane keys: the difficulty must come from the random value, not
# from an exotic phrasing the model has never seen.
KEYS = ["warehouse 7", "the north gate", "locker 42", "the service panel",
        "the rear entrance", "cabinet 13", "the roof access", "unit 91"]


# memory.py already has the careful version: it calls the inner model so the
# 151k-vocabulary head is never evaluated over 4096 positions, and it unpacks
# the Cache object the way current transformers exposes it.
from memory import kv_only


def answer_ce(model, cfg, prompt_ids, ans_ids, past, offset):
    """Mean cross-entropy of the answer tokens given prompt and whatever past.

    The prompt is scored too (the model must read it) but only the answer's
    tokens enter the loss, so the number is about the code and not about the
    boilerplate around it.
    """
    ids = torch.cat([prompt_ids, ans_ids], dim=1)
    B, S = ids.shape
    pos = torch.arange(offset, offset + S, device=ids.device)
    kw = {}
    if past is not None:
        kw["past_key_values"] = cache_from(past, cfg)
        plen = past[0][0].shape[2]
        kw["attention_mask"] = torch.ones(B, plen + S, device=ids.device, dtype=torch.long)
        kw["cache_position"] = torch.arange(plen, plen + S, device=ids.device)
    with torch.no_grad():
        logits = model(input_ids=ids, position_ids=pos.unsqueeze(0).expand(B, -1),
                       **kw).logits
    n_ans = ans_ids.shape[1]
    # predict token t from position t-1; the answer starts at len(prompt)
    lg = logits[:, prompt_ids.shape[1] - 1 : -1].reshape(-1, logits.size(-1)).float()
    tgt = ans_ids.reshape(-1)
    return F.cross_entropy(lg, tgt).item()


def greedy_hit(model, cfg, prompt_ids, ans_text, tok, past, offset, max_new=8):
    """Does greedy decoding actually emit the code? The task-level metric."""
    ids = prompt_ids
    out_ids = []
    plen0 = 0 if past is None else past[0][0].shape[2]
    with torch.no_grad():
        for i in range(max_new):
            S = ids.shape[1]
            pos = torch.arange(offset, offset + S, device=ids.device)
            kw = {}
            if past is not None:
                kw["past_key_values"] = cache_from(past, cfg)
                kw["attention_mask"] = torch.ones(1, plen0 + S, device=ids.device,
                                                  dtype=torch.long)
                kw["cache_position"] = torch.arange(plen0, plen0 + S, device=ids.device)
            logits = model(input_ids=ids, position_ids=pos.unsqueeze(0), **kw).logits
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            out_ids.append(nxt.item())
            ids = torch.cat([ids, nxt], dim=1)
    return ans_text.strip() in tok.decode(out_ids)


def build(tok, chunks, depth: float, rng, ctx_len: int, dev):
    """One sample: haystack with a needle at `depth`, plus question and answer."""
    key = rng.choice(KEYS)
    val = str(rng.randrange(10000, 100000))
    needle_txt = f" The secret code for {key} is {val}. "
    enc = tok(needle_txt, return_tensors="pt", return_offsets_mapping=True)
    needle = enc.input_ids.to(dev)
    # which of the needle's tokens are the digits themselves, as opposed to
    # the key phrase the question repeats
    c0 = needle_txt.index(val)
    code_off = [j for j, (a, b) in enumerate(enc.offset_mapping[0].tolist())
                if a < c0 + len(val) and b > c0]
    prompt = tok(f"\nQuestion: What is the secret code for {key}?\n"
                 f"Answer: The secret code for {key} is",
                 return_tensors="pt").input_ids.to(dev)
    ans = tok(f" {val}", return_tensors="pt").input_ids.to(dev)

    body = chunks[rng.randrange(len(chunks))].unsqueeze(0).to(dev)
    while body.shape[1] < ctx_len:
        body = torch.cat([body, chunks[rng.randrange(len(chunks))].unsqueeze(0).to(dev)], 1)
    body = body[:, : ctx_len - needle.shape[1]]
    cut = int(body.shape[1] * depth)
    ctx = torch.cat([body[:, :cut], needle, body[:, cut:]], dim=1)
    # where the needle sits, so an oracle retriever can be given exactly it
    span = (cut, cut + needle.shape[1])
    code = torch.tensor([cut + j for j in code_off], device=dev)
    return ctx, prompt, ans, f" {val}", span, code


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--k", type=int, default=64, help="abstract slots")
    ap.add_argument("--anchors", type=int, default=512)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--seg", type=int, default=512)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--depths", default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=str(HERE / "ckpt" / "memtok_k32_pl192_deep_frozen.pt"))
    ap.add_argument("--model", default=str(HERE / "qwen3-0.6b"))
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--score-mode", default="sumsoft",
                    choices=["mean", "maxhead", "sumsoft"])
    ap.add_argument("--block", type=int, default=1,
                    help="retrieve contiguous spans of this size")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map={"": 0},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True)).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16).to(dev).eval()
    cfg = model.config

    sd = torch.load(args.ckpt)
    sd = sd["mem"] if "mem" in sd else sd
    depth_n = sd["deep"].shape[0] if "deep" in sd else 0
    mem = MemoryTokens(sd["emb"].shape[0], cfg.hidden_size, depth=depth_n).to(dev)
    mem.load_state_dict(sd)
    mem.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    chunks = load_chunks(tok, args.seg, 400)
    rng = random.Random(args.seed)
    depths = [float(x) for x in args.depths.split(",")]

    conds = ["upper", "lower", "abstract", "streaming", "h2o", "snapkv", "snapkv_q",
             "snapkv_q_slots", "h2o_1set", "snapkv_1set", "anchors", "hybrid", "oracle"]
    pols = ("streaming", "h2o", "snapkv", "snapkv_q", "h2o_1set", "snapkv_1set", "anchors")
    ce = {d: {c: 0.0 for c in conds} for d in depths}
    em = {d: {c: 0 for c in conds} for d in depths}
    # whether each policy paged in the needle at all, separating "never found
    # it" from "found it and still could not use it"
    hit = {d: {c: 0 for c in pols} for d in depths}
    # Fraction of the needle's tokens retrieved, not just whether any were.
    # "hit" counts a policy as finding the needle if it kept a single token of
    # it, which is how depth 0.9 could show 100% hits and 0% exact match: the
    # span was clipped, and a five-digit code with a digit missing is wrong.
    cover = {d: {c: 0.0 for c in pols} for d in depths}
    # The same, split into the digits and the rest of the needle. The question
    # repeats the key phrase but not the code, so a policy that scores tokens
    # one at a time can keep the key and drop the digits next to it.
    cover_code = {d: {c: 0.0 for c in pols} for d in depths}
    cover_key = {d: {c: 0.0 for c in pols} for d in depths}
    budget = args.anchors + args.k       # what the hybrid spends in total

    for d in depths:
        for i in range(args.n):
            ctx, prompt, ans, ans_txt, span, code = build(tok, chunks, d, rng, args.ctx, dev)
            n = ctx.shape[1]
            full = kv_only(model, ctx)

            # the learned state over the last segment only -- the configuration
            # the paper reports, since rolling dilutes (Sec. 5)
            st = compress(model, mem, ctx[:, n - args.seg:], past=None,
                          past_len=n - args.seg, grad=False, k=args.k)

            # Deployable retrieval. Unlike the perplexity setting, here there
            # is a real query available at no cost: the question itself names
            # what is wanted. This is the friendliest case for retrieval, which
            # makes it the right one for asking whether retrieval is enough.
            idx = query_anchors(model, cfg, full, n, prompt,
                                args.anchors, args.sink, args)
            anc = [(k[:, :, idx], v[:, :, idx]) for k, v in full]
            span_n = span[1] - span[0]
            in_span = ((idx >= span[0]) & (idx < span[1])).sum().item()
            hit[d]["anchors"] += bool(in_span)
            cover[d]["anchors"] += in_span / span_n
            n_code = torch.isin(code, idx).sum().item()
            cover_code[d]["anchors"] += n_code / len(code)
            cover_key[d]["anchors"] += (in_span - n_code) / (span_n - len(code))

            # training-free policies at the same total budget
            bidx = {
                "streaming": streaming_idx(n, budget, args.sink, dev),
                "h2o_1set": h2o_idx(model, cfg, full, ctx, budget),
                "snapkv_1set": snapkv_idx(model, cfg, full, ctx, budget),
            }
            bsel = {}
            for bn, bi in bidx.items():
                bsel[bn] = [(kk[:, :, bi], vv[:, :, bi]) for kk, vv in full]
                k_in = ((bi >= span[0]) & (bi < span[1])).sum().item()
                hit[d][bn] += bool(k_in)
                cover[d][bn] += k_in / span_n
                c_in = torch.isin(code, bi).sum().item()
                cover_code[d][bn] += c_in / len(code)
                cover_key[d][bn] += (k_in - c_in) / (span_n - len(code))
            # per layer and KV head, as the methods are specified
            hs = hidden_inputs(model, ctx)
            for bn, hidx in (("h2o", h2o_heads(model, cfg, full, ctx, budget, hs=hs)),
                             ("snapkv", snapkv_heads(model, cfg, full, ctx, budget, hs=hs)),
                             ("snapkv_q", snapkv_question(model, cfg, ctx, prompt, budget))):
                bsel[bn] = gather_heads(full, hidx)
                sc = head_coverage(hidx, span[0], span[1])
                cc = head_coverage(hidx, int(code[0]), int(code[-1]) + 1)
                hit[d][bn] += sc > 0
                cover[d][bn] += sc
                cover_code[d][bn] += cc
                cover_key[d][bn] += (sc * span_n - cc * len(code)) / (span_n - len(code))
            del hs
            # the strong selector with the learned state behind it, at the same
            # total budget: does the state help, as it does on natural text?
            qsel = gather_heads(full, snapkv_question(model, cfg, ctx, prompt, args.anchors))
            bsel["snapkv_q_slots"] = [(torch.cat([a, s], 2), torch.cat([b, t], 2))
                                      for (a, b), (s, t) in zip(qsel, st)]

            # oracle retrieval: hand it the needle's own positions, padded with
            # the sink. This is the upper bound of any selection policy.
            oid = torch.cat([torch.arange(args.sink, device=dev),
                             torch.arange(span[0], span[1], device=dev)]).unique()
            ora = [(k[:, :, oid], v[:, :, oid]) for k, v in full]

            mix = [(torch.cat([a, s], 2), torch.cat([b, t], 2))
                   for (a, b), (s, t) in zip(anc, st)]
            omix = [(torch.cat([a, s], 2), torch.cat([b, t], 2))
                    for (a, b), (s, t) in zip(ora, st)]

            for name, past, off in (("upper", full, n),
                                    ("lower", None, 0),
                                    ("abstract", st, n),
                                    ("streaming", bsel["streaming"], n),
                                    ("h2o", bsel["h2o"], n),
                                    ("snapkv", bsel["snapkv"], n),
                                    ("snapkv_q", bsel["snapkv_q"], n),
                                    ("snapkv_q_slots", bsel["snapkv_q_slots"], n + args.k),
                                    ("h2o_1set", bsel["h2o_1set"], n),
                                    ("snapkv_1set", bsel["snapkv_1set"], n),
                                    ("anchors", anc, n),
                                    ("hybrid", mix, n + args.k),
                                    ("oracle", omix, n + args.k)):
                ce[d][name] += answer_ce(model, cfg, prompt, ans, past, off)
                em[d][name] += greedy_hit(model, cfg, prompt, ans_txt, tok, past, off)

    print(f"\n{args.n} samples x {len(depths)} depths | ctx={args.ctx} "
          f"| budget {budget} entries (hybrid: {args.anchors} + {args.k})\n")
    print("=== exact match by needle depth ===")
    print(f"{'depth':>6} " + "".join(f"{c:>12}" for c in conds))
    print("  " + "-" * (6 + 12 * len(conds)))
    for d in depths:
        print(f"{d:>6.1f} " + "".join(f"{em[d][c]/args.n:>11.0%} " for c in conds))
    hcols = pols
    print("\n=== found any of the needle / fraction of it retrieved ===")
    print(f"{'depth':>6} " + "".join(f"{c:>16}" for c in hcols))
    for d in depths:
        print(f"{d:>6.1f} " + "".join(
            f"{hit[d][c]/args.n:>7.0%}/{cover[d][c]/args.n:>7.0%} " for c in hcols))
    print("\n=== fraction retrieved: key phrase / digits ===")
    print(f"{'depth':>6} " + "".join(f"{c:>16}" for c in hcols))
    for d in depths:
        print(f"{d:>6.1f} " + "".join(
            f"{cover_key[d][c]/args.n:>7.0%}/{cover_code[d][c]/args.n:>7.0%} " for c in hcols))

    agg = {}
    for c in conds:
        agg[c] = (sum(ce[d][c] for d in depths) / (args.n * len(depths)),
                  sum(em[d][c] for d in depths) / (args.n * len(depths)))
    gap = agg["lower"][0] - agg["upper"][0]
    print(f"\n=== overall (gap {gap:.3f} nats, answer tokens only) ===")
    print(f"  {'condition':<12} {'CE':>8} {'recovery':>10} {'EM':>6}")
    print("  " + "-" * 40)
    for c in conds:
        print(f"  {c:<12} {agg[c][0]:8.3f} {(agg['lower'][0]-agg[c][0])/gap:+9.1%} "
              f"{agg[c][1]:>5.0%}")


if __name__ == "__main__":
    main()
