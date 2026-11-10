"""Hand-computed attention must be the model's attention.

Every selection policy here (our query/speculative anchors, H2O, SnapKV)
scores context positions by recomputing last-layer attention by hand, since
sdpa does not return weights. v1 of the paper did this without rotating the
query, and its scores agreed with the model's real top-32 on 1 position in 32.
This compares against eager attention, which does return weights.

    source ~/ctfenv/bin/activate && python tests/test_attention_match.py
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from baseline_cmp import (_queries, _keys, hidden_inputs, layer_queries,  # noqa: E402
                          _attn_rows, gather_heads)
from memory import kv_only                 # noqa: E402

TEXT = ("The river runs past the old mill, and every spring the water rises "
        "until the lower field floods. Farmers along the bank keep records of "
        "each year's high mark, carved into a post by the gate. ") * 12


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = str(HERE / "qwen3-0.6b")
    tok = AutoTokenizer.from_pretrained(m)
    model = AutoModelForCausalLM.from_pretrained(
        m, dtype=torch.float32, attn_implementation="eager").cuda().eval()
    ctx = tok(TEXT, return_tensors="pt").input_ids.cuda()
    n = ctx.shape[1]

    with torch.no_grad():
        truth = model(ctx, output_attentions=True).attentions[-1][0].float()
    full = kv_only(model, ctx)
    q, hd = _queries(model, model.config, ctx)
    k = _keys(model.config, full).float()
    pos = torch.arange(n, device=ctx.device)
    a = torch.einsum("thd,hcd->thc", q.float(), k) / hd ** 0.5
    a = a.masked_fill(pos[None, None, :] > pos[:, None, None], float("-inf"))
    a = a.softmax(-1).permute(1, 0, 2)

    err = (a - truth).abs().max().item()
    top_t = set(truth[:, -32:].sum((0, 1)).topk(32).indices.tolist())
    top_a = set(a[:, -32:].sum((0, 1)).topk(32).indices.tolist())
    print(f"max|diff| {err:.2e}  top-32 overlap {len(top_t & top_a)}/32")
    assert err < 1e-3, "hand-computed attention differs from the model's"
    assert len(top_t & top_a) >= 31, "selection would differ from the model's"

    # Per-layer queries (the per-head SnapKV/H2O path) against every layer's
    # own eager attention, first, middle and last.
    with torch.no_grad():
        atts = model(ctx, output_attentions=True).attentions
    hs = hidden_inputs(model, ctx)
    for i in (0, len(full) // 2, len(full) - 1):
        qi, hd = layer_queries(model, model.config, hs[i][0], i, pos)
        ai = _attn_rows(qi, full[i][0][0], pos, hd)
        e = (ai - atts[i][0].float()).abs().max().item()
        print(f"layer {i:2d} max|diff| {e:.2e}")
        assert e < 1e-3, f"layer {i} hand attention differs"

    # Gathering every position per head must reproduce the full cache exactly.
    kvh = full[0][0].shape[1]
    every = [torch.arange(n, device=ctx.device).expand(kvh, -1) for _ in full]
    for (k, v), (k2, v2) in zip(full, gather_heads(full, every)):
        assert torch.equal(k, k2) and torch.equal(v, v2)
    print("ok")


if __name__ == "__main__":
    main()
