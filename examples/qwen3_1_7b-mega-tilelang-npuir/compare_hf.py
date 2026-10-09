"""Greedy token match: the mega kernel vs HuggingFace transformers.

Both decode the same prompt greedily; report the first divergence (if any).
Near-tie logits can flip between equally-valid bf16 rounding paths, so a
divergence position is reported with its logit gap.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "kernel")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MEGA_S", "40960")

import _bootstrap  # noqa: F401

import torch  # noqa: E402
import torch_npu  # noqa: E402

from run import MegaDecoder, WDIR  # noqa: E402


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 64

    dec = MegaDecoder()
    ids = dec.tok.encode(prompt)
    ours, _, _ = dec.generate(ids, n)
    print(f"ours  : {dec.tok.decode(ours)!r}")

    from transformers import AutoModelForCausalLM

    dev = "npu" if torch.npu.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(WDIR, torch_dtype=torch.bfloat16).to(dev)
    model.eval()
    with torch.no_grad():
        out = model.generate(
            torch.tensor([ids], device=dev),
            max_new_tokens=n, do_sample=False,
            attention_mask=torch.ones((1, len(ids)), device=dev, dtype=torch.int64),
        )
    hf = out[0, len(ids):].tolist()
    print(f"hf    : {dec.tok.decode(hf)!r}")

    same = 0
    for a, b in zip(ours, hf):
        if a != b:
            break
        same += 1
    print(f"match : {same}/{min(len(ours), len(hf))} tokens", end="")
    if same == len(ours) == len(hf):
        print("  -- FULL MATCH")
    else:
        print(f"  -- diverges at step {same}")
        # show the logit gap at the divergence to judge whether it is a
        # legitimate near-tie flip
        pos = len(ids) + same - 1
        rt = dec.rt
        rt.kern(*rt.args, ours[same - 1] if same > 0 else ids[-1], pos, dec.scale)
        torch.npu.synchronize()
        lg = rt.logits.cpu()[0]
        top = torch.topk(lg, 3)
        print(f"        kernel top3 at the flip: {[(int(t), float(v)) for t, v in zip(top.indices, top.values)]}")


if __name__ == "__main__":
    main()
