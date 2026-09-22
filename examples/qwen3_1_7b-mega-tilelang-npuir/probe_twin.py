"""Controlled HIR-vs-twin probe: same weights, same activations, per-layer
k/v row comparison to find the first divergent layer."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel"))
import _bootstrap  # noqa: F401

import torch  # noqa: E402

import model  # noqa: E402
import runtime_model as rm  # noqa: E402
from tilefoundry.evaluator import evaluate  # noqa: E402
from tilefoundry.runtime.resource import DictResource  # noqa: E402

L, H, KVH, D, II, V = 28, 2048, 8, 128, 6144, 151936
Q, H_PAD, NQKV_PAD = H, 2112, 4224
MAXPOS = 40960

torch.manual_seed(0)
w = {
    "gamma_in": (1 + 0.02 * torch.randn(L, H)).bfloat16(),
    "w_qkv": (torch.randn(L, H, NQKV_PAD) * 0.02).bfloat16(),
    "gamma_q": (1 + 0.02 * torch.randn(L, D)).bfloat16(),
    "gamma_k": (1 + 0.02 * torch.randn(L, D)).bfloat16(),
    "w_o": (torch.randn(L, Q, H_PAD) * 0.02).bfloat16(),
    "gamma_post": (1 + 0.02 * torch.randn(L, H)).bfloat16(),
    "w_gu": (torch.randn(L, H, 2 * II) * 0.02).bfloat16(),
    "w_down": (torch.randn(L, II, H_PAD) * 0.02).bfloat16(),
    "gamma_final": (1 + 0.02 * torch.randn(H)).bfloat16(),
    "w_lm": (torch.randn(V, H) * 0.02).bfloat16(),
}
ctx, pos, tok = 8, 3, 5
cos = torch.randn(MAXPOS, D).bfloat16()
sin = torch.randn(MAXPOS, D).bfloat16()
kc = (torch.randn(L, ctx, KVH, D) * 0.5).bfloat16()
vc = (torch.randn(L, ctx, KVH, D) * 0.5).bfloat16()
scale = torch.full((1, 1, 1, 1), 0.088388, dtype=torch.bfloat16)
args = (torch.tensor([tok]), cos, sin, torch.tensor([pos]), scale, kc, vc)

res = DictResource(w)
ref = evaluate(model.Qwen3Mega.load(res), *args)
twin = rm.Qwen3MegaRT()
twin.load(res)
out = twin.mega_step(*args)

print("logits maxerr:", (out[0].float() - ref[0].float()).abs().max().item(),
      "| argmax:", int(out[1][0]), int(ref[1][0]))
for li in range(L):
    ke = (out[2][li].float() - ref[2][li].float()).abs().max().item()
    ve = (out[3][li].float() - ref[3][li].float()).abs().max().item()
    print(f"layer {li:2d}: k_err {ke:9.4f}   v_err {ve:9.4f}")
