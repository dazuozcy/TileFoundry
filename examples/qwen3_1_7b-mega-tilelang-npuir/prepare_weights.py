"""Build the check's weight resource: the HF checkpoint re-laid-out into the
HIR's declared ConstTensor names/shapes, saved as one safetensors directory.

Output: prepared/model.safetensors with keys
  gamma_in, w_qkv, gamma_q, gamma_k, w_o, gamma_post, w_gu, w_down,
  gamma_final, w_lm
in the exact shapes model.py declares, so
``--weights ckpt:prepared`` hands the SAME tensors to the HIR interpreter and
the runtime twin.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from safetensors.torch import save_file

WDIR = "/home/tilelang/zuochuanuong/weights/qwen3_1_7b"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prepared")

L, H, QH, KVH, D = 28, 2048, 16, 8, 128
II, V = 6144, 151936
H_PAD = 2112          # 24*88, o_proj / down_proj output padding
NQKV = QH * D + 2 * KVH * D
NQKV_PAD = 4224       # 24*176


def build():
    """The HIR-declared weights, re-laid-out from the HF checkpoint."""
    ts = {}
    for f in sorted(glob.glob(os.path.join(WDIR, "model-*.safetensors"))):
        from safetensors import safe_open

        with safe_open(f, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                ts[k] = sf.get_tensor(k)

    lay = lambda i, name: ts[f"model.layers.{i}.{name}"]  # noqa: E731

    # w_qkv (L, H, NQKV_PAD): (in, out), pad columns zero
    w_qkv = torch.zeros(L, H, NQKV_PAD, dtype=torch.bfloat16)
    for i in range(L):
        wqkv = torch.cat(
            [lay(i, f"self_attn.{p}_proj.weight") for p in ("q", "k", "v")], dim=0,
        )  # (NQKV, H) (out, in)
        w_qkv[i, :, :NQKV] = wqkv.T
    # w_o (L, H, H_PAD): (in, out), pad columns zero
    w_o = torch.zeros(L, H, H_PAD, dtype=torch.bfloat16)
    for i in range(L):
        w_o[i, :, :H] = lay(i, "self_attn.o_proj.weight").T
    # w_gu (L, H, 2I): (in, out)
    w_gu = torch.stack(
        [
            torch.cat([lay(i, "mlp.gate_proj.weight"), lay(i, "mlp.up_proj.weight")], dim=0).T
            for i in range(L)
        ]
    ).contiguous()
    # w_down (L, I, H_PAD): (in, out), pad columns zero
    w_down = torch.zeros(L, II, H_PAD, dtype=torch.bfloat16)
    for i in range(L):
        w_down[i, :, :H] = lay(i, "mlp.down_proj.weight").T
    w_lm = ts.get("lm_head.weight", ts["model.embed_tokens.weight"]).contiguous()

    out = {
        "gamma_in": torch.stack([lay(i, "input_layernorm.weight") for i in range(L)]).contiguous(),
        "w_qkv": w_qkv.contiguous(),
        "gamma_q": torch.stack([lay(i, "self_attn.q_norm.weight") for i in range(L)]).contiguous(),
        "gamma_k": torch.stack([lay(i, "self_attn.k_norm.weight") for i in range(L)]).contiguous(),
        "w_o": w_o.contiguous(),
        "gamma_post": torch.stack([lay(i, "post_attention_layernorm.weight") for i in range(L)]).contiguous(),
        "w_gu": w_gu,
        "w_down": w_down.contiguous(),
        "gamma_final": ts["model.norm.weight"].contiguous(),
        "w_lm": w_lm.bfloat16() if w_lm.dtype != torch.bfloat16 else w_lm,
    }
    for k, v in out.items():
        if v.dtype != torch.bfloat16:
            out[k] = v.bfloat16()
    return out


def main():
    out = build()
    for k, v in out.items():
        print(f"{k:12s} {tuple(v.shape)} {v.dtype}")

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "model.safetensors")
    save_file(out, path)
    print("wrote", path)


if __name__ == "__main__":
    main()
