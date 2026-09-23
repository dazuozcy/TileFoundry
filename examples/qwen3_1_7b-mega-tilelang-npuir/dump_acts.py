"""Dump the check's activation files: one coherent decode context.

Teacher-forces the real kernel over README text for CTX steps, then saves the
activations mega_step consumes at the last position -- including the cache the
kernel itself built, so both sides of the check see an on-manifold history and
the comparison tolerances can stay honest (random histories amplify bf16 noise
chaotically and prove nothing).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "kernel"))
import _bootstrap  # noqa: F401

import torch  # noqa: E402
import torch_npu  # noqa: E402

import runtime_model as rm  # noqa: E402
from prepare_weights import build  # noqa: E402

WDIR = "/home/tilelang/zuochuanuong/weights/qwen3_1_7b"
OUT = os.path.join(_HERE, "acts")
CTX = 256
L, KVH, D = 28, 8, 128
MAXPOS = 40960


def main():
    os.makedirs(OUT, exist_ok=True)
    rt = rm._Runtime()
    rt.set_weights(build())

    # rope tables: the kernel eats signed-sin, the HIR (and the dump) the
    # published doubled-unsigned form
    inv = 1e6 ** (-torch.arange(0, D, 2).float() / D)
    freqs = torch.outer(torch.arange(rm.SPADG).float(), inv)
    c, s = freqs.cos(), freqs.sin()
    rt.cos_tab[:] = torch.cat([c, c], dim=1).to(rm._DEV)
    rt.sin_tab[:] = torch.cat([-s, s], dim=1).to(rm._DEV)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WDIR)
    ids = tok.encode(open(os.path.join(WDIR, "README.md")).read())
    while len(ids) < CTX:
        ids = ids + ids
    toks = ids[:CTX]

    scale = float(torch.tensor(D**-0.5, dtype=torch.bfloat16).float())
    for pos in range(CTX):
        rt.kern(*rt.args, toks[pos], pos, scale)
    torch.npu.synchronize()

    # the activations of the LAST step, in declared param order
    torch.save(torch.tensor([toks[CTX - 1]], dtype=torch.int64), f"{OUT}/token_ids.pt")
    inv_full = 1e6 ** (-torch.arange(0, D, 2).float() / D)
    freqs_full = torch.outer(torch.arange(MAXPOS).float(), inv_full)
    c_full, s_full = freqs_full.cos(), freqs_full.sin()
    torch.save(torch.cat([c_full, c_full], dim=1).bfloat16(), f"{OUT}/cos_cache.pt")
    torch.save(torch.cat([s_full, s_full], dim=1).bfloat16(), f"{OUT}/sin_cache.pt")
    torch.save(torch.tensor([CTX - 1], dtype=torch.int32), f"{OUT}/pos_ids.pt")
    torch.save(torch.full((1, 1, 1, 1), scale, dtype=torch.bfloat16), f"{OUT}/scale.pt")
    kview = rt.Kc.view(L, KVH, rm.SPADG, D).cpu()[:, :, :CTX].permute(0, 2, 1, 3)
    vview = rt.Vc.view(L, KVH, rm.SPADG, D).cpu()[:, :, :CTX].permute(0, 2, 1, 3)
    torch.save(kview.contiguous(), f"{OUT}/k_caches.pt")
    torch.save(vview.contiguous(), f"{OUT}/v_caches.pt")
    print("dumped", sorted(os.listdir(OUT)), "| ctx:", CTX, "| pos:", CTX - 1)


if __name__ == "__main__":
    main()
