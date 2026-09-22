"""Runtime twin of the Qwen3-1.7B mega-step HIR (``model.py``): the same
``mega_step`` contract served by ONE launch of the validated tilelang kernel
(``kernel/mega.py``), all 28 layers + LM head inside.

The HIR declares weights in clean logical layouts (in/out axes, padded
extents); the kernel eats row-major GEMV slabs.  ``pack`` below is the only
place that translation happens, so the two sides of a check provably consume
the same tensors.

Gate (from this directory):

  tilefoundry check runtime_model.py:Qwen3MegaRT \\
      --inputs random --weights random --dim ctx_len=8,257 \\
      --out output[0] --fn allclose --atol 0.05 --rtol 0.05 \\
      --out output[1] --fn equal \\
      --out output[2] --fn allclose --atol 0.02 --rtol 0.02 \\
      --out output[3] --fn allclose --atol 0.02 --rtol 0.02

``MEGA_S`` picks the padded context the kernel is cut for (default 768,
enough for the check's ctx_len values; run.py uses 40960).
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "kernel")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: F401  -- LD_LIBRARY_PATH fix, before torch_npu

import torch  # noqa: E402
import torch_npu  # noqa: E402

from model import Qwen3Mega  # noqa: E402
from tilefoundry.runtime import runtime_func, runtime_module  # noqa: E402

from mega import mega, mega_shapes  # noqa: E402

_L, _H, _KVH, _D = 28, 2048, 8, 128
_II, _VV = 6144, 151936

_DEV = "npu"
_S = int(os.environ.get("MEGA_S", "768"))
SH = mega_shapes(l=_L, h=_H, qh=16, kvh=_KVH, d=_D, ii=_II, vv=_VV, s=_S)
STEP, SPADG, NSL, NB = SH["step"], SH["spadg"], SH["nsl"], SH["nb"]
SPAN, SPANP, NCH = SH["span"], SH["spanp"], SH["nch"]
NQKV, NQKV_PAD = SH["nqkv"], SH["nqkv_pad"]


def _staged(*args):
    """torch.zeros on the host, then moved: direct device allocations are the
    known stale-read hazard on this stack."""
    shape, dtype = args[:-1], args[-1]
    return torch.zeros(*shape, dtype=dtype).to(_DEV)


def _staged_full(shape, value, dtype=torch.float32):
    return torch.full(shape, value, dtype=dtype).to(_DEV)


class _Runtime:
    """The compiled kernel plus its persistent GM buffers, one per process.

    Never build a second variant in one process: the device module
    registration cap (~4-5 kernels) turns extra compilations into silent
    no-op launches.
    """

    def __init__(self):
        self.kern = mega(**SH)
        # caches, staged through the host every time
        self.Kc = _staged(_L * _KVH * SPADG, _D, torch.bfloat16)
        self.Vc = _staged(_L * _KVH * SPADG, _D, torch.bfloat16)
        self.cos_tab = _staged(SPADG, _D, torch.float32)
        self.sin_tab = _staged(SPADG, _D, torch.float32)
        # scratch
        self.x = _staged(1, _H, torch.bfloat16)
        self.xn1 = _staged(1, _H, torch.bfloat16)
        self.qkv_out = _staged(1, NQKV_PAD, torch.float32)
        self.q_buf = _staged(16, _D, torch.bfloat16)
        self.attn_out = _staged(16, _D, torch.bfloat16)
        self.attn_flat = self.attn_out.view(1, _H)
        self.o_out = _staged(1, _H, torch.float32)
        self.xn2 = _staged(1, _H, torch.bfloat16)
        self.gu_out = _staged(1, 2 * _II, torch.float32)
        self.h_buf = _staged(1, _II, torch.bfloat16)
        self.d_out = _staged(1, _H, torch.float32)
        self.scores = _staged(NB * 2 * STEP, 1, torch.float32)
        self.probs = _staged(NB * 2, STEP, torch.bfloat16)
        self.part_m = _staged_full((_KVH * 2 * NSL,), -1e30, torch.float32)
        self.part_l = _staged(_KVH * 2 * NSL, torch.float32)
        self.part_acc = _staged(_KVH * 2 * NSL, _D, torch.float32)
        self.acc16 = _staged(_KVH * 2 * SH["kpad"], _D, torch.bfloat16)
        self.wrow16 = _staged(_KVH * 2, SH["kpad"], torch.bfloat16)
        self.num = _staged(_KVH * 2, _D, torch.float32)
        self.den = _staged(_KVH * 2, torch.float32)
        self.logits = _staged_full((1, NB * SPANP), float("-inf"), torch.float32)
        self.idx_tab = torch.arange(SPANP, dtype=torch.int32).to(_DEV)
        self.part_max = _staged(NB * NCH, torch.float32)
        self.part_idx = _staged(NB * NCH, torch.float32)
        self._weight_key = None
        self.args = None

    def set_weights(self, w):
        """Pack the HIR-layout weights into the kernel's GEMV slabs (once)."""
        key = tuple(id(t) for t in w.values())
        if key == self._weight_key:
            return
        dev = {}
        # w_qkv (L, H, NQKV_PAD) -> (L*NQKV_PAD, H), pad rows zero
        host = torch.zeros(_L * NQKV_PAD, _H, dtype=torch.bfloat16)
        for li in range(_L):
            host[li * NQKV_PAD : li * NQKV_PAD + NQKV] = w["w_qkv"][li].T[:NQKV]
        dev["Wqkv"] = host.to(_DEV)
        # w_o (L, H, H_PAD) -> (L*H, H): transpose, pad outputs dropped
        host = torch.zeros(_L * _H, _H, dtype=torch.bfloat16)
        for li in range(_L):
            host[li * _H : (li + 1) * _H] = w["w_o"][li][:, :_H].T
        dev["Wo"] = host.to(_DEV)
        # w_gu (L, H, 2I) -> (L*2I, H)
        host = torch.zeros(_L * 2 * _II, _H, dtype=torch.bfloat16)
        for li in range(_L):
            host[li * 2 * _II : (li + 1) * 2 * _II] = w["w_gu"][li].T
        dev["Wgu"] = host.to(_DEV)
        # w_down (L, I, H_PAD) -> (L*H, I): transpose, pad outputs dropped
        host = torch.zeros(_L * _H, _II, dtype=torch.bfloat16)
        for li in range(_L):
            host[li * _H : (li + 1) * _H] = w["w_down"][li][:, :_H].T
        dev["Wd"] = host.to(_DEV)
        dev["rms1w"] = w["gamma_in"].to(_DEV)
        dev["rms2w"] = w["gamma_post"].to(_DEV)
        dev["rmsfw"] = w["gamma_final"].to(_DEV)
        dev["qnw"] = w["gamma_q"].to(_DEV)
        dev["knw"] = w["gamma_k"].to(_DEV)
        dev["Wlm"] = w["w_lm"].to(_DEV)
        self._dev_w = dev
        self.args = [
            dev["Wqkv"], dev["Wo"], dev["Wgu"], dev["Wd"],
            dev["rms1w"], dev["rms2w"], dev["rmsfw"], dev["qnw"], dev["knw"],
            dev["Wlm"], self.Kc, self.Vc, self.cos_tab, self.sin_tab,
            self.x, self.xn1, self.qkv_out, self.q_buf, self.attn_out,
            self.attn_flat, self.o_out, self.xn2, self.gu_out, self.h_buf,
            self.d_out, self.scores, self.probs, self.part_m, self.part_l,
            self.part_acc, self.acc16, self.wrow16, self.num, self.den,
            self.logits, self.idx_tab, self.part_max, self.part_idx,
        ]
        self._weight_key = key


_RT: _Runtime | None = None


def _runtime() -> _Runtime:
    global _RT
    if _RT is None:
        _RT = _Runtime()
    return _RT


@runtime_module(Qwen3Mega)
class Qwen3MegaRT:
    @runtime_func
    def mega_step(self, token_ids, cos_cache, sin_cache, pos_ids, scale,
                  k_caches, v_caches, gamma_in, w_qkv, gamma_q, gamma_k,
                  w_o, gamma_post, w_gu, w_down, gamma_final, w_lm):
        """One decode step, one kernel launch: caches hold [0, pos), this
        step's rows land at pos, logits cover the live vocabulary."""
        rt = _runtime()
        ctx = int(k_caches.shape[1])
        pos = int(pos_ids[0])
        if not 0 <= pos < min(ctx, SPADG):
            raise ValueError(
                f"mega_step: pos {pos} outside the live cache extent "
                f"(ctx_len={ctx}, kernel SPADG={SPADG})"
            )
        rt.set_weights({
            "gamma_in": gamma_in, "w_qkv": w_qkv, "gamma_q": gamma_q,
            "gamma_k": gamma_k, "w_o": w_o, "gamma_post": gamma_post,
            "w_gu": w_gu, "w_down": w_down, "gamma_final": gamma_final,
            "w_lm": w_lm,
        })
        # stage the given cache prefix (rows >= pos are dead but must read
        # back identically on both sides, so stage the whole extent)
        rt.Kc.zero_(); rt.Vc.zero_()
        for li in range(_L):
            for kv in range(_KVH):
                r0 = (li * _KVH + kv) * SPADG
                rt.Kc[r0 : r0 + ctx] = k_caches[li, :ctx, kv, :].to(_DEV)
                rt.Vc[r0 : r0 + ctx] = v_caches[li, :ctx, kv, :].to(_DEV)
        rt.cos_tab[:ctx] = cos_cache[:ctx].float().to(_DEV)
        # the HIR rope negates inside rotate_half; the kernel's table carries
        # the sign instead: negate the first half of every sin row
        signed_sin = sin_cache[:ctx].float().clone()
        signed_sin[:, : _D // 2] = -signed_sin[:, : _D // 2]
        rt.sin_tab[:ctx] = signed_sin.to(_DEV)

        rt.kern(*rt.args, int(token_ids[0]), pos, float(scale.float()))
        torch.npu.synchronize()

        logits = rt.logits.cpu()[0, :_VV].unsqueeze(0)
        pm = rt.part_max.view(NB, NCH).cpu()
        pi = rt.part_idx.view(NB, NCH).cpu()
        flat = int(torch.argmax(pm))
        b, c = flat // NCH, flat % NCH
        next_token = torch.tensor([int(pi[b, c]) + b * SPAN], dtype=torch.int64)
        k_rows = rt.Kc.view(_L, _KVH, SPADG, _D).cpu()[:, :, pos : pos + 1].permute(0, 2, 1, 3).clone()
        v_rows = rt.Vc.view(_L, _KVH, SPADG, _D).cpu()[:, :, pos : pos + 1].permute(0, 2, 1, 3).clone()
        return logits, next_token, k_rows, v_rows
