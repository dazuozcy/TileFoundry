r"""Runtime twin of the Qwen3-1.7B mega-step HIR (``model.py``).

The same ``mega_step`` contract served by ONE launch of the raw AscendC mix
kernel (``kernel/mega.cpp`` -> ``kernel/libmegak.so``, bisheng build,
``<<<>>>`` stub via ctypes) -- the same twin contract as the npuir example's
``runtime_model.py``, with the tilelang-npuir launch replaced by the AscendC
launch.

The HIR declares weights in clean logical layouts (in/out axes, padded
extents); the kernel eats NZ-fractal GEMV slabs and zero-padded fractal KV
planes.  ``_Runtime.set_weights`` and ``_Runtime.stage_caches`` below are
the only places those translations happen, so the two sides of a check
provably consume the same tensors.

Gate (from this directory; ``acts``/``prepared`` are the teacher-forced
activations and the HIR-layout checkpoint repack -- the same artifacts the
npuir example's gate consumes; ``QWEN3_CKPT`` points ``model.py`` at the
published checkpoint whose config it reads):

  QWEN3_CKPT=<checkpoint dir> ASCEND_RT_VISIBLE_DEVICES=0 \\
  tilefoundry check runtime_model.py:Qwen3MegaRT.mega_step \\
      --inputs files:acts/token_ids.pt,acts/cos_cache.pt,acts/sin_cache.pt,acts/pos_ids.pt,acts/scale.pt,acts/k_caches.pt,acts/v_caches.pt \\
      --weights ckpt:prepared --dim ctx_len=256 \\
      --out output[0] --fn allclose --atol 0.15 --rtol 0.05 --fn rel_l2 --max 0.02 \\
      --out output[1] --fn equal \\
      --out output[2] --fn allclose --atol 0.3  --rtol 0.05 --fn rel_l2 --max 0.05 \\
      --out output[3] --fn allclose --atol 1.0  --rtol 0.05 --fn rel_l2 --max 0.05

Two contract notes:

* ``scale``: the kernel applies the f32 constant 128**-0.5 (the checkpoint
  publishes that value in bf16); the twin asserts the input matches, so a
  random ``scale`` is rejected rather than silently mis-executed.
* cache rows >= pos are dead by the HIR contract (masked to -1e30).  The
  kernel's 256-position scan tiles instead require every row past the live
  prefix to read exactly ZERO (tail scores are q*0 and tail probs multiply
  zero V rows), so staging writes only [0, pos) and zero-fills the rest of
  each plane -- semantically identical masking.  The row at pos itself is
  computed and appended by the kernel.
"""
from __future__ import annotations

import ctypes
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402  (registers the npu device with torch)
from model import Qwen3Mega  # noqa: E402

from tilefoundry.runtime import runtime_func, runtime_module  # noqa: E402

# ---------------- constants (mirror kernel/mega.cpp) ----------------
L, H, QH, KVH, D = 28, 2048, 16, 8, 128
II, VV = 6144, 151936
NB, SLT, NSL = 24, 256, 3
STEP, SPADG = 13824, 41472
NQKV, NQKV_PAD = 4096, 4224
NO_PAD, NGU, ND_PAD = 2304, 12288, 2304
SPAN, SPANP, NCH = 6336, 8192, 4
SB = SPADG // 16                   # 16-position blocks per cache plane
KREGION = 8 * SB * 256             # elements per K plane: [d_blk][s_blk]..
VREGION = SB * 8 * 256             # elements per V plane: [s_blk][d_blk]..

_DEV = "npu"

# kernel launch (verified v2 recipe -- see kernel/build.sh):
# libmegak.so embeds the device binary + run_megak stub; the bisheng <<<>>>
# plugin registers and launches it.  chip 3 has MTE ROB ECC faults.
assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == "0", \
    "run with ASCEND_RT_VISIBLE_DEVICES=0"
_lib = ctypes.CDLL(os.path.join(_HERE, "kernel", "libmegak.so"))
_lib.run_megak.argtypes = (
    [ctypes.c_void_p] * 36 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
)
_rt = ctypes.CDLL(os.path.join(os.environ["ASCEND_HOME_PATH"], "lib64", "libruntime.so"))
_rt.rtGetC2cCtrlAddr.restype = ctypes.c_int32
_rt.rtGetC2cCtrlAddr.argtypes = [
    ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32)
]


def _pack_w(w: torch.Tensor, npad: int) -> torch.Tensor:
    """(N, K) row-major -> NZ fractal pack, N zero-padded to npad."""
    n, k = w.shape
    wp = torch.zeros(npad, k, dtype=torch.bfloat16)
    wp[:n] = w
    return (wp.reshape(npad // 16, 16, k // 16, 16)
            .permute(2, 0, 1, 3).contiguous().view(-1))


class _Runtime:
    """The ctypes-launched kernel plus its persistent GM buffers.

    One per process; the device module registration cap forbids a second
    copy.
    """

    def __init__(self):
        # NPU context first, then the FFTS mailbox address the kernel polls
        torch.zeros(16, device=_DEV)
        a, ln = ctypes.c_uint64(0), ctypes.c_uint32(0)
        assert _rt.rtGetC2cCtrlAddr(ctypes.byref(a), ctypes.byref(ln)) == 0
        self.ffts = a.value
        assert self.ffts != 0, "ffts=0 (npu context not initialized?)"

        # caches / rope tables / sync workspace (fixed)
        self.KcP = torch.zeros(L * KVH, KREGION, dtype=torch.bfloat16, device=_DEV).view(-1)
        self.VcP = torch.zeros(L * KVH, VREGION, dtype=torch.bfloat16, device=_DEV).view(-1)
        self.cosT = torch.zeros(SPADG, D, dtype=torch.float32, device=_DEV)
        self.sinT = torch.zeros(SPADG, D, dtype=torch.float32, device=_DEV)
        self.idx_tab = torch.arange(SPANP, dtype=torch.float32, device=_DEV)
        # [0,1536) SyncAll GM ws + [4096,...) seq slots; host-zeroed ONCE
        # (SyncAll counters are monotonic; the tag advances 512 per launch)
        self.syncws = torch.zeros(4096 + 48 * 256 * 16, dtype=torch.int32, device=_DEV)

        # activation / scratch buffers
        mk = lambda *s, dt=torch.bfloat16: torch.empty(*s, dtype=dt, device=_DEV)  # noqa: E731
        self.xEmb = mk(H)
        self.xn1P = mk(H * 16)
        self.attnP = mk(H * 16)
        self.xn2P = mk(H * 16)
        self.xnfP = mk(H * 16)
        self.hP = mk(II * 16)
        self.qkv_out = mk(16, NQKV_PAD, dt=torch.float32)
        self.o_out = mk(16, NO_PAD, dt=torch.float32)
        self.gu_out = mk(16, NGU, dt=torch.float32)
        self.d_out = mk(16, ND_PAD, dt=torch.float32)
        self.q2P = mk(NB, 8 * 256)
        self.scores16 = mk(NB, 16, STEP, dt=torch.float32)
        self.probs2P = mk(NB, 54 * 16 * 256)
        self.part_m = mk(48 * 16, dt=torch.float32)
        self.part_l = mk(48 * 16, dt=torch.float32)
        self.partAccP = mk(NB, 16, D, dt=torch.float32)
        self.logits = mk(16, NB * SPANP, dt=torch.float32)
        self.part_max = mk(NB * NCH * 16, dt=torch.float32)
        self.part_idx = mk(NB * NCH * 16, dt=torch.float32)

        self.launch_no = 0
        self._weight_key = None

    # ---- weights: HIR logical layouts -> kernel fractal packs (once) ----
    def set_weights(self, w):
        key = tuple(id(t) for t in w.values())
        if key == self._weight_key:
            return
        cpu = {k: t.detach().to("cpu") for k, t in w.items()}
        # w_qkv (L, H, NQKV_PAD) -> per layer (NQKV_PAD, H), pad rows zero
        wqkv = torch.cat([
            _pack_w(cpu["w_qkv"][li].T[:NQKV], NQKV_PAD) for li in range(L)])
        # w_o (L, Q, H_PAD) -> (NO_PAD, Q); w_down (L, I, H_PAD) -> (ND_PAD, I)
        wo = torch.cat([
            _pack_w(cpu["w_o"][li][:, :H].T, NO_PAD) for li in range(L)])
        wg = torch.cat([
            _pack_w(cpu["w_gu"][li].T, NGU) for li in range(L)])
        wd = torch.cat([
            _pack_w(cpu["w_down"][li][:, :H].T, ND_PAD) for li in range(L)])
        # w_lm (V, H) -> (NB*SPAN, H) padded
        wlm = _pack_w(cpu["w_lm"], NB * SPAN)
        dev = {
            "WqkvP": wqkv.to(_DEV), "WoP": wo.to(_DEV), "WguP": wg.to(_DEV),
            "WdP": wd.to(_DEV), "WlmP": wlm.to(_DEV),
            "rms1w": cpu["gamma_in"].to(torch.bfloat16).to(_DEV),
            "rms2w": cpu["gamma_post"].to(torch.bfloat16).to(_DEV),
            "rmsfw": cpu["gamma_final"].to(torch.bfloat16).to(_DEV),
            "qnw": cpu["gamma_q"].to(torch.bfloat16).to(_DEV),
            "knw": cpu["gamma_k"].to(torch.bfloat16).to(_DEV),
        }
        self.ptrs = [
            dev["WqkvP"], dev["WoP"], dev["WguP"], dev["WdP"], dev["WlmP"],
            dev["rms1w"], dev["rms2w"], dev["rmsfw"], dev["qnw"], dev["knw"],
            self.cosT, self.sinT, self.KcP, self.VcP, self.xEmb,
            self.xn1P, self.qkv_out, self.q2P, self.scores16,
            self.probs2P, self.part_m, self.part_l, self.partAccP,
            self.attnP, self.o_out, self.xn2P, self.gu_out, self.hP,
            self.d_out, self.logits, self.idx_tab, self.part_max,
            self.part_idx, self.syncws, self.xnfP,
        ]
        self._weight_key = key

    # ---- caches: logical (L, C, KVH, D) -> zero-padded fractal planes ----
    def stage_caches(self, k_caches, v_caches, pos: int):
        nb = (pos + 15) // 16               # live 16-position blocks
        kc = k_caches[:, :pos, :, :].detach().to("cpu").permute(0, 2, 1, 3) \
            .reshape(L * KVH, pos, D)
        vc = v_caches[:, :pos, :, :].detach().to("cpu").permute(0, 2, 1, 3) \
            .reshape(L * KVH, pos, D)
        kp = torch.zeros(L * KVH, nb * 16, D, dtype=torch.bfloat16)
        vp = torch.zeros(L * KVH, nb * 16, D, dtype=torch.bfloat16)
        kp[:, :pos] = kc
        vp[:, :pos] = vc
        # K pack: [d_blk][s_blk][s_row][d_col] -- prefix scatters per d_blk
        pk = (kp.view(L * KVH, nb, 16, 8, 16).permute(0, 3, 1, 2, 4)
              .contiguous().view(L * KVH, 8, nb * 256))
        # V pack: [s_blk][d_blk][d_col][s_row] -- prefix is one run
        pv = (vp.view(L * KVH, nb, 16, 8, 16).permute(0, 1, 3, 4, 2)
              .contiguous().view(L * KVH, nb * 8 * 256))
        # every row past the live prefix must read exactly ZERO (the scan
        # tiles' tail trick); full zero then prefix scatter
        self.KcP.zero_()
        self.VcP.zero_()
        self.KcP.view(L * KVH, 8, SB * 256)[:, :, : nb * 256].copy_(pk.to(_DEV))
        self.VcP.view(L * KVH, VREGION)[:, : nb * 8 * 256].copy_(pv.to(_DEV))

    def stage_rope(self, cos_cache, sin_cache, ctx: int):
        """The kernel's table form: doubled cos, first-half-negated sin."""
        self.cosT.zero_()
        self.sinT.zero_()
        self.cosT[:ctx] = cos_cache[:ctx].to(torch.float32).to(_DEV)
        ssin = sin_cache[:ctx].to(torch.float32).clone()
        ssin[:, : D // 2] = -ssin[:, : D // 2]
        self.sinT[:ctx] = ssin.to(_DEV)

    # ---- one launch: token at position pos -> argmax token id ----
    def launch_step(self, pos: int) -> int:
        self.partAccP.zero_()
        self.logits.fill_(-1e30)
        tag = (self.launch_no + 1) * 512       # stale seq words never match
        self.launch_no += 1
        stream = torch.npu.current_stream().npu_stream
        _lib.run_megak(
            ctypes.c_void_p(self.ffts),
            *[ctypes.c_void_p(t.data_ptr()) for t in self.ptrs],
            pos, L, tag, 24, ctypes.c_void_p(stream),
        )
        torch.npu.synchronize()
        pm = self.part_max.view(NB * NCH, 16)[:, 0].view(NB, NCH)
        pi = self.part_idx.view(NB * NCH, 16)[:, 0].view(NB, NCH)
        b = int(torch.argmax(pm))
        span, ch = b // NCH, b % NCH
        return int(pi[span, ch]) + span * SPAN

    # ---- outputs, in the HIR's shapes ----
    def read_logits(self) -> torch.Tensor:
        lg = self.logits.view(16, NB, SPANP)[0]
        got = torch.cat([lg[c, :SPAN] for c in range(NB - 1)] +
                        [lg[NB - 1, : VV - (NB - 1) * SPAN]])
        return got.unsqueeze(0).cpu()

    def read_rows(self, pos: int):
        ib, ir = pos // 16, pos % 16
        krows = self.KcP.view(L * KVH, 8, SB, 16, 16)[:, :, ib, ir, :]
        vrows = self.VcP.view(L * KVH, SB, 8, 16, 16)[:, ib, :, :, ir]
        k_rows = krows.reshape(L, KVH, D).unsqueeze(1).cpu()
        v_rows = vrows.reshape(L, KVH, D).unsqueeze(1).cpu()
        return k_rows, v_rows


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
        """One decode step, ONE AscendC mix-kernel launch.

        Caches hold [0, pos), this step's rows land at pos, logits cover
        the live vocabulary.
        """
        rt = _runtime()
        ctx = int(k_caches.shape[1])
        pos = int(pos_ids[0])
        if not 0 <= pos < min(ctx, SPADG):
            raise ValueError(
                f"mega_step: pos {pos} outside the live cache extent "
                f"(ctx_len={ctx}, kernel SPADG={SPADG})"
            )
        tok = int(token_ids[0])
        if not 0 <= tok < VV:
            raise ValueError(f"mega_step: token id {tok} outside the vocab")
        if abs(float(scale.float()) - D ** -0.5) > 1e-3:
            raise ValueError(
                f"mega_step: scale {float(scale.float())} != the kernel's "
                f"fixed {D ** -0.5:.6f} (the kernel does not take a scale arg)"
            )
        rt.set_weights({
            "gamma_in": gamma_in, "w_qkv": w_qkv, "gamma_q": gamma_q,
            "gamma_k": gamma_k, "w_o": w_o, "gamma_post": gamma_post,
            "w_gu": w_gu, "w_down": w_down, "gamma_final": gamma_final,
            "w_lm": w_lm,
        })
        rt.stage_caches(k_caches, v_caches, pos)
        rt.stage_rope(cos_cache, sin_cache, ctx)
        # the HIR embeds via the tied w_lm row; the kernel reads the same row
        rt.xEmb.copy_(w_lm[tok].to(torch.bfloat16))
        next_token = torch.tensor([rt.launch_step(pos)], dtype=torch.int64)
        logits = rt.read_logits()
        k_rows, v_rows = rt.read_rows(pos)
        return logits, next_token, k_rows, v_rows
