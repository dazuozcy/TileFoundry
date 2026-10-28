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

  QWEN3_CKPT=<checkpoint dir> ASCEND_RT_VISIBLE_DEVICES=6 \\
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
KREGION = 8 * SB * 256             # K: [256-pos tile][d_blk][s_blk in tile]..
VREGION = SB * 8 * 256             # elements per V plane: [s_blk][d_blk]..

_DEV = "npu"

# Select an idle healthy device explicitly; device numbers are not an ABI rule.
if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
    raise RuntimeError("select a healthy device with ASCEND_RT_VISIBLE_DEVICES")
_lib = ctypes.CDLL(os.path.join(_HERE, "kernel", os.environ.get("MEGA_LIB", "libmegak.so")))
_lib.run_megak.argtypes = [ctypes.c_void_p] * 36 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
_lib.run_megak.restype = None
# The library, rather than environment guesses, defines the packing contract.
_lib.megak_layout.argtypes = []
_lib.megak_layout.restype = ctypes.c_uint32
_layout = int(_lib.megak_layout())
NC_LM = _layout & 0xffff
COMPACT_A = bool(_layout & (1 << 16))
GU_INTERLEAVE = bool(_layout & (1 << 17))
LM_STREAM = bool(_layout & (1 << 18))
if not 0 < NC_LM <= 128 or NC_LM % 16:
    raise RuntimeError(f"unsupported LM tile in kernel layout: {NC_LM}")
NRD_LM = (SPAN + NC_LM - 1) // NC_LM
_rt = ctypes.CDLL(os.path.join(os.environ["ASCEND_HOME_PATH"], "lib64", "libruntime.so"))
_rt.rtGetC2cCtrlAddr.restype = ctypes.c_int32
_rt.rtGetC2cCtrlAddr.argtypes = [
    ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32)
]


def _pack_w_chunks(w, nGlobal, npad, slab, nc, nrd, nk):
    """Chunk-major fractal pack for the pipelined gemv (kernel v3).
    Layout: [cid][nt][kc][16 k-frac][maxncc n-frac][256] -- one contiguous
    block per gemv chunk so the kernel streams 32..64KB contiguous reads.
    Ragged n-tiles are zero-padded to maxncc (Mmad runs the pad width; the
    fixpipe trims to the real nce columns)."""
    N, K = w.shape
    maxncc = nc // 16
    wp = torch.zeros(npad, K, dtype=torch.bfloat16, device=w.device)
    wp[:N] = w
    blocks = (wp.reshape(npad // 16, 16, K // 16, 16)
               .permute(2, 0, 1, 3).contiguous())          # [kb][nb][16n][16k]
    KB, FB = blocks.shape[0], blocks.shape[1]
    flat = blocks.reshape(KB * FB, 256)
    dev = w.device
    cid = torch.arange(NB, device=dev).view(-1, 1, 1, 1, 1)
    nt = torch.arange(nrd, device=dev).view(1, -1, 1, 1, 1)
    kc = torch.arange(nk, device=dev).view(1, 1, -1, 1, 1)
    jj = torch.arange(16, device=dev).view(1, 1, 1, -1, 1)
    ii = torch.arange(maxncc, device=dev).view(1, 1, 1, 1, -1)
    n0 = cid * slab + nt * nc
    nlim = torch.clamp(cid * slab + slab, max=nGlobal)
    nce = torch.clamp(torch.minimum(n0 + nc, nlim) - n0, min=0)
    ncc = nce // 16
    n0f = n0 // 16
    valid = (ii < ncc) & (n0 < nGlobal)
    src = torch.where(valid, (kc * 16 + jj) * FB + n0f + ii,
                      torch.full_like(ii + kc + jj, KB * FB))
    zero = torch.zeros(1, 256, dtype=torch.bfloat16, device=dev)
    out = torch.cat([flat, zero], dim=0)[src.reshape(-1)]
    return out.contiguous().view(-1)


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
        # Seq slots plus trace/history headroom, matching production Runtime.
        # Zero once; monotonically increasing tags separate subsequent launches.
        self.syncws = torch.zeros(4096 + 48 * 256 * 16 + 24576, dtype=torch.int32, device=_DEV)

        # activation / scratch buffers
        mk = lambda *s, dt=torch.bfloat16: torch.zeros(*s, dtype=dt, device=_DEV)  # noqa: E731
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
        key = tuple((id(t), t._version) for t in w.values())
        if key == self._weight_key:
            return
        cpu = {k: t.detach().to("cpu") for k, t in w.items()}
        # Translate HIR logical weights to the current chunk-major cube stream.
        wqkv = torch.cat([
            _pack_w_chunks(cpu["w_qkv"][li].T[:NQKV], NQKV, NQKV_PAD,
                           176, 96, 2, H // 256) for li in range(L)])
        wo = torch.cat([
            _pack_w_chunks(cpu["w_o"][li][:, :H].T, H, NO_PAD,
                           96, 96, 1, H // 256) for li in range(L)])
        gu = cpu["w_gu"].transpose(1, 2).contiguous()
        if GU_INTERLEAVE:
            gu = torch.stack([gu[:, :II].reshape(L, NB, 256, H),
                              gu[:, II:].reshape(L, NB, 256, H)], dim=2).reshape(L, NGU, H)
        wg = torch.cat([
            _pack_w_chunks(gu[li], NGU, NGU, 512, 128, 4, H // 256)
            for li in range(L)])
        wd = torch.cat([
            _pack_w_chunks(cpu["w_down"][li][:, :H].T, H, ND_PAD,
                           96, 96, 1, II // 256) for li in range(L)])
        wlm = _pack_w_chunks(cpu["w_lm"], VV, NB * SPAN,
                            SPAN, NC_LM, NRD_LM, H // 256)
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
        # Retain inputs so Python cannot recycle ids while the packed copy is cached.
        self._weight_inputs = tuple(w.values())
        self._weight_key = key

    # ---- caches: logical (L, C, KVH, D) -> zero-padded fractal planes ----
    def stage_caches(self, k_caches, v_caches, pos: int):
        # K uses contiguous 256-position tiles (R5+), not global d-major NZ.
        nt = (pos + SLT - 1) // SLT
        nb = (pos + 15) // 16
        kc = k_caches[:, :pos].detach().to("cpu").permute(0, 2, 1, 3).reshape(L * KVH, pos, D)
        vc = v_caches[:, :pos].detach().to("cpu").permute(0, 2, 1, 3).reshape(L * KVH, pos, D)
        kp = torch.zeros(L * KVH, nt * SLT, D, dtype=torch.bfloat16)
        vp = torch.zeros(L * KVH, nb * 16, D, dtype=torch.bfloat16)
        kp[:, :pos] = kc
        vp[:, :pos] = vc
        pk = (kp.view(L * KVH, nt, 16, 16, 8, 16).permute(0, 1, 4, 2, 3, 5)
              .contiguous().view(L * KVH, nt * SLT * D))
        pv = (vp.view(L * KVH, nb, 16, 8, 16).permute(0, 1, 3, 4, 2)
              .contiguous().view(L * KVH, nb * 16 * D))
        self.KcP.zero_()
        self.VcP.zero_()
        self.KcP.view(L * KVH, KREGION)[:, :nt * SLT * D].copy_(pk.to(_DEV))
        self.VcP.view(L * KVH, VREGION)[:, :nb * 16 * D].copy_(pv.to(_DEV))

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
        if (self.launch_no + 1) * 512 + 511 > 0x7fffffff:
            raise OverflowError("launch tag exhausted; recreate the quiescent runtime")
        self.partAccP.zero_()
        self.logits.view(16, NB, SPANP)[0].fill_(-1e30)
        tag = (self.launch_no + 1) * 512       # stale seq words never match
        self.launch_no += 1
        # Raw ctypes launch bypasses torch_npu's queued host tasks.
        launch_stream = torch.npu.current_stream()
        launch_stream.synchronize()
        stream = launch_stream.npu_stream
        _lib.run_megak(
            ctypes.c_void_p(self.ffts),
            *[ctypes.c_void_p(t.data_ptr()) for t in self.ptrs],
            pos, L, tag, 24, ctypes.c_void_p(stream),
        )
        pm = torch.as_tensor(self.part_max.cpu()).numpy().reshape(NB * NCH, 16)[:, 0]
        pi = torch.as_tensor(self.part_idx.cpu()).numpy().reshape(NB * NCH, 16)[:, 0]
        b = int(pm.argmax())
        return int(pi[b]) + (b // NCH) * SPAN

    # ---- outputs, in the HIR's shapes ----
    def read_logits(self) -> torch.Tensor:
        lg = self.logits.view(16, NB, SPANP)[0]
        got = torch.cat([lg[c, :SPAN] for c in range(NB - 1)] +
                        [lg[NB - 1, : VV - (NB - 1) * SPAN]])
        return got.unsqueeze(0).cpu()

    def read_rows(self, pos: int):
        ib, ir = pos // 16, pos % 16
        krows = self.KcP.view(L * KVH, SPADG // SLT, 8, 16, 16, 16)[:, pos // SLT, :, ib % 16, ir, :]
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
        if not 0 <= pos < min(ctx, 40960):
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
