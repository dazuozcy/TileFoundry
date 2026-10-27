"""Greedy decode of Qwen3-1.7B with ONE AscendC mega-kernel launch per token.

The whole decoder -- 28 layers, attention over the full live context, the LM
head -- runs inside a single mix kernel (kernel/mega.cpp, libmegak.so);
Python only feeds the embedding of the next token and reads back the argmax.

  python model_run.py --prompt "..." --max-new-tokens 128
  python model_run.py --bench ctx          # ms/token + tok/s across the range
"""
from __future__ import annotations

import argparse
import ctypes
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
# _NPUDIR = "/home/tilelang/zuochuanuong/TileFoundry-fork/qwen3_mega_npuir"
# for _p in (_HERE, _NPUDIR):
#     if _p not in sys.path:
#         sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: E402

DEV = "npu"
L, H, QH, KVH, D = 28, 2048, 16, 8, 128
II, VV = 6144, 151936
NB, SLT, NSL = 24, 256, 3
STEP, SPADG = 13824, 41472
NQKV, NQKV_PAD = 4096, 4224
NO_PAD, NGU, ND_PAD = 2304, 12288, 2304
SPAN, SPANP, NCH = 6336, 8192, 4
NC_LM = int(os.environ.get("MEGA_LM_TILE", "96"))
assert 0 < NC_LM <= 128 and NC_LM % 16 == 0
NRD_LM = (SPAN + NC_LM - 1) // NC_LM
GU_INTERLEAVE = os.environ.get("MEGA_GU_INTERLEAVE") == "1"
MAXPOS = 40960
ROPE_BASE = 1e6            # Qwen3-1.7B

WDIR = "/home/tilelang/zuochuanuong/weights/qwen3_1_7b"

# ---------------- kernel launch (verified v2 recipe) ----------------
assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") not in (None, "", "0", "1", "2", "3"), \
    "select an explicit device >=4 (chips 0~3 have reported hardware faults)"
lib = ctypes.CDLL(f"{_HERE}/kernel/{os.environ.get('MEGA_LIB', 'libmegak.so')}")
lib.run_megak.argtypes = [ctypes.c_void_p] * 36 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
# New binaries describe their packing. Legacy binaries retain the environment
# override used by the experiment harness.
try:
    _layout_fn = lib.megak_layout
except AttributeError:
    _layout_fn = None
if _layout_fn is not None:
    _layout_fn.argtypes = []
    _layout_fn.restype = ctypes.c_uint32
    _layout = int(_layout_fn())
    _lm_tile, _gu_interleave = _layout & 0xffff, bool(_layout & (1 << 17))
    if "MEGA_LM_TILE" in os.environ:
        assert NC_LM == _lm_tile, "MEGA_LM_TILE does not match the loaded kernel"
    if "MEGA_GU_INTERLEAVE" in os.environ:
        assert GU_INTERLEAVE == _gu_interleave, "GU packing does not match the loaded kernel"
    NC_LM, GU_INTERLEAVE = _lm_tile, _gu_interleave
    NRD_LM = (SPAN + NC_LM - 1) // NC_LM

rt = ctypes.CDLL(os.path.join(os.environ["ASCEND_HOME_PATH"], "lib64", "libruntime.so"))
rt.rtGetC2cCtrlAddr.restype = ctypes.c_int32
rt.rtGetC2cCtrlAddr.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32)]

_dummy = torch.zeros(16, device=DEV)          # init npu context FIRST
_a = ctypes.c_uint64(0); _l = ctypes.c_uint32(0)
assert rt.rtGetC2cCtrlAddr(ctypes.byref(_a), ctypes.byref(_l)) == 0
FFTS = _a.value


def pack_w(w, npad):
    N, K = w.shape
    wp = torch.zeros(npad, K, dtype=torch.bfloat16)
    wp[:N] = w
    return (wp.reshape(npad // 16, 16, K // 16, 16)
            .permute(2, 0, 1, 3).contiguous().view(-1))


def pack_w_chunks(w, nGlobal, npad, slab, nc, nrd, nk):
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


def make_rope_tables():
    """Deterministic FP32 RoPE tables, with scalar FP64 transcendental math."""
    inv = ROPE_BASE ** (-torch.arange(0, D, 2).float() / D)
    fr = torch.arange(SPADG).float()[:, None] * inv[None, :]
    # This CPU backend's parallel float32 cos is not reproducible: even
    # two calls on the same tensor can disagree by 1.5e-4. Keep the FP32
    # frequencies, evaluate scalar libm in FP64 once, then round to FP32.
    angles = fr.numpy().reshape(-1)
    cosine = torch.from_numpy(np.fromiter(
        (math.cos(float(x)) for x in angles), dtype=np.float32,
        count=angles.size).reshape(fr.shape))
    sine = torch.from_numpy(np.fromiter(
        (math.sin(float(x)) for x in angles), dtype=np.float32,
        count=angles.size).reshape(fr.shape))
    return torch.cat([cosine, cosine], dim=1), torch.cat([-sine, sine], dim=1)


class Runtime:
    """Device buffers + the single-launch decode step."""

    def __init__(self):
        assert _w_ts is not None, "call _load_raw() first"
        lay = lambda i, n: _w_ts[f"model.layers.{i}.{n}"]  # noqa: E731
        bf = lambda t: t.bfloat16() if t.dtype != torch.bfloat16 else t  # noqa: E731

        # ---- weights (HF (out, in) -> fractal packs) ----
        Wqkv = torch.stack([torch.cat(
            [lay(i, f"self_attn.{p}_proj.weight") for p in ("q", "k", "v")], dim=0)
            for i in range(L)]).contiguous()                    # (L, NQKV, H)
        Wo = torch.stack([lay(i, "self_attn.o_proj.weight") for i in range(L)]).contiguous()
        Wgu = torch.stack([torch.cat(
            [lay(i, "mlp.gate_proj.weight"), lay(i, "mlp.up_proj.weight")], dim=0)
            for i in range(L)]).contiguous()                    # (L, NGU, H)
        if GU_INTERLEAVE:
            Wgu = torch.stack([Wgu[:, :II].reshape(L, NB, 256, H),
                               Wgu[:, II:].reshape(L, NB, 256, H)], dim=2).reshape(L, NGU, H)
        Wd = torch.stack([lay(i, "mlp.down_proj.weight") for i in range(L)]).contiguous()
        Wlm = _w_ts.get("lm_head.weight", _w_ts["model.embed_tokens.weight"])
        self.WqkvP = torch.cat([pack_w_chunks(Wqkv[i], NQKV, NQKV_PAD, 176, 96, 2, H // 256) for i in range(L)]).npu()
        self.WoP = torch.cat([pack_w_chunks(Wo[i], H, NO_PAD, 96, 96, 1, H // 256) for i in range(L)]).npu()
        self.WguP = torch.cat([pack_w_chunks(Wgu[i], NGU, NGU, 512, 128, 4, H // 256) for i in range(L)]).npu()
        self.WdP = torch.cat([pack_w_chunks(Wd[i], H, ND_PAD, 96, 96, 1, II // 256) for i in range(L)]).npu()
        self.WlmP = pack_w_chunks(Wlm, VV, NB * SPAN, SPAN, NC_LM, NRD_LM, H // 256).npu()
        self.rms1w = torch.stack([lay(i, "input_layernorm.weight") for i in range(L)]).bfloat16().npu()
        self.rms2w = torch.stack([lay(i, "post_attention_layernorm.weight") for i in range(L)]).bfloat16().npu()
        self.rmsfw = bf(_w_ts["model.norm.weight"]).npu()       # (H,)
        self.qnw = torch.stack([lay(i, "self_attn.q_norm.weight") for i in range(L)]).bfloat16().npu()
        self.knw = torch.stack([lay(i, "self_attn.k_norm.weight") for i in range(L)]).bfloat16().npu()
        self.emb = bf(_w_ts["model.embed_tokens.weight"]).npu()

        # rope tables (kernel form: doubled cos, first-half-negated sin)
        cosine, sine = make_rope_tables()
        self.cosT, self.sinT = cosine.npu(), sine.npu()

        # ---- caches / scratch ----
        self.KcP = torch.zeros(L * KVH, 8 * (SPADG // 16) * 256,
                               dtype=torch.bfloat16, device=DEV).view(-1)
        self.VcP = torch.zeros(L * KVH, (SPADG // 16) * 8 * 256,
                               dtype=torch.bfloat16, device=DEV).view(-1)
        self.idx_tab = torch.arange(SPANP, dtype=torch.float32, device=DEV)
        self.syncws = torch.zeros(4096 + 48 * 256 * 16 + 24576, dtype=torch.int32, device=DEV)

        # Define scratch state once rather than depending on allocator contents.
        # This initialization does not replace the per-token clears below.
        mk = lambda *s, dt=torch.bfloat16: torch.zeros(*s, dtype=dt, device=DEV)
        self.xEmb = mk(H)
        self.xn1P = mk(H * 16); self.attnP = mk(H * 16)
        self.xn2P = mk(H * 16); self.xnfP = mk(H * 16); self.hP = mk(II * 16)
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

        self.ptrs = [self.WqkvP, self.WoP, self.WguP, self.WdP, self.WlmP,
                     self.rms1w, self.rms2w, self.rmsfw, self.qnw, self.knw,
                     self.cosT, self.sinT, self.KcP, self.VcP, self.xEmb,
                     self.xn1P, self.qkv_out, self.q2P, self.scores16,
                     self.probs2P, self.part_m, self.part_l, self.partAccP,
                     self.attnP, self.o_out, self.xn2P, self.gu_out, self.hP,
                     self.d_out, self.logits, self.idx_tab, self.part_max,
                     self.part_idx, self.syncws, self.xnfP]
        self.launch_no = 0
        # Hoist the ctypes argument list and hand the kernel the embedding row
        # directly. The raw stream must still be acquired after queued work.
        # Read the embedding of the token
        # directly out of the on-device table instead of H2D-copying 4KB into
        # xEmb every token (that copy cost ~24us/token: launch, not bytes).
        self.pargs = [ctypes.c_void_p(t.data_ptr()) for t in self.ptrs]
        self.xemb_slot = next(i for i, t in enumerate(self.ptrs) if t is self.xEmb)
        self.emb_base = self.emb.data_ptr()
        self.stream = torch.npu.current_stream().npu_stream
        self.pm = self.part_max.view(NB * NCH, 16)[:, 0].view(NB, NCH).cpu()
        self.pi = self.part_idx.view(NB * NCH, 16)[:, 0].view(NB, NCH).cpu()

    def step(self, token: int, sl: int) -> int:
        """One mega launch: token at position sl -> argmax token id."""
        self.partAccP.zero_()
        # R25: only ROW 0 of the logits buffer is read (the kernel's argmax and
        # the harness both index row 0); the other 15 rows are fixpipe dead
        # rows, so masking them was 11.8MB of waste per token.
        self.logits.view(16, NB, SPANP)[0].fill_(-1e30)
        tag = (self.launch_no + 1) * 512
        self.launch_no += 1
        self.pargs[self.xemb_slot] = ctypes.c_void_p(self.emb_base + token * H * 2)
        # ctypes bypasses torch_npu's host task queue. Retire queued clears
        # before the direct launch; reading the raw handle is insufficient.
        launch_stream = torch.npu.current_stream()
        launch_stream.synchronize()
        self.stream = launch_stream.npu_stream
        lib.run_megak(ctypes.c_void_p(FFTS), *self.pargs,
                      sl, L, tag, 24, ctypes.c_void_p(self.stream))
        # R25: one D2H for both partial arrays instead of two synchronous
        # strided copies, and no 12.58MB logits pre-fill (the lm gemv writes
        # every element it will ever be asked about).
        # R25: a strided D2H of the two 96-value column views costs ~70us;
        # copying the whole (replicated) contiguous buffers costs ~8us each and
        # the strided pick is then free.  Keep the reduce in numpy: torch's CPU
        # argmax + view chain costs ~35us of dispatch per token at this size.
        self.pm = np.asarray(self.part_max.cpu()).reshape(NB * NCH, 16)[:, 0]
        self.pi = np.asarray(self.part_idx.cpu()).reshape(NB * NCH, 16)[:, 0]
        b = int(self.pm.argmax())
        return int(self.pi[b]) + (b // NCH) * SPAN


_w_ts = None


def _load_raw():
    """Load raw HF tensors (needed for embed_tokens + layer weights)."""
    import glob
    from safetensors import safe_open
    global _w_ts
    _w_ts = {}
    for f in sorted(glob.glob(os.path.join(WDIR, "model-*.safetensors"))):
        with safe_open(f, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                _w_ts[k] = sf.get_tensor(k)


class Decoder:
    def __init__(self):
        _load_raw()
        self.rt = Runtime()
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(WDIR)

    def generate(self, ids, max_new, echo=False):
        out, times = [], []
        pos = len(ids) - 1
        t0 = time.perf_counter()
        for i, t in enumerate(ids):
            nxt = self.rt.step(int(t), i)
        prefill = time.perf_counter() - t0
        out.append(nxt)
        for _ in range(max_new - 1):
            pos += 1
            if pos >= MAXPOS:
                break
            t0 = time.perf_counter()
            nxt = self.rt.step(out[-1], pos)
            times.append((time.perf_counter() - t0) * 1e3)
            out.append(nxt)
            if echo:
                print(self.tok.decode([nxt]), end="", flush=True)
            if nxt == self.tok.eos_token_id:
                break
        return out, prefill, times


def bench_ctx(dec, lengths=(1, 128, 512, 2048, 8192, 16384, 32768, 40000)):
    print("== context sweep (steady-state decode) ==")
    print(f"{'ctx':>7} {'ms/tok':>9} {'tok/s':>8}")
    ids = dec.tok.encode("The quick brown fox jumps over the lazy dog. " * 512)
    for n in lengths:
        ids_n = ids[:n]
        for i, t in enumerate(ids_n):
            dec.rt.step(int(t), i)
        for _ in range(3):
            dec.rt.step(0, n)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        N = 8
        for _ in range(N):
            dec.rt.step(0, n)
        torch.npu.synchronize()
        dt = (time.perf_counter() - t0) / N
        print(f"{n:7d} {dt*1e3:9.3f} {1/dt:8.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--bench", default="ctx")
    args = ap.parse_args()

    dec = Decoder()
    if args.bench == "ctx":
        bench_ctx(dec)
        return
    if args.prompt is None and args.prompt_file is None:
        args.prompt = "The capital of France is"
    text = args.prompt if args.prompt is not None else open(args.prompt_file).read()
    ids = dec.tok.encode(text)
    assert len(ids) + args.max_new_tokens <= MAXPOS
    out, prefill, times = dec.generate(ids, args.max_new_tokens, echo=True)
    print()
    print(f"[prefill {len(ids)} tok in {prefill*1e3:.1f} ms"
          f" ({len(ids)/prefill:.1f} tok/s)]")
    if times:
        t = torch.tensor(times)
        print(f"[decode {len(times)} tok: mean {t.mean():.2f} ms "
              f"({1000/t.mean():.1f} tok/s), min {t.min():.2f}, max {t.max():.2f}]")


if __name__ == "__main__":
    main()
