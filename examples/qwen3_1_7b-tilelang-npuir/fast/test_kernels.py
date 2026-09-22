"""Torch statements of every kernel in kernels.py -- the interface they must match.

Each test builds the kernel at production dimensions (or a smaller shape with
the same structure where marked), runs it on random data, and compares against
a torch spelling of the same math, including the rounding points the published
model itself has.
"""
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kernels as K  # noqa: E402

DT = torch.bfloat16
F32 = torch.float32
DEV = "npu"
EPS = 1e-6


def randn(*shape, dtype=DT):
    return torch.randn(*shape, dtype=torch.float32, device=DEV).to(dtype)


def close(a, b, *, atol=2e-2, rtol=2e-2):
    return torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)


def rel_close(a, b, rel=0.01):
    """Equal within *rel* of the largest value in play -- the honest bound
    for an fp16 GEMV whose error random-walks with the term scale."""
    a, b = a.float(), b.float()
    scale = max(b.abs().max().item(), a.abs().max().item(), 1e-6)
    return bool(((a - b).abs() <= rel * scale + 1e-3).all())


def bf16_close(a, b, ulps=3.0):
    """Equal up to a few bf16 rounding steps of the operands' magnitude.

    Cancellation-prone ops (rope, residuals) can move a small output by a
    full ulp of its large inputs, so the bound is relative to the *largest*
    value in play, not to each element.
    """
    a, b = a.float(), b.float()
    scale = max(b.abs().max().item(), a.abs().max().item(), 1e-6)
    return bool(((a - b).abs() <= ulps * 2.0 / 256.0 * scale).all())


def test_embed():
    V, H = 151936, 2048
    tbl = randn(V, H)
    ids = torch.tensor([12345], dtype=torch.int32, device=DEV)
    out = torch.zeros(H, dtype=DT, device=DEV)
    K.embed(V, H)(tbl, ids, out)
    assert torch.equal(out, tbl[12345]), "embed"
    print("embed ok")


def _norm_ref(x_bf, gamma, eps):
    x32 = x_bf.float()
    r = torch.rsqrt(x32.pow(2).mean() + eps)
    return ((x32 * r).bfloat16().float() * gamma.float()).bfloat16()


def test_rms_norm():
    H = 2048
    x = randn(H)
    g = randn(H)
    out = torch.zeros(H, dtype=torch.float16, device=DEV)
    K.rms_norm(H, EPS)(x, g, out)
    ref = _norm_ref(x, g, EPS)
    assert bf16_close(out, ref), f"rms_norm {(out.float()-ref.float()).abs().max()}"
    print("rms_norm ok")


def test_resid_rms_norm():
    H, SK = 2048, 4
    a = randn(H)
    p = torch.randn(SK, H, dtype=F32, device=DEV)
    g = randn(H)
    hout = torch.zeros(H, dtype=DT, device=DEV)
    xn = torch.zeros(H, dtype=torch.float16, device=DEV)
    K.resid_rms_norm(H, SK, EPS)(a, p, g, hout, xn)
    so = p.sum(0).bfloat16()
    h1 = (a.float() + so.float()).bfloat16()
    assert bf16_close(hout, h1), "resid h1"
    xn_ref = _norm_ref(h1, g, EPS)
    assert bf16_close(xn, xn_ref), f"resid norm {(xn.float()-xn_ref.float()).abs().max()}"
    print("resid_rms_norm ok")


def test_gemv():
    # production qkv shape: K=2048, N=4096
    Kis, N, BN, BK, SK, NC = 2048, 4096, 128, 256, 4, 48
    x = randn(Kis).to(torch.float16)
    w = randn(N, Kis).to(torch.float16)
    p = torch.zeros(SK * N, dtype=F32, device=DEV)
    K.gemv(Kis, N, BN, BK, SK, NC)(x, w, p)
    p = p.view(SK, N)
    ref = (w.float() @ x.float()).reshape(N)
    got = p.sum(0)
    assert close(got, ref, atol=0.5, rtol=5e-2), f"gemv {(got-ref).abs().max()}"
    print("gemv ok")


def _rope_ref(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d].float(), x[..., d:].float()
    o1 = x1 * cos[..., :d].float() + (-x2) * sin[..., :d].float()
    o2 = x2 * cos[..., d:].float() + x1 * sin[..., d:].float()
    return torch.cat([o1, o2], -1).bfloat16()


def test_qk_rope_cache():
    HQ, HKV, D, SK = 16, 8, 128, 2
    QN, KN = HQ * D, HKV * D
    CAP, MP = 512, 1024
    G = HQ // HKV
    p = torch.randn(SK, QN + 2 * KN, dtype=F32, device=DEV)
    gq, gk = randn(D), randn(D)
    pos_i = 77
    inv = 1.0 / (1e6 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    ph = torch.outer(torch.arange(MP, dtype=torch.float32), inv)
    ph = torch.cat((ph, ph), -1)
    cos = ph.cos().bfloat16().npu()
    sin = ph.sin().bfloat16().npu()
    pos = torch.tensor([pos_i], dtype=torch.int32, device=DEV)
    kc = torch.zeros(CAP, KN, dtype=DT, device=DEV)
    vc = torch.zeros(CAP, KN, dtype=DT, device=DEV)
    qp = torch.zeros(HKV, 16, D, dtype=DT, device=DEV)

    K.qk_rope_cache(HQ, HKV, D, MP, CAP, SK, EPS)(
        p, gq, gk, cos, sin, pos, pos, kc, vc, qp
    )


    full = p.sum(0)
    # q heads
    for h in range(HQ):
        v = full[h * D:(h + 1) * D]
        n = (v.float() * torch.rsqrt(v.float().pow(2).mean() + EPS)).bfloat16()
        n = (n.float() * gq.float()).bfloat16()
        r = _rope_ref(n, cos[pos_i], sin[pos_i])
        assert bf16_close(qp[h // G, h % G], r), f"rope q head {h}"
    # k heads
    for h in range(HKV):
        v = full[QN + h * D: QN + (h + 1) * D]
        n = (v.float() * torch.rsqrt(v.float().pow(2).mean() + EPS)).bfloat16()
        n = (n.float() * gk.float()).bfloat16()
        r = _rope_ref(n, cos[pos_i], sin[pos_i])
        assert bf16_close(kc[pos_i, h * D:(h + 1) * D], r), f"rope k head {h}"
    # v heads
    for h in range(HKV):
        v = full[QN + KN + h * D: QN + KN + (h + 1) * D]
        assert bf16_close(vc[pos_i, h * D:(h + 1) * D], v.bfloat16()), f"v head {h}"
    print("qk_rope_cache ok")


def test_attn_partial():
    HQ, HKV, D, CAP, SS = 16, 8, 128, 512, 128
    G = HQ // HKV
    NS = CAP // SS
    scale = D ** -0.5
    pos_i = 300  # context length
    qp = torch.zeros(HKV, 16, D, dtype=DT, device=DEV)
    q = randn(HQ, D)
    for h in range(HQ):
        qp[h // G, h % G] = q[h]
    kc = randn(CAP, HKV * D)
    vc = randn(CAP, HKV * D)
    pos = torch.tensor([pos_i], dtype=torch.int32, device=DEV)
    op = torch.zeros(HQ, NS, D, dtype=F32, device=DEV)
    mp = torch.zeros(NS * HQ, dtype=F32, device=DEV)
    lp = torch.zeros(NS * HQ, dtype=F32, device=DEV)
    K.attn_partial(HQ, HKV, D, CAP, SS, 48, scale)(qp, kc, vc, pos, op, mp, lp)
    mp = mp.view(NS, HQ)
    lp = lp.view(NS, HQ)

    for h in range(HQ):
        kvh = h // G
        k = kc[: pos_i + 1, kvh * D:(kvh + 1) * D].float()
        v = vc[: pos_i + 1, kvh * D:(kvh + 1) * D].float()
        sc = (k @ q[h].float()) * scale
        mx = sc.max()
        p = torch.exp(sc - mx)
        out = (p.bfloat16().float() @ v) / p.sum()
        s = mp[:, h].max()
        assert close(s, mx, atol=1e-3), f"attn mx {h}: {s} vs {mx}"
        # the merge the consumer performs: den from the split partials
        e = torch.exp(mp[:, h] - s)
        den = (lp[:, h] * e).sum()
        assert close(den, p.sum(), atol=1e-1), f"attn lp {h}"
        acc = torch.zeros(D, dtype=F32, device=DEV)
        for s_i in range(NS):
            acc += op[h, s_i] * e[s_i]
        assert close(acc / den, out, atol=2e-2), f"attn out {h}"
    print("attn_partial ok")


def test_gemv_attn_combine():
    HQ, D, NS = 16, 128, 5
    N, BN, BK, SK, NC = 2048, 64, 128, 2, 48
    KDIM = HQ * D
    # head-major partials, as attn_partial writes them
    op = (torch.randn(HQ, NS, D, dtype=F32, device=DEV) * 0.5)
    mp = torch.randn(NS, HQ, dtype=F32, device=DEV).reshape(-1)
    lp = (torch.rand(NS, HQ, dtype=F32, device=DEV) * 10).reshape(-1)
    w = randn(N, KDIM).to(torch.float16)
    p = torch.zeros(SK * N, dtype=F32, device=DEV)
    xg = torch.zeros(KDIM, dtype=torch.float16, device=DEV)
    K.gemv_attn_combine(HQ, D, N, BN, BK, SK, NS, NC)(op, mp, lp, w, xg, p)
    torch.npu.synchronize()
    # reference: merge per head, then the projection
    x = torch.zeros(KDIM, dtype=F32, device=DEV)
    for h in range(HQ):
        m = mp.view(NS, HQ)[:, h]
        l = lp.view(NS, HQ)[:, h]
        o = op[h]
        s = m.max()
        e = torch.exp(m - s)
        den = (l * e).sum()
        num = (o * e[:, None]).sum(0)
        x[h * D:(h + 1) * D] = num / den
    ref = w.float() @ x.half().float()
    got = p.view(SK, N).sum(0)
    assert rel_close(got, ref), f"attn combine {(got-ref).abs().max()}"
    print("gemv_attn_combine ok")


def test_gemv_silu():
    I, N, BN, BK, SK, SKG, NC = 6144, 2048, 64, 128, 4, 2, 48
    gu = torch.randn(SKG * 2 * I, dtype=F32, device=DEV)
    w = randn(N, I).to(torch.float16)
    p = torch.zeros(SK * N, dtype=F32, device=DEV)
    xg = torch.zeros(I, dtype=torch.float16, device=DEV)
    K.gemv_silu(I, N, BN, BK, SK, SKG, NC)(gu, w, xg, p)
    torch.npu.synchronize()
    g = gu.view(SKG, 2 * I)[:, :I].sum(0).bfloat16().float()
    u = gu.view(SKG, 2 * I)[:, I:].sum(0).bfloat16().float()
    x = (torch.nn.functional.silu(g).bfloat16().float() * u).bfloat16()
    # the kernel lands the activation in f16, which rounds finer than the
    # reference's bf16: allow a bf16-ulp of slack against the bf16 ref
    dxa = (xg.float() - x.float()).abs()
    scale = x.float().abs().max().item()
    assert dxa.max() <= 3.0 * 2.0 / 256.0 * scale, \
        f"silu activation {dxa.max()} scale {scale}"
    ref = w.float() @ xg.float()
    got = p.view(SK, N).sum(0)
    assert rel_close(got, ref), f"silu {(got-ref).abs().max()}"
    print("gemv_silu ok")


def test_lm_head():
    Kis, N, BN, BK, NC = 2048, 151936, 128, 256, 48
    x = randn(Kis).to(torch.float16)
    w = randn(N, Kis).to(torch.float16)
    o = torch.zeros(N, dtype=F32, device=DEV)
    kern, NB = K.lm_head(Kis, N, BN, BK, NC)
    kern(x, w, o)
    ref = w.float() @ x.float()
    assert rel_close(o, ref, rel=0.02), f"head logits {(o-ref).abs().max()}"
    # argmax stage over the padded (1, PADV) logits buffer
    ak, NB2, PADV = K.argmax_stage(N, 2048, 48)
    # host-staged buffers, as the engine allocates them: device-side
    # allocations make the argmax's loads read stale slots on some runs
    L = torch.full((1, PADV), -1.0e30, dtype=F32).to(DEV)
    L[0, :N] = o
    bv = torch.zeros(1, NB2, dtype=F32).to(DEV)
    bi = torch.zeros(NB2, 1, dtype=torch.int32).to(DEV)
    fl = torch.zeros(NB2, 2048, dtype=F32).to(DEV)
    ak(L, bv, bi, fl)
    am = int(ref.argmax())
    assert abs(bv.max().item() - ref.max().item()) < 0.2, f"argmax bv {bv.max()} vs {ref.max()}"
    assert int(bi[int(bv.flatten().argmax()), 0]) == am, f"argmax {bi[int(bv.flatten().argmax()), 0]} vs {am}"
    print("lm_head ok")


def test_sample_step():
    NB, NSTEPS = 1187, 64
    PAD = (NB + 7) // 8 * 8
    bv = torch.full((PAD,), -1.0e30, dtype=F32, device=DEV)
    bv[:NB] = torch.randn(NB, dtype=F32, device=DEV)
    bi = torch.arange(NB, dtype=torch.int32, device=DEV)
    inp = torch.arange(1000, 1000 + NSTEPS, dtype=torch.int32, device=DEV)
    plen = torch.tensor([5], dtype=torch.int32, device=DEV)
    ids = torch.zeros(1, dtype=torch.int32, device=DEV)
    pos = torch.tensor([3], dtype=torch.int32, device=DEV)
    sam = torch.zeros(NSTEPS, dtype=torch.int32, device=DEV)
    K.sample_step(NB, PAD, NSTEPS)(bv, bi, inp, plen, ids, pos, sam)
    am = int(bv.flatten().argmax())
    assert int(sam[3]) == am, f"sample best {sam[3]} vs {am}"
    assert int(ids[0]) == int(inp[4]), "prompt feed"
    assert int(pos[0]) == 4, "pos advance"
    K.sample_step(NB, PAD, NSTEPS)(bv, bi, inp, plen, ids, pos, sam)
    assert int(sam[4]) == am and int(ids[0]) == am and int(pos[0]) == 5, "step 2"
    print("sample_step ok")


if __name__ == "__main__":
    tests = [
        test_embed, test_rms_norm, test_resid_rms_norm, test_gemv,
        test_qk_rope_cache, test_attn_partial, test_gemv_attn_combine,
        test_gemv_silu, test_lm_head, test_sample_step,
    ]
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for t in tests:
        if only is None or only in t.__name__:
            t()
    print("all kernel tests passed")
