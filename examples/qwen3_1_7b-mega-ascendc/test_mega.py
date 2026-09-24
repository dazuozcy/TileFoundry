# Validation harness for the AscendC mega kernel (op-by-op vs torch).
# Gates: lstop=1 (layer-0 pipeline), then more layers / larger sl.
import ctypes, os, sys
import torch
import torch_npu

torch.manual_seed(0)
DIR = os.path.dirname(os.path.abspath(__file__))
DEV = "npu"

# ---------------- constants (mirror mega.cpp) ----------------
L, H, QH, KVH, D = 28, 2048, 16, 8, 128
II, VV = 6144, 151936
NB, KTILE, SLT, NSL = 24, 256, 256, 3
STEP, SPADG = 13824, 41472
NQKV, NQKV_PAD = 4096, 4224
NO_PAD, NGU, ND_PAD = 2304, 12288, 2304
SPAN, SPANP, NRD_LM, NC_LM = 6336, 8192, 66, 96
CH, NCH, NTM = 2048, 4, 54
MAXPOS = 40960

# ---------------- host launch (verified v2 recipe) ----------------
# libmegak.so embeds the device binary + run_megak stub; the bisheng <<<>>
# plugin registers and launches it (rtKernelLaunchWithFlagV2). Manual
# rtKernelLaunch HANGS for mix kernels -- never use it.
assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == "0", \
    "run with ASCEND_RT_VISIBLE_DEVICES=0 (chip 3 has MTE ROB ECC faults)"
lib = ctypes.CDLL(f"{DIR}/kernel/libmegak.so")
lib.run_megak.argtypes = [ctypes.c_void_p] * 36 + [ctypes.c_int] * 4 + [ctypes.c_void_p]

rt = ctypes.CDLL(os.path.join(os.environ["ASCEND_HOME_PATH"], "lib64", "libruntime.so"))
rt.rtGetC2cCtrlAddr.restype = ctypes.c_int32
rt.rtGetC2cCtrlAddr.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32)]

_dummy = torch.zeros(16, device=DEV)          # init npu context FIRST
_a = ctypes.c_uint64(0); _l = ctypes.c_uint32(0)
assert rt.rtGetC2cCtrlAddr(ctypes.byref(_a), ctypes.byref(_l)) == 0
FFTS = _a.value
assert FFTS != 0, "ffts=0 (context not initialized?)"
print(f"libmegak.so loaded, ffts=0x{FFTS:x}")

# ---------------- packing helpers ----------------
def pack_w(w, npad):
    N, K = w.shape
    wp = torch.zeros(npad, K, dtype=torch.bfloat16, device=DEV)
    wp[:N] = w
    return (wp.reshape(npad // 16, 16, K // 16, 16)
            .permute(2, 0, 1, 3).contiguous().view(-1))

def pack_a(x):
    K = x.numel()
    return x.reshape(K // 16, 16).unsqueeze(1) \
             .expand(K // 16, 16, 16).contiguous().view(-1)

def unpack_a(p):
    return p.view(-1, 256)[:, :16].reshape(-1)

# K (pos, D) -> KcP fractals [d-blk][s-blk][16s x 16d]
def pack_kc(k):   # k (SPADG, D) bf16 (zero padded)
    return (k.view(SPADG // 16, 16, 8, 16).permute(2, 0, 1, 3)
            .contiguous().view(-1))

# V (pos, D) -> VcP fractals [s-blk][d-blk][16d x 16s]
def pack_vc(v):
    return (v.view(SPADG // 16, 16, 8, 16).permute(0, 2, 3, 1)
            .contiguous().view(-1))

def mkc(*shape, dtype=torch.bfloat16, val=None):
    if val is None:
        return torch.empty(*shape, dtype=dtype, device=DEV)
    return torch.full(shape, val, dtype=dtype, device=DEV)

# ---------------- weights (random, small scale) ----------------
def rw(*shape, s=0.02):
    return (torch.randn(*shape, device=DEV) * s).to(torch.bfloat16)

Wqkv = rw(L, NQKV, H); Wo = rw(L, H, H)
Wd = rw(L, H, II)
Wlm = rw(VV, H, s=0.01)
rms1w = torch.ones(L, H, dtype=torch.bfloat16, device=DEV) + rw(L, H, s=0.05)
rms2w = torch.ones(L, H, dtype=torch.bfloat16, device=DEV) + rw(L, H, s=0.05)
rmsfw = torch.ones(H, dtype=torch.bfloat16, device=DEV) + rw(H, s=0.05)
qnw = torch.ones(L, D, dtype=torch.bfloat16, device=DEV) + rw(L, D, s=0.05)
knw = torch.ones(L, D, dtype=torch.bfloat16, device=DEV) + rw(L, D, s=0.05)
# mega layout: gu_out = [gate(II); up(II)], Wgu rows = [gate; up]
Wgu_full = torch.cat([rw(L, II, H), rw(L, II, H)], dim=1).contiguous()  # (L, NGU, H)

# rope tables
inv = 1.0 / (10000 ** (torch.arange(0, D, 2, device=DEV).float() / D))
pos = torch.arange(MAXPOS + 512, device=DEV).float()
fr = pos[:, None] * inv[None, :]                              # (P, 64)
cos_full = torch.cat([fr.cos(), fr.cos()], dim=1)             # (P, 128)
sin_full = torch.cat([-fr.sin(), fr.sin()], dim=1)            # first half negated
cosT = torch.zeros(SPADG, D, dtype=torch.float32, device=DEV)
sinT = torch.zeros(SPADG, D, dtype=torch.float32, device=DEV)
cosT[:MAXPOS + 512] = cos_full[:SPADG]
sinT[:MAXPOS + 512] = sin_full[:SPADG]

# ---------------- device buffers ----------------
KVREGION = L * KVH * 8 * (SPADG // 16) * 256      # el per cache
WqkvP = torch.cat([pack_w(Wqkv[i], NQKV_PAD) for i in range(L)])
WoP = torch.cat([pack_w(Wo[i], NO_PAD) for i in range(L)])
WguP = torch.cat([pack_w(Wgu_full[i], NGU) for i in range(L)])
WdP = torch.cat([pack_w(Wd[i], ND_PAD) for i in range(L)])
WlmP = pack_w(Wlm, NB * SPAN)
KcP = torch.zeros(L * KVH, 8 * (SPADG // 16) * 256, dtype=torch.bfloat16, device=DEV).view(-1)
VcP = torch.zeros(L * KVH, (SPADG // 16) * 8 * 256, dtype=torch.bfloat16, device=DEV).view(-1)
idx_tab = torch.arange(SPANP, dtype=torch.float32, device=DEV)
# sync workspace: [0,1536) SyncAll GM ws + [4096,196608) seq slots
# (rows 0..23 AIV, 24..47 AIC; SEQ_WANT(row,r) = 4096 + (row*256+r)*16).
# Host-zeroed ONCE -- SyncAll counters are monotonic and stale seq words
# never match (tag += 512/launch starting at 512).
SYNCWS = 4096 + 48 * 256 * 16
syncws = torch.zeros(SYNCWS, dtype=torch.int32, device=DEV)

xEmb = rw(H)
xn1P = mkc(H * 16); attnP = mkc(H * 16); xn2P = mkc(H * 16); xnfP = mkc(H * 16)
qkv_out = mkc(16, NQKV_PAD, dtype=torch.float32)
o_out = mkc(16, NO_PAD, dtype=torch.float32)
gu_out = mkc(16, NGU, dtype=torch.float32)
d_out = mkc(16, ND_PAD, dtype=torch.float32)
q2P = mkc(NB, 8 * 256)
scores16 = mkc(NB, 16, STEP, dtype=torch.float32)
probs2P = mkc(NB, NTM * 16 * 256)
part_m = mkc(48 * 16, dtype=torch.float32)
part_l = mkc(48 * 16, dtype=torch.float32)
partAccP = mkc(NB, 16, D, dtype=torch.float32)
logits = mkc(16, NB * SPANP, dtype=torch.float32)
part_max = mkc(NB * NCH * 16, dtype=torch.float32)
part_idx = mkc(NB * NCH * 16, dtype=torch.float32)

PTRS = [WqkvP, WoP, WguP, WdP, WlmP, rms1w, rms2w, rmsfw, qnw, knw, cosT, sinT,
        KcP, VcP, xEmb, xn1P, qkv_out, q2P, scores16, probs2P, part_m, part_l,
        partAccP, attnP, o_out, xn2P, gu_out, None, d_out, logits, idx_tab,
        part_max, part_idx, syncws, xnfP]
hP = mkc(II * 16)
PTRS[27] = hP

_launch_no = [0]

def launch(sl, lstop):
    partAccP.zero_()
    logits.fill_(-1e30)
    # +512 per launch, starting at 512: stale/zeroed seq words never match
    # (round values 512..735; launch 0 round 0 must NOT collide with the
    # zero-initialized slots!)
    tag = (_launch_no[0] + 1) * 512
    _launch_no[0] += 1
    stream = torch.npu.current_stream().npu_stream
    lib.run_megak(ctypes.c_void_p(FFTS), *[ctypes.c_void_p(t.data_ptr()) for t in PTRS],
                  sl, lstop, tag, 24, ctypes.c_void_p(stream))

# ---------------- torch reference (kernel numerics) ----------------
def rms_ref(x, w):
    xf = x.float()
    r = torch.rsqrt(xf.pow(2).mean() + 1e-6)
    xn = (xf * r).to(torch.bfloat16)
    return (xn.float() * w.float()).to(torch.bfloat16)

def rope_ref(x):   # x (n,128) bf16 -> bf16
    xf = x.float()
    rot = torch.cat([xf[:, D // 2:], xf[:, :D // 2]], dim=1)
    return (xf * cos_full[:1].expand(x.shape[0], D) +
            rot * sin_full[:1].expand(x.shape[0], D)).to(torch.bfloat16)

def ref_run(sl, lstop, Kref, Vref):
    """Kref/Vref: (L, KVH, pos, D) bf16 caches; position sl is the new token.
    Returns dict of intermediates for layers [0, lstop)."""
    outs = {}
    x = xEmb.clone()
    for li in range(lstop):
        xn1 = rms_ref(x, rms1w[li])
        qkv = xn1.float() @ Wqkv[li].float().t()            # (NQKV,) f32
        q = qkv[:H].view(QH, D); k = qkv[H:3 * H // 2].view(KVH, D)
        v = qkv[3 * H // 2:].view(KVH, D)
        qn = torch.stack([rms_ref(q[h], qnw[li]) for h in range(QH)])
        kn = torch.stack([rms_ref(k[j], knw[li]) for j in range(KVH)])
        pos_row = sl  # all heads at position sl
        fr_p = pos_row * inv
        cosr = torch.cat([fr_p.cos(), fr_p.cos()])
        sinr = torch.cat([-fr_p.sin(), fr_p.sin()])
        def rp(t):
            tf = t.float()
            rot = torch.cat([tf[:, D // 2:], tf[:, :D // 2]], dim=1)
            return (tf * cosr + rot * sinr).to(torch.bfloat16)
        qr = rp(qn); kr = rp(kn)
        Kref[li, :, sl] = kr; Vref[li, :, sl] = v.to(torch.bfloat16)
        scale = D ** -0.5
        # attention per kv head (2 q heads each)
        attn = torch.zeros(H, dtype=torch.float32, device=DEV)
        for j in range(KVH):
            Kj = Kref[li, j, :sl + 1].float()               # (P, D)
            sc = (qr[2 * j:2 * j + 2].float() @ Kj.t()) * scale   # (2, P)
            p = torch.softmax(sc, dim=-1)
            av = p.to(torch.bfloat16).float() @ Vref[li, j, :sl + 1].float()
            attn[2 * j * D:(2 * j + 2) * D] = av.reshape(-1)
        attnb = attn.to(torch.bfloat16)
        o = attnb.float() @ Wo[li].float().t()
        x = (x.float() + o).to(torch.bfloat16)
        xn2 = rms_ref(x, rms2w[li])
        gu = xn2.float() @ Wgu_full[li].float().t()
        g, u = gu[:II], gu[II:]
        h = (g.float() * torch.sigmoid(g.float()) * u.float()).to(torch.bfloat16)
        dn = h.float() @ Wd[li].float().t()
        x = (x.float() + dn).to(torch.bfloat16)
        if li == lstop - 1:
            outs.update(xn1=xn1, qkv=qkv, q2=qr, k2=kr, v2=v, attn=attnb,
                        o=o, xn2=xn2, h=h, d=dn, x=x)
    xnf = rms_ref(outs["x"], rmsfw)
    lg = xnf.float() @ Wlm.float().t()
    outs["xnf"] = xnf
    outs["logits"] = lg
    return outs

# ---------------- comparison helpers ----------------
def cmp(name, got, ref, atol=2e-2, rtol=2e-2):
    got = got.float().cpu(); ref = ref.float().cpu()
    ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
    md = (got - ref).abs().max().item() if got.numel() else 0.0
    print(f"  {name:14s} {'PASS' if ok else 'FAIL'}  maxdiff={md:.4g}")
    return ok

# ---------------- gate 1: lstop=1, sl=0 ----------------
def gate(sl, lstop, prefill=None):
    print(f"== gate sl={sl} lstop={lstop} ==")
    # depth/span-scaled tolerances: bf16 landings + per-slice LSE vs global
    # f32 softmax drift accumulates with layers AND prefix length (validated
    # tightly at sl<=100; large-span gates rely on logits+argmax agreement)
    tl = 2e-2 * (1 + (lstop - 1) * 0.35) * (1 + sl / 800.0)
    t5 = 5e-2 * (1 + (lstop - 1) * 0.35) * (1 + sl / 800.0)
    Kref = torch.zeros(lstop, KVH, MAXPOS + 512, D, dtype=torch.bfloat16, device=DEV)
    Vref = torch.zeros(lstop, KVH, MAXPOS + 512, D, dtype=torch.bfloat16, device=DEV)
    # re-zero the device caches: the kernel relies on score tails reading
    # EXACTLY 0 beyond the live prefix (probs tail * V-zero = 0)
    KcP.zero_(); VcP.zero_()
    if prefill is not None:
        pf = (torch.randn(lstop, KVH, sl, D, device=DEV) * 0.05).to(torch.bfloat16)
        vf = (torch.randn(lstop, KVH, sl, D, device=DEV) * 0.05).to(torch.bfloat16)
        Kref[:, :, :sl] = pf; Vref[:, :, :sl] = vf
        # host-prefill the device caches in fractal layout
        for li in range(lstop):
            for j in range(KVH):
                kk = torch.zeros(SPADG, D, dtype=torch.bfloat16, device=DEV)
                vv = torch.zeros(SPADG, D, dtype=torch.bfloat16, device=DEV)
                kk[:sl] = pf[li, j]; vv[:sl] = vf[li, j]
                KcP[(li * KVH + j) * (8 * (SPADG // 16) * 256):
                    (li * KVH + j + 1) * (8 * (SPADG // 16) * 256)] = pack_kc(kk)
                VcP[(li * KVH + j) * ((SPADG // 16) * 8 * 256):
                    (li * KVH + j + 1) * ((SPADG // 16) * 8 * 256)] = pack_vc(vv)
    ref = ref_run(sl, lstop, Kref, Vref)
    launch(sl, lstop)
    torch.npu.synchronize()
    ok = True
    ok &= cmp("xn1P", unpack_a(xn1P), ref["xn1"], atol=tl)
    ok &= cmp("qkv_out", qkv_out[0, :NQKV], ref["qkv"], atol=t5)
    # q2P rows 0,1 for each cid's kv
    kv0 = 0
    q2r = q2P.view(NB, 8, 16, 16)[kv0 + 0, :, 0, :].reshape(-1)  # cid=0: kv0 heads 0,1
    ok &= cmp("q2P row0", q2r, ref["q2"][0], atol=tl)
    q2r1 = q2P.view(NB, 8, 16, 16)[0, :, 1, :].reshape(-1)
    ok &= cmp("q2P row1", q2r1, ref["q2"][1], atol=tl)
    # K/V append of the LAST layer's kv-0 block (block per [li*KVH + kv]);
    # the fresh row sits at s-blk sl//16, s-row sl%16
    lkv = (lstop - 1) * KVH
    ib, ir = sl // 16, sl % 16
    kapp = KcP[lkv * 8 * (SPADG // 16) * 256:(lkv + 1) * 8 * (SPADG // 16) * 256] \
        .view(8, SPADG // 16, 16, 16)[:, ib, ir, :].reshape(-1)
    ok &= cmp("K append", kapp, ref["k2"][0], atol=tl)
    vapp = VcP[lkv * (SPADG // 16) * 8 * 256:(lkv + 1) * (SPADG // 16) * 8 * 256] \
        .view(SPADG // 16, 8, 16, 16)[ib, :, :, ir].reshape(-1)
    ok &= cmp("V append", vapp, ref["v2"][0], atol=tl)
    # scores16 for cid=0 (kv0, sid0): rows 0,1, tile 0, first `thirds` cols
    thirds = ((sl + 1 + NSL - 1) // NSL + SLT - 1) // SLT * SLT
    e0 = min(sl + 1, thirds)
    sc = scores16.view(NB, 16, STEP)
    sr0 = sc[0, 0, 0:e0]; sr1 = sc[0, 1, 0:e0]
    scale = D ** -0.5
    rsc0 = (ref["q2"][0].float() @ Kref[lstop - 1, 0, :sl + 1].float().t()) * scale
    rsc1 = (ref["q2"][1].float() @ Kref[lstop - 1, 0, :sl + 1].float().t()) * scale
    rsc0 = rsc0[:e0]; rsc1 = rsc1[:e0]      # slice 0's portion only
    if e0 > 0:
        ok &= cmp("scores r0", sr0 * scale, rsc0, atol=t5)   # kernel: unscaled
        ok &= cmp("scores r1", sr1 * scale, rsc1, atol=t5)
    # part_m/l for (kv0, h0, sid0) -- kernel stores UNNORMALIZED per-slice
    # exp sums (LSE combine divides later), so refs must match that form
    pm = part_m.view(48, 16)
    pl = part_l.view(48, 16)
    p0 = torch.softmax(rsc0, -1); p1 = torch.softmax(rsc1, -1)
    l0ref = torch.exp(rsc0 - rsc0.max()).sum()
    ok &= cmp("part_m", pm[0, :1], rsc0.max().reshape(1), atol=t5)
    ok &= cmp("part_l", pl[0, :1], l0ref.reshape(1), atol=t5)
    # partAcc slot (kv0, sid0) rows 0,1: probs = bf16(exp(sc - m_slice))
    pa = partAccP.view(NB, 16, D)
    pe0 = torch.exp(rsc0 - rsc0.max()).to(torch.bfloat16).float()
    pe1 = torch.exp(rsc1 - rsc1.max()).to(torch.bfloat16).float()
    racc0 = pe0 @ Vref[lstop - 1, 0, :e0].float()
    racc1 = pe1 @ Vref[lstop - 1, 0, :e0].float()
    if e0 > 0:
        ok &= cmp("partAcc r0", pa[0, 0], racc0, atol=t5)
        ok &= cmp("partAcc r1", pa[0, 1], racc1, atol=t5)
    ok &= cmp("attnP", unpack_a(attnP), ref["attn"], atol=t5)
    ok &= cmp("o_out", o_out[0, :H], ref["o"], atol=t5)
    ok &= cmp("xn2P", unpack_a(xn2P), ref["xn2"], atol=tl)
    gu_ref = ref["xn2"].float() @ Wgu_full[lstop - 1].float().t()
    ok &= cmp("gu_out", gu_out[0], gu_ref, atol=t5)
    ok &= cmp("hP", unpack_a(hP).view(-1)[:II], ref["h"], atol=t5)
    ok &= cmp("d_out", d_out[0, :H], ref["d"], atol=t5)
    ok &= cmp("xnfP", unpack_a(xnfP), ref["xnf"], atol=tl)
    # logits: gather row 0 as (NB, SPANP) -> vocab
    lg = logits.view(16, NB, SPANP)[0]
    got_lg = torch.cat([lg[c, :SPAN] for c in range(NB - 1)] +
                       [lg[NB - 1, :VV - (NB - 1) * SPAN]])
    ok &= cmp("logits", got_lg, ref["logits"], atol=1e-1 * (1 + (lstop - 1) * 0.35))
    # argmax
    gmax = part_max.view(NB * NCH, 16)[:, 0].view(NB, NCH)
    gidx = part_idx.view(NB * NCH, 16)[:, 0].view(NB, NCH)
    tok = int(gmax.max(1).values.argmax())
    row = int(gmax[tok].argmax()); gi = int(gidx[tok, row])
    gidx_abs = tok * SPAN + gi   # kernel ci is already the span-global index
    gtok = int(ref["logits"].argmax())
    near = bool(ref["logits"][gidx_abs] >= ref["logits"].max() - 0.2)
    print(f"  argmax kernel={gidx_abs} ref={gtok} "
          f"{'PASS' if gidx_abs == gtok else ('NEAR-TIE' if near else 'FAIL')}")
    ok &= (gidx_abs == gtok or near)
    return ok

if __name__ == "__main__":
    sl = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    lstop = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    ok = gate(sl, lstop, prefill=(sl > 0))
    print("GATE", "PASS" if ok else "FAIL")
