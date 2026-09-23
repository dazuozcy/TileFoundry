"""Standalone verification + timing harness for the mega kernel.

Synthetic mode (default): random weights, compares the kernel's logits and
greedy token against an fp32 torch reference that lands bf16 at the same
points the kernel does.  Env: L, SL (list, comma-separated), S, V, I, TIME.

Real mode (REAL=1): the published checkpoint, teacher-forced or free-run
greedy vs the same reference.  Env: REAL=1, PROMPT, PLEN, NDEC, TF=1, S.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401

import torch
import torch_npu  # noqa: F401
import tilelang

from mega import mega, mega_shapes  # noqa: E402

L = int(os.environ.get("L", "2"))
H = 2048
QH = 16
KVH = 8
D = 128
II = int(os.environ.get("I", "6144"))
VV = int(os.environ.get("V", "151936"))
S = int(os.environ.get("S", "40960"))
DEV = "npu"
WDIR = "/home/tilelang/zuochuanuong/weights/qwen3_1_7b"

SH = mega_shapes(l=L, h=H, qh=QH, kvh=KVH, d=D, ii=II, vv=VV, s=S)
NQKV = SH["nqkv"]
NQKV_PAD = SH["nqkv_pad"]
STEP, SPADG, NSL, NB = SH["step"], SH["spadg"], SH["nsl"], SH["nb"]
SPAN, SPANP, NCH, CH = SH["span"], SH["spanp"], SH["nch"], SH["ch"]




def staged(*args):
    """torch.zeros on the host, then moved to the device: direct device
    allocations are the known stale-read hazard on this stack.  The last
    positional argument is the dtype."""
    shape, dtype = args[:-1], args[-1]
    return torch.zeros(*shape, dtype=dtype).to(DEV)


def staged_full(shape, value, dtype=torch.float32):
    return torch.full(shape, value, dtype=dtype).to(DEV)


def rope_tables(maxpos, d, base=1e6):
    inv = base ** (-torch.arange(0, d, 2).float() / d)
    t = torch.arange(maxpos).float()
    freqs = torch.outer(t, inv)          # (maxpos, d/2)
    cos = freqs.cos()
    sin = freqs.sin()
    cos_tab = torch.cat([cos, cos], dim=1)   # doubled
    sin_tab = torch.cat([-sin, sin], dim=1)  # signed
    return cos_tab, sin_tab


def rmsnorm(xb, w, eps=1e-6):
    xf = xb.float()
    rrms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    # Qwen3RMSNorm: the normalised activation lands bf16 *before* the scale
    return ((xf * rrms).bfloat16().float() * w.float()).bfloat16()


def rope(xb, cos_row, sin_row):
    xf = xb.float()
    sw = torch.cat([xf[..., xb.shape[-1] // 2:], xf[..., : xb.shape[-1] // 2]], dim=-1)
    return (xf * cos_row + sw * sin_row).bfloat16()


def ref_decode(w, Kc, Vc, token, sl, cos_tab, sin_tab, scale):
    """One decode step: caches hold [0, sl), this step appends at sl.

    Lands bf16 where the kernel lands bf16; GEMV chains stay fp32.
    """
    x = w["Wlm"][token].bfloat16().clone()
    n = sl + 1
    for li in range(L):
        xn = rmsnorm(x, w["rms1w"][li])
        qkv = xn.float() @ w["Wqkv"][li].float().T
        q, k, v = qkv.split([QH * D, KVH * D, KVH * D], dim=-1)
        q = q.view(QH, D)
        k = k.view(KVH, D)
        v = v.view(KVH, D)
        qf = q.float()
        kf = k.float()
        qf = (qf * torch.rsqrt(qf.pow(2).mean(-1, keepdim=True) + 1e-6)).bfloat16().float() * w["qnw"][li].float()
        kf = (kf * torch.rsqrt(kf.pow(2).mean(-1, keepdim=True) + 1e-6)).bfloat16().float() * w["knw"][li].float()
        q = rope(qf, cos_tab[sl], sin_tab[sl])
        k = rope(kf, cos_tab[sl], sin_tab[sl])
        Kc[li, :, sl] = k
        Vc[li, :, sl] = v.bfloat16()
        attn = torch.zeros(QH, D)
        for qh in range(QH):
            kv = qh // 2
            s = (Kc[li, kv, :n].float() @ q[qh].float()) * scale
            p = torch.softmax(s, dim=0).bfloat16()
            attn[qh] = (p.float() @ Vc[li, kv, :n].float())
        attn = attn.bfloat16()
        o = (attn.view(-1).float() @ w["Wo"][li].float().T)
        x = (x.float() + o).bfloat16()
        xn2 = rmsnorm(x, w["rms2w"][li])
        gu = xn2.float() @ w["Wgu"][li].float().T
        g, u = gu.split([II, II], dim=-1)
        h = (torch.nn.functional.silu(g) * u).bfloat16()
        d = (h.float() @ w["Wd"][li].float().T)
        x = (x.float() + d).bfloat16()
    xf = rmsnorm(x, w["rmsfw"])
    logits = xf.float() @ w["Wlm"].float().T
    return logits, x, Kc[li, :, sl].clone(), Vc[li, :, sl].clone()


def load_real():
    from safetensors import safe_open
    ts = {}
    import glob
    for f in sorted(glob.glob(os.path.join(WDIR, "model-*.safetensors"))):
        with safe_open(f, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                ts[k] = sf.get_tensor(k)
    w = dict(H=H, QH=QH, KVH=KVH, D=D, II=II, L=L)
    w["Wqkv"] = torch.stack(
        [torch.cat([ts[f"model.layers.{i}.self_attn.{p}_proj.weight"]
                    for p in ("q", "k", "v")], dim=0) for i in range(L)])
    w["Wo"] = torch.stack([ts[f"model.layers.{i}.self_attn.o_proj.weight"] for i in range(L)])
    w["Wgu"] = torch.stack(
        [torch.cat([ts[f"model.layers.{i}.mlp.{p}_proj.weight"]
                    for p in ("gate", "up")], dim=0) for i in range(L)])
    w["Wd"] = torch.stack([ts[f"model.layers.{i}.mlp.down_proj.weight"] for i in range(L)])
    w["rms1w"] = torch.stack([ts[f"model.layers.{i}.input_layernorm.weight"] for i in range(L)])
    w["rms2w"] = torch.stack([ts[f"model.layers.{i}.post_attention_layernorm.weight"] for i in range(L)])
    w["rmsfw"] = ts["model.norm.weight"]
    w["qnw"] = torch.stack([ts[f"model.layers.{i}.self_attn.q_norm.weight"] for i in range(L)])
    w["knw"] = torch.stack([ts[f"model.layers.{i}.self_attn.k_norm.weight"] for i in range(L)])
    w["Wlm"] = ts.get("lm_head.weight", ts["model.embed_tokens.weight"])
    for k in ("Wqkv", "Wo", "Wgu", "Wd", "rms1w", "rms2w", "rmsfw", "qnw", "knw", "Wlm"):
        w[k] = w[k].contiguous().bfloat16()
    return w


def pack_qkv(w):
    """(L, 4096, H) -> (L*4224, H) with zero pad rows, kernel row-major."""
    out = torch.zeros(L, NQKV_PAD, H, dtype=torch.bfloat16)
    out[:, :NQKV] = w["Wqkv"]
    return out.view(L * NQKV_PAD, H)


def seq_test(w, kern, args, Kc_d, Vc_d, part_max, part_idx, logits,
             cos_tab, sin_tab, scale):
    """Teacher-forced sequential decode over real text: the kernel builds its
    own on-manifold KV history step by step, and at checkpoint positions the
    logits are compared against the fp32 reference conditioned on the kernel's
    own cache (random-value histories amplify bf16 noise chaotically at full
    context and are not a meaningful correctness signal there)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(WDIR)
    text = open(os.path.join(WDIR, "README.md")).read()
    ids = tok.encode(text)
    while len(ids) < S:
        ids = ids + ids
    toks = ids[:S]
    checkpoints = set(int(x) for x in os.environ.get("CKPT", "0,1,2,63,1023,4095,13823,27647,32767,40957,40958,40959").split(","))
    checkpoints = {c for c in checkpoints if 0 <= c < S}
    print(f"seq test: {S} teacher-forced steps, checkpoints at {sorted(checkpoints)}")
    ok_all = True
    for pos in range(S):
        kern(*args, toks[pos], pos, scale)
        if pos in checkpoints:
            torch.npu.synchronize()
            pm = part_max.view(NB, NCH).cpu()
            pi = part_idx.view(NB, NCH).cpu()
            flat = int(torch.argmax(pm))
            b, c = flat // NCH, flat % NCH
            tok_k = int(pi[b, c]) + b * SPAN
            lg_k = logits.cpu()[0, :VV]
            Kc_ref = Kc_d.view(L, KVH, SPADG, D).cpu()
            Vc_ref = Vc_d.view(L, KVH, SPADG, D).cpu()
            lg_r, _, _, _ = ref_decode(w, Kc_ref, Vc_ref, toks[pos], pos,
                                       cos_tab, sin_tab, scale)
            tok_r = int(lg_r.argmax())
            rel = ((lg_k - lg_r).abs().max() / lg_r.abs().max()).item()
            ok = tok_k == tok_r and rel < 0.05
            ok_all = ok_all and ok
            extra = ""
            if not ok and os.environ.get("DBG"):
                xk = args[14].cpu()[0]
                am = int(xk.float().abs().argmax())
                extra = (f" x[{am}]={xk[am].item():.4g} x_absmax={xk.float().abs().max().item():.4g}"
                         f" o_absmax={args[20].cpu().abs().max().item():.4g}"
                         f" d_absmax={args[24].cpu().abs().max().item():.4g}")
                print(f"  POISON pos={pos}: {extra}", flush=True)
                # per-layer divergence via the k rows the kernel appended at pos
                kvk = Kc_d.view(L, KVH, SPADG, D).cpu()
                Kc_p = Kc_d.view(L, KVH, SPADG, D).cpu().clone()
                Kc_p[:, :, pos + 1:] = 0
                # ref chain over the kernel's own history, recording k rows
                def rmsn(xb, g):
                    xf = xb.float()
                    rr = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
                    return ((xf * rr).bfloat16().float() * g.float()).bfloat16()
                xw = w["Wlm"][toks[pos]].bfloat16().clone()
                n = pos + 1
                per = []
                for li in range(L):
                    xn = rmsn(xw, w["rms1w"][li])
                    qkv = xn.float() @ w["Wqkv"][li].float().T
                    q, k, v = qkv.split([QH * D, KVH * D, KVH * D], -1)
                    q = q.view(QH, D); k = k.view(KVH, D); v = v.view(KVH, D)
                    qf = (q.float() * torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + 1e-6)).bfloat16().float() * w["qnw"][li].float()
                    kf = (k.float() * torch.rsqrt(k.float().pow(2).mean(-1, keepdim=True) + 1e-6)).bfloat16().float() * w["knw"][li].float()
                    qr = rope(qf, cos_tab[pos], sin_tab[pos])
                    kr = rope(kf, cos_tab[pos], sin_tab[pos])
                    Kc_p[li, :, pos] = kr
                    Vc_p = Vc_d.view(L, KVH, SPADG, D).cpu().clone()
                    Vc_p[li, :, pos] = v.bfloat16()
                    e = (kvk[li, 0, pos].float() - kr[0].float()).abs().max().item()
                    per.append(e)
                    attn = torch.zeros(QH, D)
                    for qh in range(QH):
                        kvh = qh // 2
                        s = (Kc_p[li, kvh, :n].float() @ qr[qh].float()) * scale
                        p = torch.softmax(s, 0).bfloat16()
                        attn[qh] = p.float() @ Vc_p[li, kvh, :n].float()
                    attn = attn.bfloat16()
                    o = attn.view(-1).float() @ w["Wo"][li].float().T
                    x1 = (xw.float() + o).bfloat16()
                    xn2r = rmsn(x1, w["rms2w"][li])
                    gu = xn2r.float() @ w["Wgu"][li].float().T
                    g_, u_ = gu.split([II, II], -1)
                    h_ = (torch.nn.functional.silu(g_) * u_).bfloat16()
                    dd = h_.float() @ w["Wd"][li].float().T
                    xw = (x1.float() + dd).bfloat16()
                print("  k_per_layer=" + ",".join(f"{e:.3f}" for e in per), flush=True)
            print(f"  pos={pos:6d} token={toks[pos]:6d} kernel={tok_k:6d} ref={tok_r:6d} "
                  f"{'MATCH' if ok else 'FAIL'} logits rel={rel:.4f}", flush=True)
    print("PASS" if ok_all else "FAIL")


def main():
    torch.manual_seed(0)
    real = os.environ.get("REAL", "0") == "1"
    sls = [int(x) for x in os.environ.get("SL", "0,1,33,255,1023").split(",") if x != ""]
    sls = [s for s in sls if s < S]
    scale = float(torch.tensor(D ** -0.5, dtype=torch.bfloat16).float())

    if real:
        w = load_real()
    else:
        w = dict(H=H, QH=QH, KVH=KVH, D=D, II=II, L=L)
        w["Wqkv"] = (torch.randn(L, NQKV, H) * 0.02).bfloat16()
        w["Wo"] = (torch.randn(L, H, H) * 0.02).bfloat16()
        w["Wgu"] = (torch.randn(L, 2 * II, H) * 0.02).bfloat16()
        w["Wd"] = (torch.randn(L, H, II) * 0.02).bfloat16()
        w["rms1w"] = (1 + 0.02 * torch.randn(L, H)).bfloat16()
        w["rms2w"] = (1 + 0.02 * torch.randn(L, H)).bfloat16()
        w["rmsfw"] = (1 + 0.02 * torch.randn(H)).bfloat16()
        w["qnw"] = (1 + 0.02 * torch.randn(L, D)).bfloat16()
        w["knw"] = (1 + 0.02 * torch.randn(L, D)).bfloat16()
        w["Wlm"] = (torch.randn(VV, H) * 0.02).bfloat16()

    cos_tab, sin_tab = rope_tables(SPADG, D)
    kern = mega(**SH)
    # ---------------- persistent device buffers ----------------
    # every buffer is staged through the host: direct device allocations are
    # the known stale-read hazard on this stack
    Kc_d = staged(L * KVH * SPADG, D, torch.bfloat16)
    Vc_d = staged(L * KVH * SPADG, D, torch.bfloat16)
    x = staged(1, H, torch.bfloat16)
    xn1 = staged(1, H, torch.bfloat16)
    qkv_out = staged(1, NQKV_PAD, torch.float32)
    q_buf = staged(QH, D, torch.bfloat16)
    attn_out = staged(QH, D, torch.bfloat16)
    attn_flat = attn_out.view(1, H)
    o_out = staged(1, H, torch.float32)
    xn2 = staged(1, H, torch.bfloat16)
    gu_out = staged(1, 2 * II, torch.float32)
    h_buf = staged(1, II, torch.bfloat16)
    d_out = staged(1, H, torch.float32)
    scores = staged(NB * 2 * STEP, 1, torch.float32)
    probs = staged(NB * 2, STEP, torch.bfloat16)
    part_m = staged_full((KVH * 2 * NSL,), -1e30, torch.float32)
    part_l = staged(KVH * 2 * NSL, torch.float32)
    part_acc = staged(KVH * 2 * NSL, D, torch.float32)
    acc16 = staged(KVH * 2 * SH["kpad"], D, torch.bfloat16)
    wrow16 = staged(KVH * 2, SH["kpad"], torch.bfloat16)
    num = staged(KVH * 2, D, torch.float32)
    den = staged(KVH * 2, torch.float32)
    logits = staged_full((1, NB * SPANP), float("-inf"), torch.float32)
    idx_tab = torch.arange(SPANP, dtype=torch.int32).to(DEV)
    part_max = staged(NB * NCH, torch.float32)
    part_idx = staged(NB * NCH, torch.float32)

    args = [
        pack_qkv(w).to(DEV),
        w["Wo"].to(DEV).view(L * H, H),
        w["Wgu"].to(DEV).view(L * 2 * II, H),
        w["Wd"].to(DEV).view(L * H, II),
        w["rms1w"].to(DEV), w["rms2w"].to(DEV), w["rmsfw"].to(DEV),
        w["qnw"].to(DEV), w["knw"].to(DEV),
        w["Wlm"].to(DEV), Kc_d, Vc_d,
        cos_tab.to(DEV), sin_tab.to(DEV),
        x, xn1, qkv_out, q_buf, attn_out, attn_flat, o_out, xn2,
        gu_out, h_buf, d_out, scores, probs, part_m, part_l, part_acc,
        acc16, wrow16, num, den, logits, idx_tab, part_max, part_idx,
    ]

    def run(token, sl):
        kern(*args, token, sl, scale)
        torch.npu.synchronize()
        pm = part_max.view(NB, NCH).cpu()
        pi = part_idx.view(NB, NCH).cpu()
        flat = int(torch.argmax(pm))
        b, c = flat // NCH, flat % NCH
        tok = int(pi[b, c]) + b * SPAN
        return logits.cpu(), tok

    if os.environ.get("SEQ") == "1":
        seq_test(w, kern, args, Kc_d, Vc_d, part_max, part_idx, logits,
                 cos_tab, sin_tab, scale)
        return

    print(f"mega test: L={L} S={S} SPADG={SPADG} STEP={STEP} V={VV} I={II} "
          f"real={real} scale={scale!r}")
    ok_all = True
    t_first = None
    for sl in sls:
        token = int(torch.randint(0, VV, (1,)))
        # reference state
        Kc_ref = torch.zeros(L, KVH, SPADG, D).bfloat16()
        Vc_ref = torch.zeros(L, KVH, SPADG, D).bfloat16()
        if sl > 0:
            Kc_ref[:, :, :sl] = (torch.randn(L, KVH, sl, D) * 0.5).bfloat16()
            Vc_ref[:, :, :sl] = (torch.randn(L, KVH, sl, D) * 0.5).bfloat16()
        # fill device caches
        Kc_d.zero_(); Vc_d.zero_()
        for kv in range(KVH):
            for li in range(L):
                r0 = (li * KVH + kv) * SPADG
                if sl > 0:
                    Kc_d[r0:r0 + sl] = Kc_ref[li, kv, :sl].to(DEV)
                    Vc_d[r0:r0 + sl] = Vc_ref[li, kv, :sl].to(DEV)
        t0 = time.time()
        lg_k, tok_k = run(token, sl)
        if t_first is None:
            t_first = time.time() - t0
        lg_r, x_r, k_row_r, v_row_r = ref_decode(
            w, Kc_ref, Vc_ref, token, sl, cos_tab, sin_tab, scale)
        tok_r = int(lg_r.argmax())
        err = (lg_k[0, :VV] - lg_r).abs().max().item()
        scale_r = lg_r.abs().max().item()
        rel = err / max(scale_r, 1e-9)
        # k/v rows the kernel wrote (layer 0 and last)
        kview = Kc_d.view(L, KVH, SPADG, D).cpu()
        vview = Vc_d.view(L, KVH, SPADG, D).cpu()
        kerr = max(
            (kview[0, 0, sl].float() - Kc_ref[0, 0, sl].float()).abs().max().item(),
            (kview[L - 1, 0, sl].float() - Kc_ref[L - 1, 0, sl].float()).abs().max().item(),
        )
        top2 = torch.topk(lg_r, 2).values
        gap = (top2[0] - top2[1]).item() / max(scale_r, 1e-9)
        mark = "MATCH" if tok_k == tok_r else f"MISMATCH(gap={gap:.4f})"
        if tok_k != tok_r or rel > 0.05:
            ok_all = False
        extra = ""
        if os.environ.get("DBG"):
            kv_k = kview
            per = [(kv_k[li, 0, sl].float() - Kc_ref[li, 0, sl].float()).abs().max().item()
                   for li in range(L)]
            extra = " k_per_layer=" + ",".join(f"{e:.3f}" for e in per)
        print(f"  sl={sl:6d} token={token:6d} kernel={tok_k:6d} ref={tok_r:6d} {mark} "
              f"logits rel={rel:.4f} k_row_err={kerr:.4f}{extra}")

    print("PASS" if ok_all else "FAIL")

    if os.environ.get("TIME"):
        for _ in range(3):
            kern(*args, 100, min(1000, S - 1), scale)
        torch.npu.synchronize()
        for sl in [int(x) for x in os.environ["TIME"].split(",")]:
            if sl >= S:
                continue
            for _ in range(3):
                kern(*args, 100, sl, scale)
            torch.npu.synchronize()
            t0 = time.time()
            N = 20
            for _ in range(N):
                kern(*args, 100, sl, scale)
            torch.npu.synchronize()
            dt = (time.time() - t0) / N
            print(f"  timing sl={sl}: {dt * 1e3:.2f} ms/token = {1 / dt:.1f} tok/s")


if __name__ == "__main__":
    main()
