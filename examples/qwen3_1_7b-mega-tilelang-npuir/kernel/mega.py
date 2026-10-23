"""Qwen3-1.7B decode megakernel -- ONE tilelang MIX launch per token.

Expert-mode tilelang-npuir (target="npuir"): the whole decode step -- embed
gather, 28 layers (rmsnorm -> fused QKV GEMV -> q/k-norm + RoPE + KV append
-> sequence-parallel GQA flash attention with device-side combine -> o GEMV
-> residual + rmsnorm -> gate/up GEMV -> silu*up -> down GEMV -> residual),
final norm, tied lm_head GEMV and per-block argmax partials -- runs inside a
single MIX kernel launch: 24 blocks, each 1 aic (cube) + 2 aiv (vector)
subblocks, communicating through GM buffers and FFTS flags.

Decomposition (mirrored by the authored HIR in ../model.py):

* every GEMV splits its output axis N over the 24 blocks in *contiguous
  slabs* of the padded packed weight (qkv: 24 x 176 = 4224 >= 4096 in two
  88-wide cube tiles; o/down: 24 x 88 = 2112 >= 2048; gate_up: 24 x 512 =
  12288 exact in 64-wide tiles; lm_head: 24 x 6336 = 152064 >= 151936 in
  48-wide tiles). The K axis walks in KTILE=512 chunks through L1.
* attention is sharded (kv-head x seq-slice): block (cid) owns kv = cid % 8,
  sid = cid // 8, i.e. an (8 x 3) mesh over 24 blocks; the live prefix is
  split into nsl dynamic thirds (SLT-tile aligned, from the runtime sl) and
  each block scans its third in SLT=128 score tiles, produces
  (m, l, acc) partials, and the sid==0 leader block merges its group's three
  slices by log-sum-exp (the new token's own row is appended to the cache
  first and scanned inside its owner slice).  Dynamic thirds (vs fixed
  sid*step windows) keep all 24 blocks streaming cache rows at every context
  length; the split degenerates to the fixed windows at full context.
* MTE2 pipelining (this kernel is memory-bound; see ../OPTLOG.md): every
  weight GEMV (qkv/gu/down/lm_head; o stays single for L1 budget) keeps two
  L1 B-tile buffers (plus two A-tile buffers) and software-pipelines the kt
  loop -- tile kt+1 streams while tile kt is in the cube, with a cross-round
  prefetch of the next nt round's first tile at kt == nkt-1.  The first B
  tile is prefetched BEFORE the activation flag wait (weights do not depend
  on the handshake).  The attention K/V tiles use the same double-buffer
  pattern (K/V rows are only safe after fb+9/fb+5, so their prologue stays
  behind the wait).  CAVEAT: gemm operands must be whole L1 buffers -- a
  sliced region like l1_q[1:2, :] is silently miscompiled by the backend.
* the elementwise stages (norms, RoPE, softmax, silu, residuals) run
  replicated/redundantly on the vector side where a block-wide reduce or a
  shared residual carrier forces it; the stages that decompose cleanly are
  split across the two aivs (P1-1): S2's two q heads by subid (k/v stays
  redundant), S3's per-head softmax, S4's per-head combine, S5'b's silu by
  ii halves, and the argmax chunks.  The block-level sync_block rendezvous
  (a cube wait passes only after BOTH aivs set the flag -- spike-verified)
  covers each split half with no extra barrier.

Sync protocol (spike-validated, see the qwen3_mega_tl NOTES this was built
from): INTER FFTS flags live on per-core-type planes, so every mixed
cube<->aiv rendezvous is a plane barrier + INTRA relay.  An aiv->cube
sync_block_wait passes only after BOTH of the block's aiv subblocks set the
flag (verified by spike: a lone-set variant hangs), which is what makes the
P1-1 splits safe; a cube->aiv set broadcasts to both aivs.  Per layer, with
ping-pong flag base fb = 16*(li%2):

  fb+1  cube-plane  INTER barrier (PIPE_FIX), shared in order by
        qkv / o / gu / down
  fb+2  vector-plane INTER barrier (PIPE_MTE3), shared in order by
        partials / attn_flat
  relays: fb+4 qkv, fb+6 partials, fb+10 o, fb+12 gu, fb+14 d
          (cube set FIX -> own aivs wait MTE2)
          fb+9 append, fb+15 attn_flat (aiv set MTE3 -> own cube wait MTE2)
  same-block: fb+0 xn1, fb+3 scores, fb+5 probs, fb+11 xn2, fb+13 h-ready
  final (fbf = 16*(l%2)): fbf+0 xn_f (same-block), fbf+1 logits (same-block;
        each aiv reads only its own cube's span)

P2-narrowcast (2026-10-22): the two rendezvous whose data was NOT truly
all-block lost their plane barrier -- fb+2 after the KV append (the owner
appends row sl itself and every block rewrites its own group's q_buf rows
identically, so the same-block fb+9 relay suffices) and fb+1 after gemm2
(the leader only reads its own kv group's partials; each cube now relays
fb+6 right after its own gemm2, and the vector-plane fb+2 at S4 entry --
whose arrivals are gated per block by fb+6 and ordered on PIPE_MTE3 after
the aiv's own S3 stores -- is the carrier that aggregates the group).
The surviving six plane barriers are genuine all-block exchanges: qkv_out
(S2 reads a sliding window of producer slabs), o_out/gu_out/d_out (every
aiv reads the full activation row) and attn_flat (every cube reads all
eight leaders' combine output).

The greedy argmax stores per-(block, chunk) partials; the launcher combines
them on the host: token = part_idx[b, c] + b*SPAN (part_idx already carries
the chunk offset c*CH -- adding it again was the bug in the original harness).

Landings (rounding discipline): rmsnorm lands bf16 before gamma; q/k-norm
land bf16 before gamma; rope lands bf16; scores stay f32 and take the scale
after the dot; probabilities land bf16 into the PV product; per-slice PV
accumulates f32; the combine's acc/weights land bf16 for the gemm3 product;
attention output lands bf16; GEMV outputs stay f32 in GM; silu lands bf16
before the up-product; the product lands bf16; residuals land bf16.
"""
import os

import tilelang
import tilelang.language as T


@tilelang.jit(target="npuir", pass_configs={"npuir.enable_auto_multi_buffer": False})
def mega(l: T.int32, h: T.int32, qh: T.int32, kvh: T.int32, d: T.int32,
         ii: T.int32, vv: T.int32, slt: T.int32, nb: T.int32, nsl: T.int32,
         step: T.int32, spadg: T.int32, nqkv: T.int32, nqkv_pad: T.int32,
         ktile: T.int32, nc_qkv: T.int32, nrd_qkv: T.int32, nc_o: T.int32,
         nrd_o: T.int32, nc_gu: T.int32, nrd_gu: T.int32, nc_d: T.int32,
         nrd_d: T.int32, span: T.int32, nrd_lm: T.int32, nc_lm: T.int32,
         spanp: T.int32, kpad: T.int32, ch: T.int32, nch: T.int32):
    @T.prim_func
    def main(
        Wqkv: T.Tensor((l * nqkv_pad, h), "bfloat16"),
        Wo: T.Tensor((l * h, h), "bfloat16"),
        Wgu: T.Tensor((l * 2 * ii, h), "bfloat16"),
        Wd: T.Tensor((l * h, ii), "bfloat16"),
        rms1w: T.Tensor((l, h), "bfloat16"),
        rms2w: T.Tensor((l, h), "bfloat16"),
        rmsfw: T.Tensor((h,), "bfloat16"),
        qnw: T.Tensor((l, d), "bfloat16"),
        knw: T.Tensor((l, d), "bfloat16"),
        Wlm: T.Tensor((vv, h), "bfloat16"),
        Kc: T.Tensor((l * kvh * spadg, d), "bfloat16"),
        Vc: T.Tensor((l * kvh * spadg, d), "bfloat16"),
        cos_tab: T.Tensor((spadg, d), "float32"),
        sin_tab: T.Tensor((spadg, d), "float32"),
        x: T.Tensor((1, h), "bfloat16"),
        xn1: T.Tensor((1, h), "bfloat16"),
        qkv_out: T.Tensor((1, nqkv_pad), "float32"),
        q_buf: T.Tensor((qh, d), "bfloat16"),
        attn_out: T.Tensor((qh, d), "bfloat16"),
        attn_flat: T.Tensor((1, h), "bfloat16"),
        o_out: T.Tensor((1, h), "float32"),
        xn2: T.Tensor((1, h), "bfloat16"),
        gu_out: T.Tensor((1, 2 * ii), "float32"),
        h_buf: T.Tensor((1, ii), "bfloat16"),
        d_out: T.Tensor((1, h), "float32"),
        scores: T.Tensor((nb * 2 * step, 1), "float32"),
        probs: T.Tensor((nb * 2, step), "bfloat16"),
        part_m: T.Tensor((kvh * 2 * nsl,), "float32"),
        part_l: T.Tensor((kvh * 2 * nsl,), "float32"),
        part_acc: T.Tensor((kvh * 2 * nsl, d), "float32"),
        acc16: T.Tensor((kvh * 2 * kpad, d), "bfloat16"),
        wrow16: T.Tensor((kvh * 2, kpad), "bfloat16"),
        num: T.Tensor((kvh * 2, d), "float32"),
        den: T.Tensor((kvh * 2,), "float32"),
        logits: T.Tensor((1, nb * spanp), "float32"),
        idx_tab: T.Tensor((spanp,), "int32"),
        part_max: T.Tensor((nb * nch,), "float32"),
        part_idx: T.Tensor((nb * nch,), "float32"),
        token: T.int32,
        sl: T.int32,
        scale: T.float32,
    ):
        with T.Kernel(nb, is_npu=True) as (cid, subid):
            kv = cid % kvh
            sid = cid // kvh
            # dynamic even split of the live prefix across the nsl slices
            # (rounded to SLT tiles): with fixed sid*step windows only the
            # slices below the live length stream cache rows, so at mid
            # context 8 (or 16) of 24 blocks did all the cache traffic and
            # aggregate HBM rate fell to ~2/3; thirds makes every block
            # stream at every context length.  Rounded so s0 stays tile
            # aligned, which also keeps the tail tile inside the padded cache.
            thirds = T.ceildiv(T.ceildiv(sl + 1, nsl), slt) * slt
            s0 = sid * thirds
            e0 = T.min(sl + 1, s0 + thirds)
            sblk = e0 - s0
            eps = 1e-6
            inv_h = 1.0 / h
            inv_d = 1.0 / d
            neg_inf = -1e30
            zero = 0.0
            big = 1e30
            two = 2.0
            kbase = qh * d
            vbase = (qh + kvh) * d
            slab_qkv = nrd_qkv * nc_qkv
            slab_gu = nrd_gu * nc_gu

            # ---------- aiv: embed gather (once, before the layer loop) -----
            # the token's own row of the tied embedding table becomes the
            # residual carrier x; S5' of each layer rewrites it in place
            with T.Scope("Vector"):
                ub_x0 = T.alloc_ub((1, 1, h), "bfloat16")
                T.copy(Wlm[token, 0:h], ub_x0[0, 0, 0:h])
                T.copy(ub_x0[0, 0, 0:h], x[0, 0:h])

            for li in T.serial(l):
                fb = 16 * (li % 2)  # ids fb+0..15 (see module docstring)
                kc0 = (li * kvh + kv) * spadg
                wq0 = li * nqkv_pad
                wo0 = li * h
                wg0 = li * 2 * ii

                # ---------- aiv S1: embed gather + rmsnorm1 ----------
                with T.Scope("Vector"):
                    T.pipe_barrier("PIPE_ALL")
                    ub_x = T.alloc_ub((1, 1, h), "bfloat16")
                    ub_f = T.alloc_ub((1, 1, h), "float32")
                    ub_g = T.alloc_ub((1, 1, h), "float32")
                    ub_w = T.alloc_ub((1, 1, h), "bfloat16")
                    ub_wf = T.alloc_ub((1, 1, h), "float32")
                    ub_o = T.alloc_ub((1, 1, h), "bfloat16")
                    red = T.alloc_ub((1, 1, 1), "float32")
                    T.copy(x[0, 0:h], ub_x[0, 0, 0:h])
                    T.vcast(ub_x, ub_f)
                    T.vmul(ub_f, ub_f, ub_g)
                    T.reduce(ub_g, red, dims=[2], reduce_mode="sum", clear=True)
                    T.vmul(red, inv_h, red)
                    T.vadd(red, eps, red)
                    T.vsqrt(red, red)
                    T.vdiv(ub_f, red, ub_f)
                    T.copy(rms1w[li, 0:h], ub_w[0, 0, 0:h])
                    T.vcast(ub_w, ub_wf)
                    # land bf16 before the learned scale (Qwen3RMSNorm)
                    T.vcast(ub_f, ub_o)
                    T.vcast(ub_o, ub_f)
                    T.vmul(ub_f, ub_wf, ub_f)
                    T.vcast(ub_f, ub_o)
                    T.copy(ub_o[0, 0, 0:h], xn1[0, 0:h])
                    with T.rs("PIPE_MTE3"):
                        T.sync_block_set(fb + 0)

                # ---------- cube: qkv gemv (contiguous 176-wide slabs) ----------
                # double-buffered B tiles (l1_b0/l1_b1) + A tiles (l1_a0/l1_a1):
                # the kt loop prefetches tile kt+1 (and round nt+1's first tile)
                # while the cube consumes tile kt, so MTE2 stays busy across the
                # gemm boundaries.  The first B tile is prefetched BEFORE the
                # fb+0 wait: weights do not depend on the xn1 handshake.
                # (buffer parity requires h//ktile even -- holds for all configs)
                with T.Scope("Cube"):
                    l1_a0 = T.alloc_L1((1, ktile), "bfloat16")
                    l1_a1 = T.alloc_L1((1, ktile), "bfloat16")
                    l1_b0 = T.alloc_L1((nc_qkv, ktile), "bfloat16")
                    l1_b1 = T.alloc_L1((nc_qkv, ktile), "bfloat16")
                    l0_c = T.alloc_L0C((1, nc_qkv), "float32")
                    pf0 = T.min(nc_qkv, nqkv - cid * slab_qkv)
                    T.copy(
                        Wqkv[wq0 + cid * slab_qkv : wq0 + cid * slab_qkv + pf0, 0:ktile],
                        l1_b0[0:pf0, 0:ktile],
                    )
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(fb + 0)
                    T.copy(xn1[0:1, 0:ktile], l1_a0[0:1, 0:ktile])
                    for nt in T.serial(nrd_qkv):
                        n0 = cid * slab_qkv + nt * nc_qkv
                        if n0 < nqkv:
                            tail = T.min(nc_qkv, nqkv - n0)
                            for kt in T.serial(h // ktile):
                                k0 = kt * ktile
                                if kt % 2 == 0:
                                    if kt + 1 < h // ktile:
                                        T.copy(xn1[0:1, k0 + ktile : k0 + 2 * ktile], l1_a1[0:1, 0:ktile])
                                        T.copy(
                                            Wqkv[wq0 + n0 : wq0 + n0 + tail, k0 + ktile : k0 + 2 * ktile],
                                            l1_b1[0:tail, 0:ktile],
                                        )
                                    else:
                                        n0n = n0 + nc_qkv
                                        if nt + 1 < nrd_qkv:
                                            if n0n < nqkv:
                                                tailn = T.min(nc_qkv, nqkv - n0n)
                                                T.copy(xn1[0:1, 0:ktile], l1_a1[0:1, 0:ktile])
                                                T.copy(
                                                    Wqkv[wq0 + n0n : wq0 + n0n + tailn, 0:ktile],
                                                    l1_b1[0:tailn, 0:ktile],
                                                )
                                    T.gemm(
                                        l1_a0, l1_b0, l0_c,
                                        initC=(kt == 0), b_transpose=True,
                                        size=[1, ktile, tail],
                                    )
                                else:
                                    if kt + 1 < h // ktile:
                                        T.copy(xn1[0:1, k0 + ktile : k0 + 2 * ktile], l1_a0[0:1, 0:ktile])
                                        T.copy(
                                            Wqkv[wq0 + n0 : wq0 + n0 + tail, k0 + ktile : k0 + 2 * ktile],
                                            l1_b0[0:tail, 0:ktile],
                                        )
                                    else:
                                        n0n = n0 + nc_qkv
                                        if nt + 1 < nrd_qkv:
                                            if n0n < nqkv:
                                                tailn = T.min(nc_qkv, nqkv - n0n)
                                                T.copy(xn1[0:1, 0:ktile], l1_a0[0:1, 0:ktile])
                                                T.copy(
                                                    Wqkv[wq0 + n0n : wq0 + n0n + tailn, 0:ktile],
                                                    l1_b0[0:tailn, 0:ktile],
                                                )
                                    T.gemm(
                                        l1_a1, l1_b1, l0_c,
                                        initC=(kt == 0), b_transpose=True,
                                        size=[1, ktile, tail],
                                    )
                            with T.rs("PIPE_FIX"):
                                T.copy(l0_c[0:1, 0:tail], qkv_out[0:1, n0 : n0 + tail])
                    with T.rs("PIPE_FIX"):
                        T.block_barrier(fb + 1)
                        T.sync_block_set(fb + 4)

                # ---------- aiv S2: qk-norm + rope + KV append ----------
                # P1-1: the two q heads of the kv group split across the two
                # aivs (aiv0 -> qh even, aiv1 -> qh odd); the k/v path stays
                # redundant (identical values, as the old full-redundancy).
                # fb+9's rendezvous (cube passes only after BOTH aivs set)
                # covers both q_buf rows.
                with T.Scope("Vector"):
                    ub_c = T.alloc_ub((1, 1, d), "float32")
                    ub_sn = T.alloc_ub((1, 1, d), "float32")
                    ub_f0 = T.alloc_ub((1, 1, d), "float32")
                    ub_r = T.alloc_ub((1, 1, d), "float32")
                    ub_fk = T.alloc_ub((1, 1, d), "float32")
                    ub_fv = T.alloc_ub((1, 1, d), "float32")
                    ub_b = T.alloc_ub((1, 1, d), "bfloat16")
                    ub_qn = T.alloc_ub((1, 1, d), "bfloat16")
                    ub_kn = T.alloc_ub((1, 1, d), "bfloat16")
                    ub_qnf = T.alloc_ub((1, 1, d), "float32")
                    ub_knf = T.alloc_ub((1, 1, d), "float32")
                    ub_t = T.alloc_ub((1, 1, d), "float32")
                    red2 = T.alloc_ub((1, 1, 1), "float32")
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(fb + 4)
                    qh_ = kv * 2 + subid
                    with T.rs("PIPE_MTE2"):
                        T.copy(cos_tab[sl, 0:d], ub_c[0, 0, 0:d])
                        T.copy(sin_tab[sl, 0:d], ub_sn[0, 0, 0:d])
                        T.copy(qnw[li, 0:d], ub_qn[0, 0, 0:d])
                        T.copy(knw[li, 0:d], ub_kn[0, 0, 0:d])
                        T.copy(qkv_out[0:1, qh_ * d : qh_ * d + d], ub_f0[0, 0, 0:d])
                        T.copy(qkv_out[0:1, kbase + kv * d : kbase + kv * d + d], ub_fk[0, 0, 0:d])
                        T.copy(qkv_out[0:1, vbase + kv * d : vbase + kv * d + d], ub_fv[0, 0, 0:d])
                    T.vcast(ub_qn, ub_qnf)
                    T.vcast(ub_kn, ub_knf)
                    # qk-norm (per head over d): land bf16 before gamma, then rope
                    ub_nb = T.alloc_ub((1, 1, d), "bfloat16")
                    T.vmul(ub_f0, ub_f0, ub_t)
                    T.reduce(ub_t, red2, dims=[2], reduce_mode="sum", clear=True)
                    T.vmul(red2, inv_d, red2)
                    T.vadd(red2, eps, red2)
                    T.vsqrt(red2, red2)
                    T.vdiv(ub_f0, red2, ub_f0)
                    T.vcast(ub_f0, ub_nb)
                    T.vcast(ub_nb, ub_f0)
                    T.vmul(ub_f0, ub_qnf, ub_f0)
                    T.vmul(ub_fk, ub_fk, ub_t)
                    T.reduce(ub_t, red2, dims=[2], reduce_mode="sum", clear=True)
                    T.vmul(red2, inv_d, red2)
                    T.vadd(red2, eps, red2)
                    T.vsqrt(red2, red2)
                    T.vdiv(ub_fk, red2, ub_fk)
                    T.vcast(ub_fk, ub_nb)
                    T.vcast(ub_nb, ub_fk)
                    T.vmul(ub_fk, ub_knf, ub_fk)
                    T.copy(ub_f0[0, 0, d // 2 : d], ub_r[0, 0, 0 : d // 2])
                    T.copy(ub_f0[0, 0, 0 : d // 2], ub_r[0, 0, d // 2 : d])
                    T.vmul(ub_f0, ub_c, ub_f0)
                    T.vmul(ub_r, ub_sn, ub_r)
                    T.vadd(ub_f0, ub_r, ub_f0)
                    T.vcast(ub_f0, ub_b)
                    T.copy(ub_b[0, 0, 0:d], q_buf[qh_, 0:d])
                    if sl >= s0:
                        if sl < s0 + thirds:
                            T.copy(ub_fk[0, 0, d // 2 : d], ub_r[0, 0, 0 : d // 2])
                            T.copy(ub_fk[0, 0, 0 : d // 2], ub_r[0, 0, d // 2 : d])
                            T.vmul(ub_fk, ub_c, ub_fk)
                            T.vmul(ub_r, ub_sn, ub_r)
                            T.vadd(ub_fk, ub_r, ub_fk)
                            T.vcast(ub_fk, ub_b)
                            T.copy(ub_b[0, 0, 0:d], Kc[kc0 + sl, 0:d])
                            T.vcast(ub_fv, ub_b)
                            T.copy(ub_b[0, 0, 0:d], Vc[kc0 + sl, 0:d])
                    with T.rs("PIPE_MTE3"):
                        # P2-narrowcast: no vector-plane barrier here.  The
                        # append (row sl) and the q_buf rows this block's
                        # gemm1 consumes are all SAME-BLOCK data (the slice
                        # owning sl appends it itself; q_buf rows are written
                        # by every block of the kv group with identical
                        # values), so the same-block fb+9 rendezvous alone
                        # covers gemm1 -- dropping the barrier lets each
                        # cube start gemm1 as soon as its own aivs finish S2.
                        T.sync_block_set(fb + 9)

                # ---------- cube: gemm1 scores (local layout) ----------
                # K tiles double-buffered (l1_k0/l1_k1); first tile loads only
                # AFTER fb+9: the slice may hold the row the aiv just appended.
                with T.Scope("Cube"):
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(fb + 9)
                    if sblk > 0:
                        l1_q = T.alloc_L1((1, d), "bfloat16")
                        l1_k0 = T.alloc_L1((slt, d), "bfloat16")
                        l1_k1 = T.alloc_L1((slt, d), "bfloat16")
                        l0_s = T.alloc_L0C((slt, 1), "float32")
                        for gh in T.serial(2):
                            qh_ = kv * 2 + gh
                            prow = cid * 2 + gh
                            T.copy(q_buf[qh_ : qh_ + 1, 0:d], l1_q[0:1, 0:d])
                            nloc = T.ceildiv(sblk, slt)
                            T.copy(
                                Kc[kc0 + s0 : kc0 + s0 + slt, 0:d],
                                l1_k0[0:slt, 0:d],
                            )
                            for jr in T.serial(T.ceildiv(step, slt)):
                                if jr < nloc:
                                    l0 = jr * slt
                                    if jr % 2 == 0:
                                        if jr + 1 < nloc:
                                            T.copy(
                                                Kc[kc0 + s0 + l0 + slt : kc0 + s0 + l0 + 2 * slt, 0:d],
                                                l1_k1[0:slt, 0:d],
                                            )
                                        T.gemm(
                                            l1_k0, l1_q, l0_s,
                                            initC=True, b_transpose=True,
                                            size=[slt, d, 1],
                                        )
                                    else:
                                        if jr + 1 < nloc:
                                            T.copy(
                                                Kc[kc0 + s0 + l0 + slt : kc0 + s0 + l0 + 2 * slt, 0:d],
                                                l1_k0[0:slt, 0:d],
                                            )
                                        T.gemm(
                                            l1_k1, l1_q, l0_s,
                                            initC=True, b_transpose=True,
                                            size=[slt, d, 1],
                                        )
                                    with T.rs("PIPE_FIX"):
                                        T.copy(
                                            l0_s[0:slt, 0:1],
                                            scores[prow * step + l0 : prow * step + l0 + slt, 0:1],
                                        )
                        with T.rs("PIPE_FIX"):
                            T.sync_block_set(fb + 3)

                # ---------- aiv S3: softmax (local layout) ----------
                # P1-1: each aiv softmaxes one q head (gh = subid); both set
                # fb+5 after their half, and the cube's rendezvous wait then
                # covers both probs rows / both partial slots.
                with T.Scope("Vector"):
                    with T.If(sblk > 0):
                        with T.Then():
                            ub_s = T.alloc_ub((1, 1, step), "float32")
                            ub_de = T.alloc_ub((1, 1, step), "float32")
                            ub_pe = T.alloc_ub((1, 1, step), "bfloat16")
                            ub_m = T.alloc_ub((1, 1, 1), "float32")
                            ub_l = T.alloc_ub((1, 1, 1), "float32")
                            with T.rs("PIPE_MTE2"):
                                T.sync_block_wait(fb + 3)
                            gh = subid
                            prow = cid * 2 + gh
                            pgrp = (kv * 2 + gh) * nsl + sid
                            T.vbrc(neg_inf, ub_s[0, 0, 0:step])
                            with T.rs("PIPE_MTE2"):
                                T.copy(
                                    scores[prow * step : prow * step + sblk, 0],
                                    ub_s[0, 0, 0:sblk],
                                )
                            T.vmul(ub_s[0, 0, 0:step], scale, ub_s[0, 0, 0:step])
                            T.reduce(
                                ub_s[0, 0, 0:step], ub_m[0, 0, 0:1],
                                dims=[2], reduce_mode="max",
                            )
                            T.vsub(ub_s[0, 0, 0:step], ub_m[0, 0, 0:1], ub_de[0, 0, 0:step])
                            T.vexp(ub_de[0, 0, 0:step], ub_de[0, 0, 0:step])
                            T.reduce(
                                ub_de[0, 0, 0:step], ub_l[0, 0, 0:1],
                                dims=[2], reduce_mode="sum",
                            )
                            T.vcast(ub_de[0, 0, 0:step], ub_pe[0, 0, 0:step])
                            ext = T.ceildiv(sblk, slt) * slt
                            T.copy(ub_pe[0, 0, 0:ext], probs[prow : prow + 1, 0:ext])
                            T.copy(ub_m[0, 0, 0:1], part_m[pgrp : pgrp + 1])
                            T.copy(ub_l[0, 0, 0:1], part_l[pgrp : pgrp + 1])
                            with T.rs("PIPE_MTE3"):
                                T.sync_block_set(fb + 5)
                        with T.Else():
                            # empty slice: store the neutral partial itself, so
                            # the leader's combine never reads a stale slot (a
                            # slice empty this step may have been active in an
                            # earlier call sharing these buffers)
                            gh = subid
                            pgrp = (kv * 2 + gh) * nsl + sid
                            ub_nm = T.alloc_ub((1, 1, 1), "float32")
                            ub_nz = T.alloc_ub((1, 1, d), "float32")
                            T.vbrc(neg_inf, ub_nm[0, 0, 0:1])
                            T.vbrc(zero, ub_nz[0, 0, 0:d])
                            T.copy(ub_nm[0, 0, 0:1], part_m[pgrp : pgrp + 1])
                            T.copy(ub_nm[0, 0, 0:1], part_l[pgrp : pgrp + 1])
                            T.copy(ub_nz[0, 0, 0:d], part_acc[pgrp : pgrp + 1, 0:d])

                # ---------- cube: gemm2 acc ----------
                # V tiles double-buffered (l1_v0/l1_v1), probs too (tiny);
                # prologue after fb+5 (covers probs and the appended V row).
                with T.Scope("Cube"):
                    if sblk > 0:
                        l1_p0 = T.alloc_L1((1, slt), "bfloat16")
                        l1_p1 = T.alloc_L1((1, slt), "bfloat16")
                        l1_v0 = T.alloc_L1((slt, d), "bfloat16")
                        l1_v1 = T.alloc_L1((slt, d), "bfloat16")
                        l0_o = T.alloc_L0C((1, d), "float32")
                        with T.rs("PIPE_MTE2"):
                            T.sync_block_wait(fb + 5)
                        for gh in T.serial(2):
                            prow = cid * 2 + gh
                            pgrp = (kv * 2 + gh) * nsl + sid
                            nloc = T.ceildiv(sblk, slt)
                            T.copy(probs[prow : prow + 1, 0:slt], l1_p0[0:1, 0:slt])
                            T.copy(
                                Vc[kc0 + s0 : kc0 + s0 + slt, 0:d],
                                l1_v0[0:slt, 0:d],
                            )
                            for jr in T.serial(T.ceildiv(step, slt)):
                                if jr < nloc:
                                    g0 = jr * slt
                                    if jr % 2 == 0:
                                        if jr + 1 < nloc:
                                            T.copy(
                                                probs[prow : prow + 1, g0 + slt : g0 + 2 * slt],
                                                l1_p1[0:1, 0:slt],
                                            )
                                            T.copy(
                                                Vc[kc0 + s0 + g0 + slt : kc0 + s0 + g0 + 2 * slt, 0:d],
                                                l1_v1[0:slt, 0:d],
                                            )
                                        T.gemm(
                                            l1_p0, l1_v0, l0_o,
                                            initC=(jr == 0), b_transpose=False,
                                            size=[1, slt, d],
                                        )
                                    else:
                                        if jr + 1 < nloc:
                                            T.copy(
                                                probs[prow : prow + 1, g0 + slt : g0 + 2 * slt],
                                                l1_p0[0:1, 0:slt],
                                            )
                                            T.copy(
                                                Vc[kc0 + s0 + g0 + slt : kc0 + s0 + g0 + 2 * slt, 0:d],
                                                l1_v0[0:slt, 0:d],
                                            )
                                        T.gemm(
                                            l1_p1, l1_v1, l0_o,
                                            initC=(jr == 0), b_transpose=False,
                                            size=[1, slt, d],
                                        )
                            with T.rs("PIPE_FIX"):
                                T.copy(
                                    l0_o[0:1, 0:d],
                                    part_acc[pgrp : pgrp + 1, 0:d],
                                )
                    with T.rs("PIPE_FIX"):
                        # P2-narrowcast: the partials rendezvous moved off
                        # the cube plane.  Each cube relays fb+6 the moment
                        # ITS OWN gemm2 fixpipe lands (no 24-cube barrier);
                        # the vector-plane fb+2 at S4 entry is the carrier:
                        # every aiv arrives there only after its own cube's
                        # fb+6, and arriving on PIPE_MTE3 orders it after
                        # its own S3 stores, so a released leader sees all
                        # three slices' part_m/l/acc.  The freed cube flows
                        # straight into the o-scope Wo prefetch below,
                        # overlapping slower blocks' gemm2 tail.
                        T.sync_block_set(fb + 6)

                # ---------- aiv S4: combine (leader sid==0) ----------
                # the LSE merge is folded into ONE vector pass: with nsl=3 the
                # weighted sum num = sum_j w_j * acc_j (3 x d f32 in UB) needs
                # no cube trip -- this removes the acc16/wrow16/num GM staging,
                # the whole gemm3 cube scope and the fb+7/fb+8 rendezvous from
                # the per-layer critical path (fb+15 gates the o gemv).
                # numerics: acc/weights stay f32 through the weighted sum (the
                # old path landed both to bf16 for the tiny gemm) -- closer to
                # the fp32 reference.
                with T.Scope("Vector"):
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(fb + 6)
                    with T.rs("PIPE_MTE3"):
                        T.block_barrier(fb + 2)
                if sid == 0:
                    with T.Scope("Vector"):
                        ub_ms = T.alloc_ub((nsl, 1), "float32")
                        ub_ls = T.alloc_ub((nsl, 1), "float32")
                        ub_ws = T.alloc_ub((nsl, 1), "float32")
                        ub_wl = T.alloc_ub((nsl, 1), "float32")
                        ub_mstar = T.alloc_ub((1, 1), "float32")
                        ub_den = T.alloc_ub((1, 1), "float32")
                        ub_a3 = T.alloc_ub((nsl, d), "float32")
                        ub_w16 = T.alloc_ub((nsl, 1), "bfloat16")
                        ub_a16 = T.alloc_ub((nsl, d), "bfloat16")
                        ub_num = T.alloc_ub((1, d), "float32")
                        ub_o2 = T.alloc_ub((1, d), "bfloat16")
                        # P1-1: the two aivs split the combine -- each handles
                        # one q head of the kv group.  fb+15's rendezvous
                        # semantics (cube passes only after BOTH aivs set)
                        # keep this safe with no extra barrier.
                        gh = subid
                        gbase = (kv * 2 + gh) * nsl
                        with T.rs("PIPE_MTE2"):
                            T.copy(part_m[gbase : gbase + nsl], ub_ms[0:nsl, 0:1])
                            T.copy(part_l[gbase : gbase + nsl], ub_ls[0:nsl, 0:1])
                            T.copy(part_acc[gbase : gbase + nsl, 0:d], ub_a3[0:nsl, 0:d])
                        # w_j = exp(m_j - m*); den = sum_j w_j*l_j
                        T.reduce(ub_ms, ub_mstar, dims=[0], reduce_mode="max", clear=True)
                        T.vsub(ub_ms, ub_mstar, ub_ws)
                        T.vexp(ub_ws, ub_ws)
                        T.vmul(ub_ws, ub_ls, ub_wl)
                        T.reduce(ub_wl, ub_den, dims=[0], reduce_mode="sum", clear=True)
                        # num = sum_j w_j * acc_j (column-broadcast), /den;
                        # land bf16 on both factors first -- the same
                        # rounding discipline the old gemm3 path applied
                        T.vcast(ub_ws, ub_w16)
                        T.vcast(ub_w16, ub_ws)
                        T.vcast(ub_a3, ub_a16)
                        T.vcast(ub_a16, ub_a3)
                        T.vmul(ub_a3, ub_ws, ub_a3)
                        T.reduce(ub_a3, ub_num, dims=[0], reduce_mode="sum", clear=True)
                        T.vdiv(ub_num, ub_den, ub_num)
                        T.vcast(ub_num, ub_o2)
                        T.copy(ub_o2[0, 0:d], attn_out[kv * 2 + gh, 0:d])
                with T.Scope("Vector"):
                    with T.rs("PIPE_MTE3"):
                        T.block_barrier(fb + 2)
                        T.sync_block_set(fb + 15)

                # ---------- cube: MLP tail (o + gate_up + down) in ONE scope ----------
                # the three MLP GEMVs share ONE explicit ring (l1_b0/l1_b1 at
                # the max nc = 88; l1_a0/l1_a1; per-GEMV l0_c) instead of three
                # scope-local buffer sets the allocator had to overlay.  Wins:
                #  * o is finally double-buffered (it lost the old budget fight)
                #  * each GEMV's two first weight tiles stream BEFORE its flag
                #    wait (fb+15/fb+11/fb+13), so the MTE2 pipe stays busy
                #    through the combine tail, the rms2 stage and the silu
                #    stage instead of draining at every rendezvous
                #  * the ring lifetime is explicit -> deterministic footprint
                # (gu tiles occupy only the first 64 of the 88 ring rows; the
                # gemm size tail covers it, as with the old tail blocks.)
                with T.Scope("Cube"):
                    l1_a0 = T.alloc_L1((1, ktile), "bfloat16")
                    l1_a1 = T.alloc_L1((1, ktile), "bfloat16")
                    l1_b0 = T.alloc_L1((nc_o, ktile), "bfloat16")
                    l1_b1 = T.alloc_L1((nc_o, ktile), "bfloat16")
                    # one accumulator per GEMV: a single shared l0_c spanning
                    # o..down gets its LCA hoisted to the layer-loop level,
                    # which trips a stateful placement bug in
                    # PlanAndUpdateBufferAllocationLocation (its need_extract
                    # latch then drops lm_head's l1_a1/l1_b1 allocations)
                    l0_co = T.alloc_L0C((1, nc_o), "float32")
                    l0_cg = T.alloc_L0C((1, nc_o), "float32")
                    l0_cd = T.alloc_L0C((1, nc_o), "float32")

                    # keep the ring's LCA inside this scope: a dynamic trip-1
                    # loop gives PlanAndUpdateBufferAllocationLocation an inner
                    # ForNode to attach the ring allocations to (hoisting them
                    # to the layer loop anchored their liveness to the whole
                    # layer and blew the 512 KiB cbuf budget)
                    for _r in T.serial(T.min(sl + 1, 1)):

                        # ---- o gemv (double-buffered on the ring) ----
                        pfo = T.min(nc_o, h - cid * nc_o * nrd_o)
                        T.copy(
                            Wo[wo0 + cid * nc_o * nrd_o : wo0 + cid * nc_o * nrd_o + pfo, 0:ktile],
                            l1_b0[0:pfo, 0:ktile],
                        )
                        T.copy(
                            Wo[wo0 + cid * nc_o * nrd_o : wo0 + cid * nc_o * nrd_o + pfo, ktile : 2 * ktile],
                            l1_b1[0:pfo, 0:ktile],
                        )
                        with T.rs("PIPE_MTE2"):
                            T.sync_block_wait(fb + 15)
                        T.copy(attn_flat[0:1, 0:ktile], l1_a0[0:1, 0:ktile])
                        T.copy(attn_flat[0:1, ktile : 2 * ktile], l1_a1[0:1, 0:ktile])
                        for nt in T.serial(nrd_o):
                            n0 = cid * nc_o * nrd_o + nt * nc_o
                            if n0 < h:
                                tail = T.min(nc_o, h - n0)
                                for kt in T.serial(h // ktile):
                                    k0 = kt * ktile
                                    if kt % 2 == 0:
                                        if kt + nt > 0:
                                            if kt + 1 < h // ktile:
                                                T.copy(attn_flat[0:1, k0 + ktile : k0 + 2 * ktile], l1_a1[0:1, 0:ktile])
                                                T.copy(
                                                    Wo[wo0 + n0 : wo0 + n0 + tail, k0 + ktile : k0 + 2 * ktile],
                                                    l1_b1[0:tail, 0:ktile],
                                                )
                                        T.gemm(
                                            l1_a0, l1_b0, l0_co,
                                            initC=(kt == 0), b_transpose=True,
                                            size=[1, ktile, tail],
                                        )
                                    else:
                                        if kt + 1 < h // ktile:
                                            T.copy(attn_flat[0:1, k0 + ktile : k0 + 2 * ktile], l1_a0[0:1, 0:ktile])
                                            T.copy(
                                                Wo[wo0 + n0 : wo0 + n0 + tail, k0 + ktile : k0 + 2 * ktile],
                                                l1_b0[0:tail, 0:ktile],
                                            )
                                        T.gemm(
                                            l1_a1, l1_b1, l0_co,
                                            initC=(kt == 0), b_transpose=True,
                                            size=[1, ktile, tail],
                                        )
                                with T.rs("PIPE_FIX"):
                                    T.copy(l0_co[0:1, 0:tail], o_out[0:1, n0 : n0 + tail])
                        with T.rs("PIPE_FIX"):
                            T.block_barrier(fb + 1)
                            T.sync_block_set(fb + 10)

                        # ---- gate_up gemv (contiguous 512-wide slabs) ----
                        pfg = T.min(nc_gu, 2 * ii - cid * slab_gu)
                        T.copy(
                            Wgu[wg0 + cid * slab_gu : wg0 + cid * slab_gu + pfg, 0:ktile],
                            l1_b0[0:pfg, 0:ktile],
                        )
                        T.copy(
                            Wgu[wg0 + cid * slab_gu : wg0 + cid * slab_gu + pfg, ktile : 2 * ktile],
                            l1_b1[0:pfg, 0:ktile],
                        )
                        with T.rs("PIPE_MTE2"):
                            T.sync_block_wait(fb + 11)
                        T.copy(xn2[0:1, 0:ktile], l1_a0[0:1, 0:ktile])
                        T.copy(xn2[0:1, ktile : 2 * ktile], l1_a1[0:1, 0:ktile])
                        for nt in T.serial(nrd_gu):
                            n0 = cid * slab_gu + nt * nc_gu
                            if n0 < 2 * ii:
                                tail = T.min(nc_gu, 2 * ii - n0)
                                for kt in T.serial(h // ktile):
                                    k0 = kt * ktile
                                    if kt % 2 == 0:
                                        if kt + nt > 0:
                                            if kt + 1 < h // ktile:
                                                T.copy(xn2[0:1, k0 + ktile : k0 + 2 * ktile], l1_a1[0:1, 0:ktile])
                                                T.copy(
                                                    Wgu[wg0 + n0 : wg0 + n0 + tail, k0 + ktile : k0 + 2 * ktile],
                                                    l1_b1[0:tail, 0:ktile],
                                                )
                                            else:
                                                n0n = n0 + nc_gu
                                                if nt + 1 < nrd_gu:
                                                    if n0n < 2 * ii:
                                                        tailn = T.min(nc_gu, 2 * ii - n0n)
                                                        T.copy(xn2[0:1, 0:ktile], l1_a1[0:1, 0:ktile])
                                                        T.copy(
                                                            Wgu[wg0 + n0n : wg0 + n0n + tailn, 0:ktile],
                                                            l1_b1[0:tailn, 0:ktile],
                                                        )
                                        T.gemm(
                                            l1_a0, l1_b0, l0_cg,
                                            initC=(kt == 0), b_transpose=True,
                                            size=[1, ktile, tail],
                                        )
                                    else:
                                        if kt + 1 < h // ktile:
                                            T.copy(xn2[0:1, k0 + ktile : k0 + 2 * ktile], l1_a0[0:1, 0:ktile])
                                            T.copy(
                                                Wgu[wg0 + n0 : wg0 + n0 + tail, k0 + ktile : k0 + 2 * ktile],
                                                l1_b0[0:tail, 0:ktile],
                                            )
                                        else:
                                            n0n = n0 + nc_gu
                                            if nt + 1 < nrd_gu:
                                                if n0n < 2 * ii:
                                                    tailn = T.min(nc_gu, 2 * ii - n0n)
                                                    T.copy(xn2[0:1, 0:ktile], l1_a0[0:1, 0:ktile])
                                                    T.copy(
                                                        Wgu[wg0 + n0n : wg0 + n0n + tailn, 0:ktile],
                                                        l1_b0[0:tailn, 0:ktile],
                                                    )
                                        T.gemm(
                                            l1_a1, l1_b1, l0_cg,
                                            initC=(kt == 0), b_transpose=True,
                                            size=[1, ktile, tail],
                                        )
                                with T.rs("PIPE_FIX"):
                                    T.copy(l0_cg[0:1, 0:tail], gu_out[0:1, n0 : n0 + tail])
                        with T.rs("PIPE_FIX"):
                            T.block_barrier(fb + 1)
                            T.sync_block_set(fb + 12)

                        # ---- down gemv (K axis is ii) ----
                        pfd = T.min(nc_d, h - cid * nc_d * nrd_d)
                        T.copy(
                            Wd[wo0 + cid * nc_d * nrd_d : wo0 + cid * nc_d * nrd_d + pfd, 0:ktile],
                            l1_b0[0:pfd, 0:ktile],
                        )
                        T.copy(
                            Wd[wo0 + cid * nc_d * nrd_d : wo0 + cid * nc_d * nrd_d + pfd, ktile : 2 * ktile],
                            l1_b1[0:pfd, 0:ktile],
                        )
                        with T.rs("PIPE_MTE2"):
                            T.sync_block_wait(fb + 13)
                        T.copy(h_buf[0:1, 0:ktile], l1_a0[0:1, 0:ktile])
                        T.copy(h_buf[0:1, ktile : 2 * ktile], l1_a1[0:1, 0:ktile])
                        for nt in T.serial(nrd_d):
                            n0 = cid * nc_d * nrd_d + nt * nc_d
                            if n0 < h:
                                tail = T.min(nc_d, h - n0)
                                for kt in T.serial(ii // ktile):
                                    k0 = kt * ktile
                                    if kt % 2 == 0:
                                        if kt + nt > 0:
                                            if kt + 1 < ii // ktile:
                                                T.copy(h_buf[0:1, k0 + ktile : k0 + 2 * ktile], l1_a1[0:1, 0:ktile])
                                                T.copy(
                                                    Wd[wo0 + n0 : wo0 + n0 + tail, k0 + ktile : k0 + 2 * ktile],
                                                    l1_b1[0:tail, 0:ktile],
                                                )
                                            else:
                                                n0n = n0 + nc_d
                                                if nt + 1 < nrd_d:
                                                    if n0n < h:
                                                        tailn = T.min(nc_d, h - n0n)
                                                        T.copy(h_buf[0:1, 0:ktile], l1_a1[0:1, 0:ktile])
                                                        T.copy(
                                                            Wd[wo0 + n0n : wo0 + n0n + tailn, 0:ktile],
                                                            l1_b1[0:tailn, 0:ktile],
                                                        )
                                        T.gemm(
                                            l1_a0, l1_b0, l0_cd,
                                            initC=(kt == 0), b_transpose=True,
                                            size=[1, ktile, tail],
                                        )
                                    else:
                                        if kt + 1 < ii // ktile:
                                            T.copy(h_buf[0:1, k0 + ktile : k0 + 2 * ktile], l1_a0[0:1, 0:ktile])
                                            T.copy(
                                                Wd[wo0 + n0 : wo0 + n0 + tail, k0 + ktile : k0 + 2 * ktile],
                                                l1_b0[0:tail, 0:ktile],
                                            )
                                        else:
                                            n0n = n0 + nc_d
                                            if nt + 1 < nrd_d:
                                                if n0n < h:
                                                    tailn = T.min(nc_d, h - n0n)
                                                    T.copy(h_buf[0:1, 0:ktile], l1_a0[0:1, 0:ktile])
                                                    T.copy(
                                                        Wd[wo0 + n0n : wo0 + n0n + tailn, 0:ktile],
                                                        l1_b0[0:tailn, 0:ktile],
                                                    )
                                        T.gemm(
                                            l1_a1, l1_b1, l0_cd,
                                            initC=(kt == 0), b_transpose=True,
                                            size=[1, ktile, tail],
                                        )
                                with T.rs("PIPE_FIX"):
                                    T.copy(l0_cd[0:1, 0:tail], d_out[0:1, n0 : n0 + tail])
                        with T.rs("PIPE_FIX"):
                            T.block_barrier(fb + 1)
                            T.sync_block_set(fb + 14)

                # ---------- aiv S5': resid1 + rms2 + silu + resid2 ----------
                with T.Scope("Vector"):
                    ub_x = T.alloc_ub((1, 1, h), "bfloat16")
                    ub_xf = T.alloc_ub((1, 1, h), "float32")
                    ub_ov = T.alloc_ub((1, 1, h), "float32")
                    ub_xm = T.alloc_ub((1, 1, h), "float32")
                    ub_xn = T.alloc_ub((1, 1, h), "float32")
                    ub_g = T.alloc_ub((1, 1, h), "float32")
                    red = T.alloc_ub((1, 1, 1), "float32")
                    ub_w = T.alloc_ub((1, 1, h), "bfloat16")
                    ub_wf = T.alloc_ub((1, 1, h), "float32")
                    ub_o2 = T.alloc_ub((1, 1, h), "bfloat16")
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(fb + 10)
                    with T.rs("PIPE_MTE2"):
                        T.copy(x[0, 0:h], ub_x[0, 0, 0:h])
                        T.copy(o_out[0, 0:h], ub_ov[0, 0, 0:h])
                    T.vcast(ub_x, ub_xf)
                    T.vadd(ub_xf, ub_ov, ub_xm)
                    T.vmul(ub_xm, ub_xm, ub_g)
                    T.reduce(ub_g, red, dims=[2], reduce_mode="sum", clear=True)
                    T.vmul(red, inv_h, red)
                    T.vadd(red, eps, red)
                    T.vsqrt(red, red)
                    T.vdiv(ub_xm, red, ub_xn)
                    T.copy(rms2w[li, 0:h], ub_w[0, 0, 0:h])
                    T.vcast(ub_w, ub_wf)
                    # land bf16 before the learned scale (Qwen3RMSNorm)
                    T.vcast(ub_xn, ub_o2)
                    T.vcast(ub_o2, ub_xn)
                    T.vmul(ub_xn, ub_wf, ub_xn)
                    T.vcast(ub_xn, ub_o2)
                    T.copy(ub_o2[0, 0, 0:h], xn2[0, 0:h])
                    with T.rs("PIPE_MTE3"):
                        T.sync_block_set(fb + 11)
                    ub_gate = T.alloc_ub((1, 1, ii // 2), "float32")
                    ub_up = T.alloc_ub((1, 1, ii // 2), "float32")
                    ub_sg = T.alloc_ub((1, 1, ii // 2), "float32")
                    ub_hb = T.alloc_ub((1, 1, ii // 2), "bfloat16")
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(fb + 12)
                    i0 = subid * (ii // 2)
                    with T.rs("PIPE_MTE2"):
                        T.copy(gu_out[0:1, i0 : i0 + ii // 2], ub_gate[0, 0, 0:ii // 2])
                        T.copy(gu_out[0:1, ii + i0 : ii + i0 + ii // 2], ub_up[0, 0, 0:ii // 2])
                    T.vsigmoid(ub_gate, ub_sg)
                    T.vmul(ub_gate, ub_sg, ub_sg)
                    T.vmul(ub_sg, ub_up, ub_sg)
                    T.vcast(ub_sg, ub_hb)
                    T.copy(ub_hb[0, 0, 0:ii // 2], h_buf[0, i0 : i0 + ii // 2])
                    with T.rs("PIPE_MTE3"):
                        T.sync_block_set(fb + 13)
                    ub_dv = T.alloc_ub((1, 1, h), "float32")
                    with T.rs("PIPE_MTE2"):
                        T.sync_block_wait(fb + 14)
                    with T.rs("PIPE_MTE2"):
                        T.copy(d_out[0, 0:h], ub_dv[0, 0, 0:h])
                    T.vadd(ub_xm, ub_dv, ub_xm)
                    T.vcast(ub_xm, ub_o2)
                    T.copy(ub_o2[0, 0, 0:h], x[0, 0:h])

            # ================= final: rms + lm_head + argmax =================
            fbf = 16 * (l % 2)
            with T.Scope("Vector"):
                T.pipe_barrier("PIPE_ALL")
                ub_x = T.alloc_ub((1, 1, h), "bfloat16")
                ub_f = T.alloc_ub((1, 1, h), "float32")
                ub_g = T.alloc_ub((1, 1, h), "float32")
                ub_w = T.alloc_ub((1, 1, h), "bfloat16")
                ub_wf = T.alloc_ub((1, 1, h), "float32")
                ub_o = T.alloc_ub((1, 1, h), "bfloat16")
                red = T.alloc_ub((1, 1, 1), "float32")
                T.copy(x[0, 0:h], ub_x[0, 0, 0:h])
                T.vcast(ub_x, ub_f)
                T.vmul(ub_f, ub_f, ub_g)
                T.reduce(ub_g, red, dims=[2], reduce_mode="sum", clear=True)
                T.vmul(red, inv_h, red)
                T.vadd(red, eps, red)
                T.vsqrt(red, red)
                T.vdiv(ub_f, red, ub_f)
                T.copy(rmsfw[0:h], ub_w[0, 0, 0:h])
                T.vcast(ub_w, ub_wf)
                # land bf16 before the learned scale (Qwen3RMSNorm)
                T.vcast(ub_f, ub_o)
                T.vcast(ub_o, ub_f)
                T.vmul(ub_f, ub_wf, ub_f)
                T.vcast(ub_f, ub_o)
                T.copy(ub_o[0, 0, 0:h], xn1[0, 0:h])
                with T.rs("PIPE_MTE3"):
                    T.sync_block_set(fbf + 0)

            with T.Scope("Cube"):
                l1_a0 = T.alloc_L1((1, ktile), "bfloat16")
                l1_a1 = T.alloc_L1((1, ktile), "bfloat16")
                l1_b0 = T.alloc_L1((nc_lm, ktile), "bfloat16")
                l1_b1 = T.alloc_L1((nc_lm, ktile), "bfloat16")
                l0_c = T.alloc_L0C((1, nc_lm), "float32")
                pf0 = T.min(nc_lm, vv - cid * span)
                T.copy(
                    Wlm[cid * span : cid * span + pf0, 0:ktile],
                    l1_b0[0:pf0, 0:ktile],
                )
                with T.rs("PIPE_MTE2"):
                    T.sync_block_wait(fbf + 0)
                T.copy(xn1[0:1, 0:ktile], l1_a0[0:1, 0:ktile])
                for nt in T.serial(nrd_lm):
                    n0 = cid * span + nt * nc_lm
                    if n0 < vv:
                        tail = T.min(nc_lm, vv - n0)
                        for kt in T.serial(h // ktile):
                            k0 = kt * ktile
                            if kt % 2 == 0:
                                if kt + 1 < h // ktile:
                                    T.copy(xn1[0:1, k0 + ktile : k0 + 2 * ktile], l1_a1[0:1, 0:ktile])
                                    T.copy(
                                        Wlm[n0 : n0 + tail, k0 + ktile : k0 + 2 * ktile],
                                        l1_b1[0:tail, 0:ktile],
                                    )
                                else:
                                    n0n = n0 + nc_lm
                                    if nt + 1 < nrd_lm:
                                        if n0n < vv:
                                            tailn = T.min(nc_lm, vv - n0n)
                                            T.copy(xn1[0:1, 0:ktile], l1_a1[0:1, 0:ktile])
                                            T.copy(
                                                Wlm[n0n : n0n + tailn, 0:ktile],
                                                l1_b1[0:tailn, 0:ktile],
                                            )
                                T.gemm(
                                    l1_a0, l1_b0, l0_c,
                                    initC=(kt == 0), b_transpose=True,
                                    size=[1, ktile, tail],
                                )
                            else:
                                if kt + 1 < h // ktile:
                                    T.copy(xn1[0:1, k0 + ktile : k0 + 2 * ktile], l1_a0[0:1, 0:ktile])
                                    T.copy(
                                        Wlm[n0 : n0 + tail, k0 + ktile : k0 + 2 * ktile],
                                        l1_b0[0:tail, 0:ktile],
                                    )
                                else:
                                    n0n = n0 + nc_lm
                                    if nt + 1 < nrd_lm:
                                        if n0n < vv:
                                            tailn = T.min(nc_lm, vv - n0n)
                                            T.copy(xn1[0:1, 0:ktile], l1_a0[0:1, 0:ktile])
                                            T.copy(
                                                Wlm[n0n : n0n + tailn, 0:ktile],
                                                l1_b0[0:tailn, 0:ktile],
                                            )
                                T.gemm(
                                    l1_a1, l1_b1, l0_c,
                                    initC=(kt == 0), b_transpose=True,
                                    size=[1, ktile, tail],
                                )
                        with T.rs("PIPE_FIX"):
                            T.copy(l0_c[0:1, 0:tail], logits[0:1, n0 : n0 + tail])
                with T.rs("PIPE_FIX"):
                    T.sync_block_set(fbf + 1)

            with T.Scope("Vector"):
                # chunked argmax: logits GM is host-padded to nb*spanp with -inf,
                # so every chunk load is a full static-extent copy (no tail fill)
                ub_lc = T.alloc_ub((1, 1, ch), "float32")
                ub_ic = T.alloc_ub((1, 1, ch), "int32")
                ub_ifc = T.alloc_ub((1, 1, ch), "float32")
                ub_mc = T.alloc_ub((1, 1, ch), "float32")
                redc = T.alloc_ub((1, 1, 1), "float32")
                redi = T.alloc_ub((1, 1, 1), "float32")
                with T.rs("PIPE_MTE2"):
                    T.sync_block_wait(fbf + 1)
                for c0 in T.serial(nch // 2):
                    c = subid * (nch // 2) + c0
                    off = cid * span + c * ch
                    with T.rs("PIPE_MTE2"):
                        T.copy(logits[0, off : off + ch], ub_lc[0, 0, 0:ch])
                        T.copy(idx_tab[c * ch : c * ch + ch], ub_ic[0, 0, 0:ch])
                    T.reduce(ub_lc, redc, dims=[2], reduce_mode="max", clear=True)
                    T.vsub(ub_lc, redc, ub_mc)
                    T.vmul(ub_mc, big, ub_mc)
                    T.vsigmoid(ub_mc, ub_mc)
                    T.vmul(ub_mc, two, ub_mc)
                    T.vcast(ub_ic, ub_ifc)
                    T.vmul(ub_mc, ub_ifc, ub_mc)
                    T.reduce(ub_mc, redi, dims=[2], reduce_mode="max", clear=True)
                    T.copy(redc[0, 0, 0:1], part_max[cid * nch + c : cid * nch + c + 1])
                    T.copy(redi[0, 0, 0:1], part_idx[cid * nch + c : cid * nch + c + 1])
    return main


# ---------------- shape helper (single source of truth for the layouts) ---- #
def mega_shapes(l=28, h=2048, qh=16, kvh=8, d=128, ii=6144, vv=151936,
                s=40960, slt=128, nb=24, ktile=512, kpad=16,
                nc_qkv=88, nc_o=88, nc_gu=64, nc_d=88, nc_lm=64, ch=2048):
    """The kernel's tiling constants for one model/context-capacity pair."""
    nsl = nb // kvh
    step = ((s + nsl - 1) // nsl + slt - 1) // slt * slt
    spadg = step * nsl
    nqkv = qh * d + 2 * kvh * d
    nqkv_pad = nb * nc_qkv * ((nqkv - 1) // (nb * nc_qkv) + 1)  # 24*88*2 = 4224
    nrd_qkv = (nqkv - 1) // (nc_qkv * nb) + 1
    nrd_o = (h - 1) // (nc_o * nb) + 1
    nrd_gu = (2 * ii - 1) // (nc_gu * nb) + 1
    nrd_d = (h - 1) // (nc_d * nb) + 1
    span = (vv + nb - 1) // nb
    nrd_lm = (span - 1) // nc_lm + 1
    spanp = (span + ch - 1) // ch * ch
    nch = spanp // ch
    return dict(l=l, h=h, qh=qh, kvh=kvh, d=d, ii=ii, vv=vv, slt=slt, nb=nb,
                nsl=nsl, step=step, spadg=spadg, nqkv=nqkv, nqkv_pad=nqkv_pad,
                ktile=ktile, nc_qkv=nc_qkv, nrd_qkv=nrd_qkv, nc_o=nc_o,
                nrd_o=nrd_o, nc_gu=nc_gu, nrd_gu=nrd_gu, nc_d=nc_d, nrd_d=nrd_d,
                span=span, nrd_lm=nrd_lm, nc_lm=nc_lm, spanp=spanp, kpad=kpad,
                ch=ch, nch=nch)


__all__ = ["mega", "mega_shapes"]
