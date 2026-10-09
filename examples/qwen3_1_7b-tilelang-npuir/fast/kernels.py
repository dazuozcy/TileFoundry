"""tilelang-npuir kernels for one Qwen3-1.7B decode step on Ascend 910B.

Every kernel takes caller-owned buffers and reads the step's position from a
one-element device tensor, so the whole decode step can be captured into one
NPU graph and replayed without a host round trip.

What the Ascend backend fixes about the shapes here:

* The vector units take fp16/fp32 only -- no bf16 arithmetic. The projections
  therefore run as *fp16* vector GEMVs (weights converted once at load; the
  products keep more mantissa than the model's own bf16), while attention
  stays on the cube in bf16, which is where `T.gemm` wants it. Between
  kernels everything is computed in f32 and rounded once, at the point the
  published model itself rounds.
* A decode GEMV is pure streaming, and the cube's skinny-M load path reaches
  only ~80 GB/s on this part -- the vector form (`vbrc` the input across a
  (BN, BK) tile, `vmul`, `reduce_sum`, accumulate in f32) reaches ~1.4 TB/s
  when the block count is near the 48 AI cores, so every projection launches
  a fixed 48 cores and walks its tiles serially in-core.
* `T.gemm` is the cube path: 2D operands, f32 accumulator, `b_transpose` for
  the score product. A decode GEMV is (N, K) @ (K, 1): one block owns BN rows
  of the weight and walks K.
* The grid is one-dimensional and the device has 48 AI cores / 24 cube units,
  so the wide projections launch a fixed core count and walk tiles serially
  in-core (a block count past the core count gets serialized by the scheduler
  anyway -- walking inside the block skips the re-scheduling).
* Row-shaped work (norms, merges, sampling) runs on (1, N) buffers inside
  2D `T.Parallel` loops and buffer intrinsics (`vbrc`, `vcmp`, `vselect`,
  `vexp`, `vsqrt`, `vdiv`, `arange`): a scalar read from a (1,) buffer inside
  a Parallel loop crashes the vectorizer, so every broadcast goes through a
  buffer op; `vrsqrt` is approximate (0.3% error) so rsqrt is spelled
  `vdiv(1, vsqrt(x))`.
* Global-memory writes go through `T.copy` onto 1D slices or matching-shape
  origin regions -- an elementwise store with a computed 2D index faults the
  AI core, while reads with computed indices are fine. Partials and per-head
  scratch are therefore laid out flat (1D) and every kernel addresses them
  with computed offsets on read, `T.copy` on write.

Two deliberate departures from the authored reference, both toward the
published model (same two the shipped CUDA twin makes):

* the `1/sqrt(head_dim)` factor is applied to the finished f32 score, not to
  q in bf16 first;
* attention probabilities are rounded to bf16 before the V product, which is
  what Hugging Face's own attention does.
"""
from functools import lru_cache

import tilelang
import tilelang.language as T

DT = "bfloat16"
F16 = "float16"
F32 = "float32"
NEG = -1.0e30
TILELANG_TARGET = "npuir"


#: The auto-multi-buffer pass remaps storage slots of loop-carried buffers,
#: so a vector read after the loop sees stale data (argreduce's
#: CG-2026-0011, reproduced on this build). Every kernel here carries state
#: across its serial tile loop, so it is off for all of them.
_PASS_CONFIGS = {"npuir.enable_auto_multi_buffer": False}


def _compile(prim):
    return tilelang.compile(
        prim, target=TILELANG_TARGET, pass_configs=_PASS_CONFIGS
    )


def _neg(dst):
    T.vbrc(T.cast(NEG, F32), dst)


# --------------------------------------------------------------------------- #
# embedding
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=None)
def embed(V: int, H: int):
    """The decoded token's own row of the table."""
    @T.prim_func
    def main(Tbl: T.Tensor((V, H), DT), Ids: T.Tensor((1,), "int32"),
             O: T.Tensor((H,), DT)):
        with T.Kernel(1, is_npu=True) as (bn, _):
            buf = T.alloc_shared((1, H), DT)
            T.copy(Tbl[Ids[0], 0:H], buf)
            T.copy(buf, O)

    return _compile(main)


# --------------------------------------------------------------------------- #
# norms
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=None)
def rms_norm(H: int, eps: float):
    """``xn = bf16(x * rsqrt(mean(x^2)+eps)) * gamma``, one row, one block.

    Qwen3RMSNorm rounds to bf16 *before* the learned scale multiplies, so the
    cast sits inside the product -- spelled out rather than `tf.rms_norm`,
    which would stay f32 through the multiply. The result lands in fp16: the
    projections that consume it run as fp16 vector GEMVs, and fp16 carries
    *more* mantissa than the bf16 the reference would round to.
    """
    @T.prim_func
    def main(X: T.Tensor((H,), DT), G: T.Tensor((H,), DT), Xn: T.Tensor((H,), F16)):
        with T.Kernel(1, is_npu=True) as (bn, _):
            x = T.alloc_shared((1, H), DT)
            g = T.alloc_shared((1, H), DT)
            x32 = T.alloc_shared((1, H), F32)
            sq = T.alloc_shared((1, H), F32)
            tot = T.alloc_shared((1, 1), F32)
            rr = T.alloc_shared((1, 1), F32)
            one = T.alloc_shared((1, 1), F32)
            rinv = T.alloc_shared((1, 1), F32)
            y = T.alloc_shared((1, H), F16)
            g32 = T.alloc_shared((1, H), F32)
            n32 = T.alloc_shared((1, H), F32)
            nb = T.alloc_shared((1, H), DT)
            nb32 = T.alloc_shared((1, H), F32)
            rinvb = T.alloc_shared((1, H), F32)
            T.copy(X, x)
            T.copy(G, g)
            # every row op is a buffer intrinsic: an elementwise loop with
            # dtype casts in it runs at scalar-load speed on this toolchain
            T.vcast(x, x32, round_mode="rint")
            T.vmul(x32, x32, sq)
            T.reduce_sum(sq, tot, dim=1, clear=True)
            for i0 in T.Parallel(1):
                rr[i0, 0] = tot[i0, 0] / float(H) + eps
            T.vsqrt(rr, rr)
            T.vbrc(T.cast(1.0, F32), one)
            T.vdiv(one, rr, rinv)
            T.vbrc(rinv[0, 0], rinvb)
            T.vmul(x32, rinvb, n32)
            T.vcast(n32, nb, round_mode="rint")
            T.vcast(nb, nb32, round_mode="rint")
            T.vcast(g, g32, round_mode="rint")
            T.vmul(nb32, g32, n32)
            # land in bf16 first -- the rounding the reference's bf16
            # activations have -- then carry it in f16 (an exact widening
            # through f32, since vcast has no direct bf16->f16)
            T.vcast(n32, nb, round_mode="rint")
            T.vcast(nb, nb32, round_mode="rint")
            T.vcast(nb32, y, round_mode="rint")
            T.copy(y, Xn)

    return _compile(main)


@lru_cache(maxsize=None)
def resid_rms_norm(H: int, SK: int, eps: float):
    """``h1 = x + bf16(sum(P))``; ``xn = bf16(h1 * rsqrt(..)) * gamma``.

    The residual add and the following norm read the same vector, so they are
    one kernel; *P* is the producer GEMV's split-K partial, reduced here
    rather than by a kernel of its own.
    """
    @T.prim_func
    def main(
        A: T.Tensor((H,), DT),
        P: T.Tensor((SK, H), F32),
        G: T.Tensor((H,), DT),
        Hout: T.Tensor((H,), DT),
        Xn: T.Tensor((H,), F16),
    ):
        with T.Kernel(1, is_npu=True) as (bn, _):
            a = T.alloc_shared((1, H), DT)
            g = T.alloc_shared((1, H), DT)
            s = T.alloc_shared((1, H), F32)
            sl = T.alloc_shared((1, H), F32)
            h1 = T.alloc_shared((1, H), DT)
            h32 = T.alloc_shared((1, H), F32)
            sq = T.alloc_shared((1, H), F32)
            tot = T.alloc_shared((1, 1), F32)
            rr = T.alloc_shared((1, 1), F32)
            one = T.alloc_shared((1, 1), F32)
            rinv = T.alloc_shared((1, 1), F32)
            y = T.alloc_shared((1, H), F16)
            g32 = T.alloc_shared((1, H), F32)
            a32 = T.alloc_shared((1, H), F32)
            sb = T.alloc_shared((1, H), DT)
            sb32 = T.alloc_shared((1, H), F32)
            hpre = T.alloc_shared((1, H), F32)
            n32 = T.alloc_shared((1, H), F32)
            nb = T.alloc_shared((1, H), DT)
            nb32 = T.alloc_shared((1, H), F32)
            rinvb = T.alloc_shared((1, H), F32)
            T.copy(A, a)
            T.copy(G, g)
            T.vbrc(T.cast(0.0, F32), s)
            # the partial reduce goes through row-slice copies: an
            # elementwise 2D gather of P runs at scalar-load speed
            for k in T.serial(SK):
                T.copy(P[k, 0:H], sl)
                T.vadd(s, sl, s)
            # h1 = bf16(f32(A) + f32(bf16(sum))); every step a buffer op
            T.vcast(s, sb, round_mode="rint")
            T.vcast(sb, sb32, round_mode="rint")
            T.vcast(a, a32, round_mode="rint")
            T.vadd(a32, sb32, hpre)
            T.vcast(hpre, h1, round_mode="rint")
            T.copy(h1, Hout)
            T.vcast(h1, h32, round_mode="rint")
            T.vmul(h32, h32, sq)
            T.reduce_sum(sq, tot, dim=1, clear=True)
            for i0 in T.Parallel(1):
                rr[i0, 0] = tot[i0, 0] / float(H) + eps
            T.vsqrt(rr, rr)
            T.vbrc(T.cast(1.0, F32), one)
            T.vdiv(one, rr, rinv)
            T.vbrc(rinv[0, 0], rinvb)
            T.vmul(h32, rinvb, n32)
            T.vcast(n32, nb, round_mode="rint")
            T.vcast(nb, nb32, round_mode="rint")
            T.vcast(g, g32, round_mode="rint")
            T.vmul(nb32, g32, n32)
            T.vcast(n32, nb, round_mode="rint")
            T.vcast(nb, nb32, round_mode="rint")
            T.vcast(nb32, y, round_mode="rint")
            T.copy(y, Xn)

    return _compile(main)


# --------------------------------------------------------------------------- #
# GEMV
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=None)
def gemv(K: int, N: int, BN: int, BK: int, SK: int, NC: int, stages: int = 2):
    """``P[s, n] = sum(X[k] * W[n, k])``; W is stored (N, K) -- the HF layout.

    Grid is a fixed ``NC`` cores walking ``(N/BN) * SK`` tiles serially; the
    product runs in fp16 on the vector units (weights converted at load) and
    the split partial lands in f32 for the consumer to reduce. Split *s*'s
    share of K is ``K/SK``.
    """
    NT = (N // BN) * SK
    KS = K // SK

    @T.prim_func
    def main(
        X: T.Tensor((K,), F16),
        W: T.Tensor((N, K), F16),
        P: T.Tensor((SK * N,), F32),
    ):
        with T.Kernel(NC, is_npu=True) as (cid, _):
            for t in T.serial(T.ceildiv(NT, NC)):
                tid = t * NC + cid
                if tid < NT:
                    # buffers scoped to the tile: carried across the serial
                    # walk, a later tile's loads read a stale slot
                    W16 = T.alloc_shared((BN, BK), F16)
                    X16 = T.alloc_shared((1, BK), F16)
                    prod = T.alloc_shared((BN, BK), F16)
                    part = T.alloc_shared((BN, 1), F16)
                    part32 = T.alloc_shared((BN, 1), F32)
                    acc = T.alloc_shared((BN, 1), F32)
                    bx = tid // SK
                    bs = tid % SK
                    T.vbrc(T.cast(0.0, F32), acc)
                    for ko in T.Pipelined(KS // BK, num_stages=stages):
                        k0 = bs * KS + ko * BK
                        T.copy(W[bx * BN, k0], W16)
                        T.copy(X[k0:k0 + BK], X16)
                        T.vmul(W16, X16, prod)
                        T.reduce_sum(prod, part, dim=1, clear=True)
                        T.vcast(part, part32, round_mode="rint")
                        T.vadd(acc, part32, acc)
                    T.copy(
                        acc[:, 0], P[bs * N + bx * BN:bs * N + bx * BN + BN]
                    )

    return _compile(main)


# --------------------------------------------------------------------------- #
# q/k norm + rope + cache write
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=None)
def qk_rope_cache(HQ: int, HKV: int, D: int, MP: int, CAP: int, SK: int, eps: float):
    """Reduce the fused QKV partials, then per-head norm, rope, cache write.

    *P* is laid out ``[q | k | v]`` over its N axis. Query heads write the
    zero-padded per-group buffer *Qp* ``(HKV, 16, D)`` (rows ``G..15`` stay
    zero forever, which is what the score GEMM's M=16 padding needs); key and
    value heads stop here, written straight into the cache at ``Wr`` -- no
    caller appends anything and the cache buffer never moves.

    The partial reduction reads P through row-slice copies at dynamic
    offsets: an elementwise gather with a block-derived index miscompiles on
    this toolchain, a `T.copy` region read does not.
    """
    G = HQ // HKV
    QN = HQ * D
    KN = HKV * D
    TOT = HQ + 2 * HKV
    HALF = D // 2
    NC = QN + 2 * KN

    @T.prim_func
    def main(
        P: T.Tensor((SK, NC), F32),
        Gq: T.Tensor((D,), DT),
        Gk: T.Tensor((D,), DT),
        Cos: T.Tensor((MP, D), DT),
        Sin: T.Tensor((MP, D), DT),
        Pos: T.Tensor((1,), "int32"),
        Wr: T.Tensor((1,), "int32"),
        Kc: T.Tensor((CAP, KN), DT),
        Vc: T.Tensor((CAP, KN), DT),
        Qp: T.Tensor((HKV, 16, D), DT),
    ):
        with T.Kernel(TOT, is_npu=True) as (bh, _):
            acc = T.alloc_shared((1, D), F32)
            sl = T.alloc_shared((1, D), F32)
            sq = T.alloc_shared((1, D), F32)
            tot = T.alloc_shared((1, 1), F32)
            rr = T.alloc_shared((1, 1), F32)
            one = T.alloc_shared((1, 1), F32)
            rinv = T.alloc_shared((1, 1), F32)
            nrm = T.alloc_shared((1, D), DT)
            rot_s = T.alloc_shared((1, D), DT)
            cs = T.alloc_shared((1, D), DT)
            sn = T.alloc_shared((1, D), DT)
            gq = T.alloc_shared((1, D), DT)
            gk = T.alloc_shared((1, D), DT)
            acc32 = T.alloc_shared((1, D), F32)
            gq32 = T.alloc_shared((1, D), F32)
            gk32 = T.alloc_shared((1, D), F32)
            n2 = T.alloc_shared((1, D), F32)
            rinvb = T.alloc_shared((1, D), F32)
            x1 = T.alloc_shared((1, D // 2), F32)
            x2 = T.alloc_shared((1, D // 2), F32)
            c1 = T.alloc_shared((1, D // 2), F32)
            c2 = T.alloc_shared((1, D // 2), F32)
            s1 = T.alloc_shared((1, D // 2), F32)
            s2 = T.alloc_shared((1, D // 2), F32)
            t1 = T.alloc_shared((1, D // 2), F32)
            t2 = T.alloc_shared((1, D // 2), F32)
            r1 = T.alloc_shared((1, D // 2), F32)
            rb = T.alloc_shared((1, D // 2), DT)
            c32 = T.alloc_shared((1, D), F32)
            s32 = T.alloc_shared((1, D), F32)
            base = T.alloc_shared((1,), "int32")
            with T.If(bh < HQ):
                with T.Then():
                    base[0] = bh * D
                with T.Else():
                    base[0] = QN + (bh - HQ) * D
            T.copy(Gq, gq)
            T.copy(Gk, gk)
            with T.If(bh >= HQ + HKV):
                # ---- v head: reduce, round, write straight to the value cache
                with T.Then():
                    vbase = QN + KN + (bh - HQ - HKV) * D
                    T.vbrc(T.cast(0.0, F32), acc)
                    for s in T.serial(SK):
                        T.copy(P[s, vbase:vbase + D], sl)
                        T.vadd(acc, sl, acc)
                    T.vcast(acc, rot_s, round_mode="rint")
                    T.copy(rot_s, Vc[Wr[0], (bh - HQ - HKV) * D])
                # ---- q / k head: reduce, per-head norm, rope
                with T.Else():
                    T.vbrc(T.cast(0.0, F32), acc)
                    for s in T.serial(SK):
                        T.copy(P[s, base[0]:base[0] + D], sl)
                        T.vadd(acc, sl, acc)
                    T.vmul(acc, acc, sq)
                    T.reduce_sum(sq, tot, dim=1, clear=True)
                    for d0 in T.Parallel(1):
                        rr[d0, 0] = tot[d0, 0] / float(D) + eps
                    T.vsqrt(rr, rr)
                    T.vbrc(T.cast(1.0, F32), one)
                    T.vdiv(one, rr, rinv)
                    T.vbrc(rinv[0, 0], rinvb)
                    T.vmul(acc, rinvb, acc32)
                    T.vcast(acc32, nrm, round_mode="rint")
                    T.copy(Cos[Pos[0], 0:D], cs)
                    T.copy(Sin[Pos[0], 0:D], sn)
                    T.vcast(cs, c32, round_mode="rint")
                    T.vcast(sn, s32, round_mode="rint")
                    # gamma applies before rope; the gamma'd vector and the
                    # whole rope are duplicated per branch -- a value written
                    # under T.If does not survive the branch
                    T.vcast(gq, gq32, round_mode="rint")
                    T.vcast(gk, gk32, round_mode="rint")
                    T.vcast(nrm, n2, round_mode="rint")
                    with T.If(bh < HQ):
                        with T.Then():
                            T.vmul(n2, gq32, n2)
                        with T.Else():
                            T.vmul(n2, gk32, n2)
                    T.vcast(n2, nrm, round_mode="rint")
                    T.vcast(nrm, n2, round_mode="rint")
                    # rope pairs element d with d +/- HALF: stage the halves
                    # out and back through row slices
                    T.copy(n2[0:1, 0:HALF], x1)
                    T.copy(n2[0:1, HALF:D], x2)
                    T.copy(c32[0:1, 0:HALF], c1)
                    T.copy(c32[0:1, HALF:D], c2)
                    T.copy(s32[0:1, 0:HALF], s1)
                    T.copy(s32[0:1, HALF:D], s2)
                    T.vmul(x1, c1, t1)
                    T.vmul(x2, s1, t2)
                    T.vsub(t1, t2, r1)
                    T.vcast(r1, rb, round_mode="rint")
                    T.copy(rb, rot_s[0:1, 0:HALF])
                    T.vmul(x2, c2, t1)
                    T.vmul(x1, s2, t2)
                    T.vadd(t1, t2, r1)
                    T.vcast(r1, rb, round_mode="rint")
                    T.copy(rb, rot_s[0:1, HALF:D])
                    with T.If(bh < HQ):
                        with T.Then():
                            T.copy(rot_s, Qp[bh // G, bh % G, 0])
                        with T.Else():
                            T.copy(rot_s, Kc[Wr[0], (bh - HQ) * D])

    return _compile(main)


# --------------------------------------------------------------------------- #
# attention
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=None)
def attn_partial(HQ: int, HKV: int, D: int, CAP: int, SS: int, NC: int, scale: float):
    """One split's ``(max, sum, weighted values)`` over its slice of context.

    Grid is a fixed ``NC`` cores walking the ``NS * HKV`` (split, kv-head)
    pairs serially; one block serves a kv head's whole GQA group, so a key is
    read once per split. Structured after the shipped Developer-mode
    flash-attention example -- cube results and softmax state in fragments,
    staging copies in shared. There is no branch and no ``vcmp/vselect`` after
    the cube: positions past the current length are gated arithmetically
    (``sc*gate - (1-gate)*BIG``), so a dead split's partial comes out
    ``(NEG, 0, 0)`` on its own. Partials are split-major and flat; the merge
    in ``o_proj`` reads them with computed 1D offsets, and the stores go
    through staged shared rows so every global write is a 1D copy.
    """
    G = HQ // HKV
    NS = CAP // SS
    MP = 16

    @T.prim_func
    def main(
        Qp: T.Tensor((HKV, 16, D), DT),
        Kc: T.Tensor((CAP, HKV * D), DT),
        Vc: T.Tensor((CAP, HKV * D), DT),
        Pos: T.Tensor((1,), "int32"),
        Op: T.Tensor((HQ, NS, D), F32),
        Mp: T.Tensor((NS * HQ,), F32),
        Lp: T.Tensor((NS * HQ,), F32),
    ):
        with T.Kernel(NC, is_npu=True) as (cid, _):
            for t in T.serial(T.ceildiv(NS * HKV, NC)):
                bid = t * NC + cid
                if bid < NS * HKV:
                    bs = bid // HKV
                    bh = bid % HKV
                    Qs = T.alloc_shared((16, D), DT)
                    Ks = T.alloc_shared((SS, D), DT)
                    Vs = T.alloc_shared((SS, D), DT)
                    sc = T.alloc_fragment((MP, SS), F32)
                    scv = T.alloc_fragment((MP, SS), F32)
                    p = T.alloc_fragment((MP, SS), F32)
                    pb = T.alloc_fragment((MP, SS), DT)
                    orun = T.alloc_fragment((MP, D), F32)
                    o_sh = T.alloc_shared((MP, D), F32)
                    mx = T.alloc_fragment((MP, 1), F32)
                    mxb = T.alloc_fragment((MP, SS), F32)
                    sm = T.alloc_fragment((MP, 1), F32)
                    iota = T.alloc_fragment((1, SS), F32)
                    iota_b = T.alloc_fragment((MP, SS), F32)
                    lim_b = T.alloc_fragment((MP, SS), F32)
                    gate = T.alloc_fragment((MP, SS), F32)
                    inv_g = T.alloc_fragment((MP, SS), F32)
                    bigt = T.alloc_fragment((MP, SS), F32)
                    sct = T.alloc_fragment((MP, SS), F32)
                    one_t = T.alloc_fragment((MP, SS), F32)
                    zero_t = T.alloc_fragment((MP, SS), F32)
                    m_sh = T.alloc_shared((1, G), F32)
                    l_sh = T.alloc_shared((1, G), F32)
                    pos = T.alloc_shared((1,), "int32")
                    limf = T.alloc_shared((1, 1), F32)
                    T.vbrc(T.cast(1.0, F32), one_t)
                    T.vbrc(T.cast(0.0, F32), zero_t)
                    T.vbrc(T.cast(1.0e30, F32), bigt)
                    T.copy(Pos, pos)
                    T.copy(Qp[bh, 0, 0], Qs)
                    T.copy(Kc[bs * SS, bh * D], Ks)
                    T.copy(Vc[bs * SS, bh * D], Vs)
                    # the score row; the 1/sqrt(D) factor on the finished f32 dot,
                    # not on q in bf16 first (what HF does, and what the authored
                    # reference's bf16 pre-scale would round away)
                    T.gemm(Qs, Ks, sc, initC=True, b_transpose=True)
                    T.vbrc(T.cast(scale, F32), sct)
                    T.vmul(sc, sct, scv)
                    # gate = clamp(lim - iota, 0, 1): 1 inside the context, 0 past it
                    T.arange(iota, strides=[1, 1], offset=0)
                    T.vbrc(iota, iota_b)
                    limf[0, 0] = T.cast(pos[0] + 1 - bs * SS, F32)
                    T.vbrc(limf[0, 0], lim_b)
                    T.vsub(lim_b, iota_b, gate)
                    T.vmin(gate, one_t, gate)
                    T.vmax(gate, zero_t, gate)
                    # masked score = scv*gate - (1-gate)*BIG: exact where valid
                    T.vsub(one_t, gate, inv_g)
                    T.vmul(inv_g, bigt, inv_g)
                    T.vmul(scv, gate, sc)
                    T.vsub(sc, inv_g, sc)
                    # softmax over what is left
                    T.reduce_max(sc, mx, dim=1)
                    T.vbrc(mx, mxb)
                    T.vsub(sc, mxb, p)
                    T.vexp(p, p)
                    T.vmul(p, gate, p)
                    T.reduce_sum(p, sm, dim=1)
                    # bf16 probabilities into the V product, as HF does
                    T.vcast(p, pb, round_mode="rint")
                    T.gemm(pb, Vs, orun, initC=True)
                    # stores: stage through shared, write with 1D copies
                    T.copy(orun, o_sh)
                    for g in T.Parallel(G):
                        m_sh[0, g] = mx[g, 0]
                        l_sh[0, g] = sm[g, 0]
                    for g in T.serial(G):
                        T.copy(o_sh[g, 0:D], Op[bh * G + g, bs, 0])
                    T.copy(m_sh, Mp[bs * HQ + bh * G:bs * HQ + bh * G + G])
                    T.copy(l_sh, Lp[bs * HQ + bh * G:bs * HQ + bh * G + G])

    return _compile(main)


# --------------------------------------------------------------------------- #
# GEMV with a folded producer
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=None)
def gemv_attn_combine(
    HQ: int, D: int, N: int, BN: int, BK: int, SK: int, NS: int, NC: int,
    stages: int = 2,
):
    """``o_proj``, with the attention splits merged into its own input read.

    A block owning ``K/SK`` inputs owns a whole number of heads, so it merges
    just those heads' partials itself (log-sum-exp against the joint max) and
    then streams its weight rows -- the merge never becomes a launch. The
    merged inputs are staged through a small GM scratch so the pipelined
    loads read global memory exactly the way the plain `gemv` does.
    """
    K = HQ * D
    KS = K // SK
    HH = KS // D
    NT = (N // BN) * SK

    @T.prim_func
    def main(
        Op: T.Tensor((HQ, NS, D), F32),
        Mp: T.Tensor((NS * HQ,), F32),
        Lp: T.Tensor((NS * HQ,), F32),
        W: T.Tensor((N, K), F16),
        Xg: T.Tensor((K,), F16),
        P: T.Tensor((SK * N,), F32),
    ):
        with T.Kernel(NC, is_npu=True) as (cid, _):
            mps = T.alloc_fragment((NS, 1), F32)
            lps = T.alloc_fragment((NS, 1), F32)
            mx = T.alloc_fragment((1, 1), F32)
            mpsb = T.alloc_fragment((NS, D), F32)
            lpsb = T.alloc_fragment((NS, D), F32)
            mxb = T.alloc_fragment((NS, D), F32)
            eb = T.alloc_fragment((NS, D), F32)
            ops = T.alloc_fragment((NS, D), F32)
            wt = T.alloc_fragment((NS, D), F32)
            den = T.alloc_fragment((1, D), F32)
            num = T.alloc_fragment((1, D), F32)
            xh = T.alloc_shared((1, D), F16)
            xhb = T.alloc_shared((1, D), DT)
            for t in T.serial(T.ceildiv(NT, NC)):
                tid = t * NC + cid
                if tid < NT:
                    W16 = T.alloc_shared((BN, BK), F16)
                    X16 = T.alloc_shared((1, BK), F16)
                    prod = T.alloc_shared((BN, BK), F16)
                    part = T.alloc_shared((BN, 1), F16)
                    part32 = T.alloc_shared((BN, 1), F32)
                    acc = T.alloc_shared((BN, 1), F32)
                    bx = tid // SK
                    bs = tid % SK
                    # merge this split's HH heads into the GM scratch slice
                    for hh in T.serial(HH):
                        h = bs * HH + hh
                        for s, d0 in T.Parallel(NS, 1):
                            mps[s, d0] = Mp[s * HQ + h]
                            lps[s, d0] = Lp[s * HQ + h]
                        T.reduce_max(mps, mx, dim=0, clear=True)
                        T.vbrc(mps, mpsb)
                        T.vbrc(lps, lpsb)
                        T.vbrc(mx[0, 0], mxb)
                        T.vsub(mpsb, mxb, eb)
                        T.vexp(eb, eb)
                        T.copy(Op[h, 0, 0], ops)
                        T.vmul(eb, lpsb, wt)
                        T.reduce_sum(wt, den, dim=0, clear=True)
                        T.vmul(ops, eb, wt)
                        T.reduce_sum(wt, num, dim=0, clear=True)
                        T.vdiv(num, den, num)
                        T.vcast(num, xhb, round_mode="rint")
                        T.vcast(xhb, num, round_mode="rint")
                        T.vcast(num, xh, round_mode="rint")
                        T.copy(xh, Xg[bs * KS + hh * D:bs * KS + hh * D + D])
                    # the GEMV itself, streaming the scratch like any input
                    T.vbrc(T.cast(0.0, F32), acc)
                    for ko in T.Pipelined(KS // BK, num_stages=stages):
                        k0 = bs * KS + ko * BK
                        T.copy(W[bx * BN, k0], W16)
                        T.copy(Xg[k0:k0 + BK], X16)
                        T.vmul(W16, X16, prod)
                        T.reduce_sum(prod, part, dim=1, clear=True)
                        T.vcast(part, part32, round_mode="rint")
                        T.vadd(acc, part32, acc)
                    T.copy(
                        acc[:, 0], P[bs * N + bx * BN:bs * N + bx * BN + BN]
                    )

    return _compile(main)


@lru_cache(maxsize=None)
def gemv_silu(
    I: int, N: int, BN: int, BK: int, SK: int, SKG: int, NC: int,
    stages: int = 2,
):
    """``down_proj``, with ``silu(gate) * up`` folded into its own input read.

    *GU* is the fused gate/up GEMV's partial, gate in ``[0, I)`` and up in
    ``[I, 2I)``; a block reduces and activates only the ``I/SK`` entries it
    walks, so the intermediate never reaches HBM. The activated slice lands
    in a GM scratch the pipelined loads stream from, as in
    `gemv_attn_combine`.
    """
    KS = I // SK
    NT = (N // BN) * SK

    @T.prim_func
    def main(
        GU: T.Tensor((SKG, 2 * I), F32),
        W: T.Tensor((N, I), F16),
        Xg: T.Tensor((I,), F16),
        P: T.Tensor((SK * N,), F32),
    ):
        with T.Kernel(NC, is_npu=True) as (cid, _):
            g = T.alloc_shared((1, KS), F32)
            u = T.alloc_shared((1, KS), F32)
            gl = T.alloc_shared((1, KS), F32)
            ul = T.alloc_shared((1, KS), F32)
            xs = T.alloc_shared((1, KS), F16)
            gb = T.alloc_shared((1, KS), DT)
            gb32 = T.alloc_shared((1, KS), F32)
            ub = T.alloc_shared((1, KS), DT)
            ub32 = T.alloc_shared((1, KS), F32)
            neg = T.alloc_shared((1, KS), F32)
            e = T.alloc_shared((1, KS), F32)
            den = T.alloc_shared((1, KS), F32)
            act = T.alloc_shared((1, KS), F32)
            actb = T.alloc_shared((1, KS), DT)
            actb32 = T.alloc_shared((1, KS), F32)
            y32 = T.alloc_shared((1, KS), F32)
            xb = T.alloc_shared((1, KS), DT)
            one_k = T.alloc_shared((1, KS), F32)
            zero_k = T.alloc_shared((1, KS), F32)
            for t in T.serial(T.ceildiv(NT, NC)):
                tid = t * NC + cid
                if tid < NT:
                    W16 = T.alloc_shared((BN, BK), F16)
                    X16 = T.alloc_shared((1, BK), F16)
                    prod = T.alloc_shared((BN, BK), F16)
                    part = T.alloc_shared((BN, 1), F16)
                    part32 = T.alloc_shared((BN, 1), F32)
                    acc = T.alloc_shared((BN, 1), F32)
                    bx = tid // SK
                    bs = tid % SK
                    T.vbrc(T.cast(0.0, F32), g)
                    T.vbrc(T.cast(0.0, F32), u)
                    # row-slice copies again: an elementwise computed-index
                    # gather of GU runs at scalar-load speed
                    for s in T.serial(SKG):
                        T.copy(GU[s, bs * KS:bs * KS + KS], gl)
                        T.copy(GU[s, I + bs * KS:I + bs * KS + KS], ul)
                        T.vadd(g, gl, g)
                        T.vadd(u, ul, u)
                    # silu lands in bf16 first, then the product -- the
                    # rounding points the reference's bf16 ops have; every
                    # step a buffer intrinsic, elementwise casts crawl
                    T.vcast(g, gb, round_mode="rint")
                    T.vcast(gb, gb32, round_mode="rint")
                    T.vcast(u, ub, round_mode="rint")
                    T.vcast(ub, ub32, round_mode="rint")
                    T.vbrc(T.cast(0.0, F32), zero_k)
                    T.vsub(zero_k, gb32, neg)
                    T.vexp(neg, e)
                    T.vbrc(T.cast(1.0, F32), one_k)
                    T.vadd(one_k, e, den)
                    T.vdiv(gb32, den, act)
                    T.vcast(act, actb, round_mode="rint")
                    T.vcast(actb, actb32, round_mode="rint")
                    T.vmul(actb32, ub32, y32)
                    T.vcast(y32, xb, round_mode="rint")
                    T.vcast(xb, y32, round_mode="rint")
                    T.vcast(y32, xs, round_mode="rint")
                    T.copy(xs, Xg[bs * KS:bs * KS + KS])
                    T.vbrc(T.cast(0.0, F32), acc)
                    for ko in T.Pipelined(KS // BK, num_stages=stages):
                        k0 = bs * KS + ko * BK
                        T.copy(W[bx * BN, k0], W16)
                        T.copy(Xg[k0:k0 + BK], X16)
                        T.vmul(W16, X16, prod)
                        T.reduce_sum(prod, part, dim=1, clear=True)
                        T.vcast(part, part32, round_mode="rint")
                        T.vadd(acc, part32, acc)
                    T.copy(
                        acc[:, 0], P[bs * N + bx * BN:bs * N + bx * BN + BN]
                    )

    return _compile(main)


# --------------------------------------------------------------------------- #
# head + sampling
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=None)
def lm_head(K: int, N: int, BN: int, BK: int, NC: int, stages: int = 2):
    """The head projection, on a fixed core count; logits only.

    The argmax lives in `argmax_stage` -- a reduce chain after a cube result
    inside the serial tile loop hits the toolchain's auto-multi-buffer
    stale-read bug, so the two are separate launches and the head is a plain
    vector GEMV.
    """
    NB = N // BN

    @T.prim_func
    def main(
        X: T.Tensor((K,), F16),
        W: T.Tensor((N, K), F16),
        O: T.Tensor((N,), F32),
    ):
        with T.Kernel(NC, is_npu=True) as (cid, _):
            for t in T.serial(T.ceildiv(NB, NC)):
                bx = t * NC + cid
                if bx < NB:
                    W16 = T.alloc_shared((BN, BK), F16)
                    X16 = T.alloc_shared((1, BK), F16)
                    prod = T.alloc_shared((BN, BK), F16)
                    part = T.alloc_shared((BN, 1), F16)
                    part32 = T.alloc_shared((BN, 1), F32)
                    acc = T.alloc_shared((BN, 1), F32)
                    T.vbrc(T.cast(0.0, F32), acc)
                    for ko in T.Pipelined(K // BK, num_stages=stages):
                        T.copy(W[bx * BN, ko * BK], W16)
                        T.copy(X[ko * BK:(ko + 1) * BK], X16)
                        T.vmul(W16, X16, prod)
                        T.reduce_sum(prod, part, dim=1, clear=True)
                        T.vcast(part, part32, round_mode="rint")
                        T.vadd(acc, part32, acc)
                    T.copy(acc[:, 0], O[bx * BN:bx * BN + BN])

    return _compile(main), NB


@lru_cache(maxsize=None)
def argmax_stage(V: int, BN: int, NC: int):
    """Per-block best ``(value, first index)`` over the logits.

    One tile per block -- the shipped argreduce C1 shape: copy, reduce,
    broadcast, compare, select over an iota, min-reduce for the first index.
    The logits buffer is ``(1, PADV)`` with a ``-inf`` tail, so every block's
    tile is full and the tail never wins. Outputs land through one-element
    region copies: scalar scatters of reduce results read stale.
    """
    NB = (V + BN - 1) // BN
    PADV = NB * BN

    @T.prim_func
    def main(
        L: T.Tensor((1, PADV), F32),
        Bv: T.Tensor((1, NB), F32),
        Bi: T.Tensor((NB, 1), "int32"),
        Flush: T.Tensor((NB, BN), F32),
    ):
        with T.Kernel(NB, is_npu=True) as (bid, _):
            f = T.alloc_shared((1, BN), F32)
            mx = T.alloc_shared((1, 1), F32)
            mxb = T.alloc_shared((1, BN), F32)
            iota = T.alloc_shared((1, BN), F32)
            inf = T.alloc_shared((1, BN), F32)
            sel = T.alloc_shared((1, BN), F32)
            first = T.alloc_shared((1, 1), F32)
            mask = T.alloc_shared((1, BN), "bool")
            T.arange(iota, strides=[1, 1], offset=0)
            T.vbrc(T.cast(T.infinity(F32), F32), inf)
            T.copy(L[0:1, bid * BN:bid * BN + BN], f)
            # a load at a dynamic origin followed directly by a reduce
            # reads a stale slot on this toolchain; touching the loaded tile
            # with a GM write lands it (the shipped argreduce's debug path
            # does the same by accident)
            T.copy(f, Flush[bid, 0:BN])
            T.reduce_max(f, mx, dim=1, clear=True)
            T.vbrc(mx[0, 0], mxb)
            T.vcmp(f, mxb, mask, "eq")
            T.vselect(mask, iota, inf, sel)
            T.reduce(sel, first, dims=[1], reduce_mode="min", clear=True)
            T.copy(mx, Bv[0:1, bid:bid + 1])
            bi32 = T.alloc_shared((1, 1), "int32")
            bi32[0, 0] = bid * BN + T.cast(first[0, 0], "int32")
            T.copy(bi32, Bi[bid, 0])

    return _compile(main), NB, PADV


@lru_cache(maxsize=None)
def sample_step(NB: int, PAD: int, NSTEPS: int):
    """Finish the greedy pick, record it, hand the next input on, advance pos.

    The graph's last node and the only one that decides anything: while the
    prompt still has a token left it feeds that, otherwise what it just
    sampled. That one device-side choice lets a single capture walk the prompt
    and continue past it with no host round trip between steps.
    """

    @T.prim_func
    def main(
        Bv: T.Tensor((1, NB), F32),
        Bi: T.Tensor((NB, 1), "int32"),
        Inp: T.Tensor((NSTEPS,), "int32"),
        Plen: T.Tensor((1,), "int32"),
        Ids: T.Tensor((1,), "int32"),
        Pos: T.Tensor((1,), "int32"),
        Sam: T.Tensor((NSTEPS,), "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (bn, _):
            f = T.alloc_shared((1, PAD), F32)
            mx = T.alloc_shared((1, 1), F32)
            mxb = T.alloc_shared((1, PAD), F32)
            iota = T.alloc_shared((1, PAD), F32)
            inf = T.alloc_shared((1, PAD), F32)
            sel = T.alloc_shared((1, PAD), F32)
            win = T.alloc_shared((1, 1), F32)
            mask = T.alloc_shared((1, PAD), "bool")
            w32 = T.alloc_shared((1,), "int32")
            best = T.alloc_shared((1,), "int32")
            nxt = T.alloc_shared((1,), "int32")
            p = T.alloc_shared((1,), "int32")
            T.copy(Bv, f[0:1, 0:NB])
            for i in T.Parallel(PAD - NB):
                f[0, NB + i] = NEG
            T.reduce_max(f, mx, dim=1, clear=True)
            T.vbrc(mx[0, 0], mxb)
            T.vcmp(f, mxb, mask, "eq")
            T.arange(iota, strides=[1, 1], offset=0)
            T.vbrc(T.cast(T.infinity(F32), F32), inf)
            T.vselect(mask, iota, inf, sel)
            T.reduce(sel, win, dims=[1], reduce_mode="min", clear=True)
            w32[0] = T.cast(win[0, 0], "int32")
            best[0] = Bi[w32[0], 0]
            T.copy(Pos, p)
            Sam[p[0]] = best[0]
            with T.If(p[0] + 1 < Plen[0]):
                with T.Then():
                    nxt[0] = Inp[p[0] + 1]
                with T.Else():
                    nxt[0] = best[0]
            Ids[0] = nxt[0]
            Pos[0] = p[0] + 1

    return _compile(main)
