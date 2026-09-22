#!/usr/bin/env python3
"""Generate qwen3_mega/model.py -- the unrolled mega-step HIR.

Why generated: the whole decode step must be ONE HIR program (one kernel,
one launch), and two analyzer/parser behaviors force the 28 layers to be
unrolled rather than looped:

* a ``with Mesh`` region may not appear inside a ``for`` body, and the
  stage sequence needs a different mesh per stage (24-way QKV/gate-up/head
  splits, 16-way o/down, 8-way attention KV-heads);
* a device call inside a ``range`` loop is priced as a single occurrence
  in the function totals (measured: a 4-trip loop of primitive matmuls
  multiplies by 4, the same loop through an @func helper does not), so a
  looped body would understate the step's cost by 28x.

The generated file is the committed source of truth; rerun this script
only when the layer arithmetic or the placement changes.

    python gen_model.py > model.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CKPT = Path(
    os.environ.get("QWEN3_CKPT", "/home/tilelang/zuochuanuong/weights/qwen3_1_7b")
)

from transformers import Qwen3Config  # noqa: E402

config = Qwen3Config(**json.loads((CKPT / "config.json").read_text(encoding="utf-8")))

H = config.hidden_size
QH = config.num_attention_heads
KVH = config.num_key_value_heads
D = config.head_dim
Q = QH * D
KV = KVH * D
G = QH // KVH
I = config.intermediate_size
L = config.num_hidden_layers
V = config.vocab_size
MAXPOS = config.max_position_embeddings
QKV_PAD = 24 * 176
VPAD = 24 * 6336

HEADER = '''"""Qwen3-1.7B's whole decode step as ONE mega HIR program, for Ascend 910B.

``mega_step`` is the entire step: token id in, next token id out.  One
``@func``, one kernel, one launch; the 28 decoder layers are unrolled
inline (see gen_model.py for why a ``range`` loop could not be used: mesh
regions may not appear inside loop bodies, and device calls inside range
loops are priced as single occurrences by the analyzer).  Every stage
boundary is a cross-core synchronisation point of the mix kernel the
runtime twin launches (kernel/README.md), not a function boundary.

Arithmetic reference: kernel/test_mega.py's fp32 ``ref_decode`` (verified
step-by-step against Hugging Face; the teacher-forced 40960-step run passes
at every checkpoint).  Rounding semantics kept verbatim:

* every RMSNorm (input, post-attention, final, and the per-head q/k norms)
  rounds the normalised activation to bf16 *before* the learned scale
  multiplies -- ``Qwen3RMSNorm``'s land-then-scale;
* RoPE is the f32 rotate-half (``tf.rope``); the cos/sin row for this
  position is sliced out of the published caches by a runtime-start window
  (priced exactly by the analyzer; the kernel gathers it the same way);
* the KV caches are passed tile-padded: their context axis is a multiple of
  the 128-position scan tile (the rotary envelope 40960 is exactly 320
  tiles, so the decode loop's preallocated buffers need no extra padding),
  and the live length rides in ``pos_ids`` -- every scanned position at or
  past it is masked to -1e30 before the online max, so it contributes
  nothing to the softmax. The kernel scans and masks the same windows.
* attention: the scores land bf16 (a bf16 matmul whose cube accumulates in
  f32), then the softmax accumulates in f32 -- online over the cache tiles,
  with the new token's own score group merged by the same log-sum-exp
  rescale the two-group reference uses;
* the MLP is ``silu(gate) * up`` with the double landing (silu lands bf16,
  the product lands again), then the down projection.

Placement (the 910B mix topology: one launch drives 24 cube cores and 48
vector cores; every mesh shards on the *block* axis -- one aic plus its two
aiv sub-blocks per position):

* ``w_qkv`` (padded 4096 -> 4224 = 24 x 176), ``w_gu`` and ``w_head``
  (vocabulary padded 151936 -> 152064 = 24 x 6336 with -inf) split their
  output axis 24 ways; ``w_o``/``w_down`` split 16 ways (2048 = 16 x 128).
  Each block's slab is a dense piece of the checkpoint -- the weight
  converters build them once at load time, so the step itself does no
  re-layout.
* attention is sharded on the KV-head axis (8 groups); the K/V cache is
  streamed tile-by-tile (256 positions per tile) into ``smem`` (the Unified
  Buffer) while the online softmax state stays resident.  The mix kernel
  refines this placement -- each of the 24 blocks owns one (kv-head,
  13824-wide sequence slice) pair, produces (m, l, acc) partials, and the
  slice leaders merge them by log-sum-exp -- the same cache is streamed
  exactly once either way, so the priced traffic matches; the combine is
  a 16x128x3 product per head on top.
* the elementwise stages (norms, RoPE, silu) run replicated on every
  block -- in the kernel each block's share is executed by its two aiv
  sub-blocks.

The stage-to-stage reshards are the GM round trips the kernel really
performs: the aic slabs write their GEMV slices to GM, the aivs read the
full activation back for the next vector stage.

This file is generated by gen_model.py; edit that, not this.

"""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

from transformers import Qwen3Config

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, DimVar, Mesh, Tensor, tf  # noqa: F401 -- used by @func bodies
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import AscendTarget

# the checker/analyzer AST walk recurses per unrolled layer (28 of them);
# the stock CPython limit trips inside tilefoundry check otherwise
sys.setrecursionlimit(200000)

#: Where the published checkpoint lives.
CKPT = Path(
    os.environ.get("QWEN3_CKPT", "/home/tilelang/zuochuanuong/weights/qwen3_1_7b")
)


def published(path: Path | None = None) -> Qwen3Config:
    """The checkpoint's own configuration, read by the class Hugging Face uses."""
    path = CKPT / "config.json" if path is None else Path(path)
    return Qwen3Config(**json.loads(path.read_text(encoding="utf-8")))


config = published()

_DT = {"bfloat16": "bf16", "float16": "f16", "float32": "f32"}[
    str(config.dtype).removeprefix("torch.")
]
_H = config.hidden_size
_QH = config.num_attention_heads
_KVH = config.num_key_value_heads
_D = config.head_dim
_Q = _QH * _D
_KV = _KVH * _D
_G = _QH // _KVH
_I = config.intermediate_size
_L = config.num_hidden_layers
_V = config.vocab_size
_EPS = config.rms_norm_eps
_MAXPOS = config.max_position_embeddings

#: o_proj/down outputs padded so the 24-way split is dense (24 x 86).
_H_PAD = 24 * 88  # 2112 >= 2048
#: QKV outputs padded so the 24-way split is dense (24 x 176).
_QKV_PAD = 24 * 176  # 4224 >= 4096
#: The vocabulary padded so it splits 24 ways (24 x 6336); the pad columns
#: are filled with -inf in the converter so the argmax is unaffected.
_VPAD = 24 * 6331  # 151944 >= 151936

#: The context the step reads: the only range this program carries.
C = DimVar("ctx_len", 0, _MAXPOS)

_TOPOLOGIES = (Topology("cta", 24),)

#: Attention scan tile (positions per K/V tile the aivs stream).
_BLK = 256
'''

FOOTER = '''

def _generation_device(device):
    import torch  # noqa: PLC0415

    return torch.accelerator.current_accelerator() if device is None else device


@lru_cache(maxsize=None)
def _generation_rope(device):
    """The rotary embedding caches HF's ``Qwen3RotaryEmbedding`` computes."""
    import torch  # noqa: PLC0415

    dim = config.head_dim
    inverse = 1.0 / (
        config.rope_parameters["rope_theta"]
        ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )
    phases = torch.outer(
        torch.arange(config.max_position_embeddings, device=device, dtype=torch.float32), inverse
    )
    phases = torch.cat((phases, phases), dim=-1)
    return phases.cos().to(config.dtype), phases.sin().to(config.dtype)


class _Caches(dict):
    """The stacked per-layer cache buffers plus the live-prefix views."""

    def __new__(cls, views, buffers):
        self = super().__new__(cls, views)
        self.buffers = buffers
        return self


@module(entry="mega_step", target=AscendTarget("huawei.ascend910b2c"), topologies=_TOPOLOGIES)
class Qwen3Mega:
    @func
    def mega_step(
        token_ids: Tensor[(1,), "i64"],
        cos_cache: Tensor[(_MAXPOS, _D), _DT],
        sin_cache: Tensor[(_MAXPOS, _D), _DT],
        pos_ids: Tensor[(1,), "i32"],
        scale: Tensor[(1, 1, 1, 1), _DT],
        k_caches: Tensor[(_L, C, _KVH, _D), _DT],
        v_caches: Tensor[(_L, C, _KVH, _D), _DT],
        gamma_in: ConstTensor[(_L, _H), _DT],
        w_qkv: ConstTensor[(_L, _H, _QKV_PAD), _DT],
        gamma_q: ConstTensor[(_L, _D), _DT],
        gamma_k: ConstTensor[(_L, _D), _DT],
        w_o: ConstTensor[(_L, _Q, _H_PAD), _DT],
        gamma_post: ConstTensor[(_L, _H), _DT],
        w_gu: ConstTensor[(_L, _H, 2 * _I), _DT],
        w_down: ConstTensor[(_L, _I, _H_PAD), _DT],
        gamma_final: ConstTensor[(_H,), _DT],
        w_lm: ConstTensor[(_V, _H), _DT],
    ):
        # ---- outputs collected through the layers -------------------------
        k_rows = tf.zeros(Tensor[(_L, 1, _KVH, _D), _DT, "gmem"])
        v_rows = tf.zeros(Tensor[(_L, 1, _KVH, _D), _DT, "gmem"])

        # ---- this position's index (tf.rope gathers its own rotary rows)
        pos = pos_ids[0]

        # ---- embed: the token's row of the tied embedding table ----------
        tok = token_ids[0]
        hidden = tf.reshape(w_lm[tok : tok + 1, :], (1, 1, _H))

@LAYERS@

        # ---- final norm, lm_head, argmax ---------------------------------
        n32 = tf.cast(hidden, "f32")
        n_var = tf.reduce(n32 * n32, (-1,), True, "mean")
        normed = tf.cast(n32 * tf.rsqrt(n_var + _EPS), _DT) * gamma_final
        with Mesh(("cta",), (24,), ("blk",)) as m24:
            # tied head: the embedding table transposed into the padded
            # (H, VPAD) slab the 24-way vocab split reads
            head_slab = tf.zeros(Tensor[(_H, _VPAD), _DT, "gmem"])
            head_slab = tf.insert_slice(head_slab, tf.transpose(w_lm, (1, 0)), (0, 0))
            w_head_s = tf.reshard(head_slab, (1, _H, _VPAD @ m24.blk), "gmem")
            logits_pad = tf.reshard(tf.matmul(tf.cast(normed, "f32"), tf.cast(w_head_s, "f32")), (1, 1, _VPAD), "gmem")
        logits = tf.reshape(logits_pad, (1, _VPAD))[:, :_V]
        next_token = tf.reshape(tf.argmax(logits, -1), (1,))
        return logits, next_token, k_rows, v_rows

    def init_caches(self, device=None):
        """The stacked per-layer cache buffers, zero positions long."""
        import torch  # noqa: PLC0415

        from tilefoundry.evaluator.value import to_torch_dtype  # noqa: PLC0415
        from tilefoundry.ir.types import DType  # noqa: PLC0415

        device = _generation_device(device)
        shape = (_L, 0, _KVH, _D)
        dtype = to_torch_dtype(DType.from_name(_DT))
        k_all = torch.zeros(shape, device=device, dtype=dtype)
        v_all = torch.zeros(shape, device=device, dtype=dtype)
        return _Caches({0: (k_all, v_all)}, (k_all, v_all))

    def append_cache(self, caches, fresh):
        """Grow the live prefix by writing this step's own rows in place."""
        import torch  # noqa: PLC0415

        if not isinstance(caches, _Caches):
            caches = self.init_caches()
        k_all, v_all = caches.buffers
        ctx = caches[0][0].shape[1]
        k_all[:, ctx : ctx + 1] = fresh[0]
        v_all[:, ctx : ctx + 1] = fresh[1]
        return _Caches({0: (k_all[:, : ctx + 1], v_all[:, : ctx + 1])}, (k_all, v_all))

    def prepare_inputs_for_generation(self, input_ids, step, caches, device=None):
        """The token and positional activations for one decode step."""
        import torch  # noqa: PLC0415

        device = _generation_device(device)
        token_ids = input_ids[step].reshape(1).to(device=device, dtype=torch.int64)
        cos_cache, sin_cache = _generation_rope(device)
        pos_ids = torch.tensor([step], device=device, dtype=torch.int32)
        scale = torch.full((1, 1, 1, 1), config.head_dim**-0.5, device=device, dtype=config.dtype)
        return token_ids, cos_cache, sin_cache, pos_ids, scale, caches

    def forward(self, token_ids, cos_cache, sin_cache, pos_ids, scale, caches):
        """The whole decode step: this token's logits and cache rows."""
        raise NotImplementedError("driven through the twin; see runtime_model.py")
'''

LAYER = '''        # ================= layer {i} =====================================
        with Mesh(("cta",), (24,), ("blk",)) as m24:
            # input RMSNorm (replicated: trivial work, no cross-block reduce)
            x32_{i} = tf.cast(hidden, "f32")
            var_{i} = tf.reduce(x32_{i} * x32_{i}, (-1,), True, "mean")
            hn_{i} = tf.cast(x32_{i} * tf.rsqrt(var_{i} + _EPS), _DT) * gamma_in[{i}, :]
            # fused QKV GEMV, weight split 24 ways on the (padded) output axis
            w_qkv_s_{i} = tf.reshard(w_qkv[{i} : {i} + 1, :, :], (1, _H, _QKV_PAD @ m24.blk), "gmem")
            qkv_{i} = tf.reshard(tf.matmul(tf.cast(hn_{i}, "f32"), tf.cast(w_qkv_s_{i}, "f32")), (1, 1, _QKV_PAD), "gmem")
        qkv_{i} = tf.reshard(qkv_{i}, (1, 1, _QKV_PAD), "gmem")
        q_l_{i} = tf.reshape(qkv_{i}[:, :, :_Q], (1, 1, _QH, _D))
        k_l_{i} = tf.reshape(qkv_{i}[:, :, _Q : _Q + _KV], (1, 1, _KVH, _D))
        v_l_{i} = tf.reshape(qkv_{i}[:, :, _Q + _KV : _Q + 2 * _KV], (1, 1, _KVH, _D))

        with Mesh(("cta",), (_KVH,), ("kvh",)) as km:
            # per-head q/k norm + RoPE (replicated), then the KV-head-sharded
            # online-softmax scan over the live cache prefix
            q_var_{i} = tf.reduce(q_l_{i} * q_l_{i}, (-1,), True, "mean")
            q_n_{i} = tf.cast(q_l_{i} * tf.rsqrt(q_var_{i} + _EPS), _DT)
            q_n_{i} = tf.cast(q_n_{i}, "f32") * tf.cast(gamma_q[{i}, :], "f32")
            k_var_{i} = tf.reduce(k_l_{i} * k_l_{i}, (-1,), True, "mean")
            k_n_{i} = tf.cast(k_l_{i} * tf.rsqrt(k_var_{i} + _EPS), _DT)
            k_n_{i} = tf.cast(k_n_{i}, "f32") * tf.cast(gamma_k[{i}, :], "f32")
            q_rf_{i}, _ = tf.rope(q_n_{i}, q_n_{i}, cos_cache, sin_cache, pos_ids)
            _, k_rf_{i} = tf.rope(k_n_{i}, k_n_{i}, cos_cache, sin_cache, pos_ids)
            q_rope_{i} = tf.cast(q_rf_{i}, _DT)
            k_rope_{i} = tf.cast(k_rf_{i}, _DT)
            k_row_{i} = tf.reshard(tf.reshape(k_rope_{i}, (1, 1, _KVH, _D)), (1, 1, _KVH, _D), "gmem")
            v_nb_{i} = tf.cast(v_l_{i}, _DT)
            v_row_{i} = tf.reshard(tf.reshape(v_nb_{i}, (1, 1, _KVH, _D)), (1, 1, _KVH, _D), "gmem")

            q_s_{i} = tf.cast(tf.reshard(
                tf.reshape(q_rope_{i}, (1, _KVH, _G, _D)),
                (1, _KVH @ km.kvh, _G, _D), "smem",
            ), "f32")
            scale_s_{i} = tf.reshard(tf.cast(scale, "f32"), (1, 1, 1, 1), "smem")
            # whole-context scan: the cache streams once, masked to the live
            # prefix (rows >= pos_ids are dead and contribute nothing)
            k_ctx_{i} = tf.reshard(k_caches[{i} : {i} + 1, :, :, :], (1, C, _KVH @ km.kvh, _D), "gmem")
            v_ctx_{i} = tf.reshard(v_caches[{i} : {i} + 1, :, :, :], (1, C, _KVH @ km.kvh, _D), "gmem")
            kT_{i} = tf.cast(tf.reshard(tf.transpose(k_ctx_{i}, (0, 2, 3, 1)), (1, _KVH @ km.kvh, _D, C), "smem"), "f32")
            s_ctx_{i} = tf.matmul(q_s_{i}, kT_{i}) * scale_s_{i}
            live_{i} = tf.reshard(tf.cast(tf.reshape(tf.cast(tf.arange(type=Tensor[(C,), "i32"]), "f32"), (1, 1, 1, C)) < tf.cast(tf.reshape(pos_ids, (1, 1, 1, 1)), "f32"), "f32"), (1, 1, 1, C), "smem")
            s_{i} = s_ctx_{i} - (1e30 * (1.0 - live_{i}))
            m_{i} = tf.reduce(s_{i}, (-1,), True, "max")
            p_{i} = tf.exp(s_{i} - m_{i})
            l_{i} = tf.reduce(p_{i}, (-1,), True, "sum")
            p_b_{i} = tf.cast(p_{i}, _DT)
            v_t_{i} = tf.cast(tf.reshard(tf.transpose(v_ctx_{i}, (0, 2, 1, 3)), (1, _KVH @ km.kvh, C, _D), "smem"), "f32")
            acc_{i} = tf.matmul(tf.cast(p_b_{i}, "f32"), v_t_{i})
            # the new token's own score group, merged by the same LSE
            k_nT_{i} = tf.cast(tf.transpose(
                tf.reshard(tf.reshape(k_rope_{i}, (1, _KVH, 1, _D)), (1, _KVH @ km.kvh, 1, _D), "smem"),
                (0, 1, 3, 2),
            ), "f32")
            s_n_{i} = tf.matmul(q_s_{i}, k_nT_{i}) * scale_s_{i}
            m_f_{i} = tf.max(m_{i}, s_n_{i})
            corr_{i} = tf.exp(m_{i} - m_f_{i})
            p_n_{i} = tf.exp(s_n_{i} - m_f_{i})
            v_n_{i} = tf.cast(tf.reshard(tf.reshape(v_nb_{i}, (1, _KVH, 1, _D)), (1, _KVH @ km.kvh, 1, _D), "smem"), "f32")
            l_{i} = l_{i} * corr_{i} + p_n_{i}
            acc_{i} = acc_{i} * corr_{i} + p_n_{i} * tf.cast(v_n_{i}, "f32")
            attn_{i} = tf.reshard(tf.reshape(tf.cast(acc_{i} / l_{i}, _DT), (1, 1, _Q)), (1, 1, _Q), "gmem")
        attn_{i} = tf.reshard(attn_{i}, (1, 1, _Q), "gmem")
        k_row_{i} = tf.reshard(k_row_{i}, (1, 1, _KVH, _D), "gmem")
        v_row_{i} = tf.reshard(v_row_{i}, (1, 1, _KVH, _D), "gmem")

        # this layer's cache rows: the kernel writes them in place
        k_rows = tf.insert_slice(k_rows, k_row_{i}, ({i}, 0, 0, 0))
        v_rows = tf.insert_slice(v_rows, v_row_{i}, ({i}, 0, 0, 0))

        with Mesh(("cta",), (24,), ("blk",)) as m24:
            # o_proj (24-way N split, padded), first residual, post-attention norm
            w_o_s_{i} = tf.reshard(w_o[{i} : {i} + 1, :, :], (1, _Q, _H_PAD @ m24.blk), "gmem")
            attn_out_{i} = tf.reshard(tf.matmul(tf.cast(attn_{i}, "f32"), tf.cast(w_o_s_{i}, "f32")), (1, 1, _H_PAD), "gmem")[:, :, :_H]
            h1_{i} = tf.cast(tf.cast(hidden, "f32") + attn_out_{i}, _DT)
            h32_{i} = tf.cast(h1_{i}, "f32")
            h_var_{i} = tf.reduce(h32_{i} * h32_{i}, (-1,), True, "mean")
            hn2_{i} = tf.cast(h32_{i} * tf.rsqrt(h_var_{i} + _EPS), _DT) * gamma_post[{i}, :]
        h1_{i} = tf.reshard(h1_{i}, (1, 1, _H), "gmem")
        hn2_{i} = tf.reshard(hn2_{i}, (1, 1, _H), "gmem")

        with Mesh(("cta",), (24,), ("blk",)) as m24:
            # fused gate_{i}/up_{i} GEMV (24-way), then silu(gate_{i})*up_{i}, double landing
            w_gu_s_{i} = tf.reshard(w_gu[{i} : {i} + 1, :, :], (1, _H, (2 * _I) @ m24.blk), "gmem")
            gu_{i} = tf.reshard(tf.matmul(tf.cast(hn2_{i}, "f32"), tf.cast(w_gu_s_{i}, "f32")), (1, 1, 2 * _I), "gmem")
            gate_{i} = gu_{i}[:, :, :_I]
            up_{i} = gu_{i}[:, :, _I:]
            h2_{i} = tf.cast(tf.silu(gate_{i}) * up_{i}, _DT)
        h2_{i} = tf.reshard(h2_{i}, (1, 1, _I), "gmem")

        with Mesh(("cta",), (24,), ("blk",)) as m24:
            w_d_s_{i} = tf.reshard(w_down[{i} : {i} + 1, :, :], (1, _I, _H_PAD @ m24.blk), "gmem")
            mlp_out_{i} = tf.reshard(tf.matmul(tf.cast(h2_{i}, "f32"), tf.cast(w_d_s_{i}, "f32")), (1, 1, _H_PAD), "gmem")[:, :, :_H]
            hidden = tf.cast(tf.cast(h1_{i}, "f32") + mlp_out_{i}, _DT)
        hidden = tf.reshard(hidden, (1, 1, _H), "gmem")
'''


def main() -> int:
    import os
    n = int(os.environ.get('GEN_LAYERS', L))
    layers = "\n".join(LAYER.format(i=i) for i in range(n))
    print(HEADER)
    print(FOOTER.replace("@LAYERS@", layers).lstrip("\n"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
