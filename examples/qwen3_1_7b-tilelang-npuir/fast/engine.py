"""One captured NPU-graph decode step, replayed.

The authored reference hands each step's key and value back for the caller to
``torch.cat`` on. That is the right contract for a reference -- it keeps every
shape expressed in ``ctx_len`` alone -- but it means the cache buffer moves
every step, and a graph records addresses. This engine takes the other form:
a cache of fixed capacity whose write window advances, with the position in a
one-element device tensor.

Everything a step needs then has a fixed address, so the whole step -- 229
kernel launches, embedding through the greedy pick -- is captured once into a
torch_npu graph and replayed. The chosen token is written back into the input
slot by the last kernel, and while the prompt still has a token left that
kernel feeds that one instead, so the same capture walks the prompt and
continues past it with no host round trip anywhere in the loop.

Weights are repacked once at load: ``q|k|v`` become one matrix and ``gate|up``
another, because a decode GEMV is bandwidth-bound and its cost is the block
count it can fill, not its arithmetic. Two fused reads beat five thin ones.

Every kernel is compiled with the auto-multi-buffer pass off (see
``kernels._PASS_CONFIGS``): it remaps storage slots of loop-carried buffers
and the reads after the loop go stale.
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
import torch_npu  # noqa: F401

import kernels as K

#: GEMV tiles: (BN, BK, SK, NC). SK keeps enough blocks in flight to fill the
#: 24 cube units; NC is the fixed core count the serial tile walk runs on.
TILES = {
    "qkv": (128, 256, 4, 48),
    "o": (128, 128, 2, 48),
    "gate_up": (128, 256, 1, 48),
    "down": (128, 256, 8, 48),
    "head": (128, 256, 48),
}
#: Context positions one attention block owns.
SPLIT = 256
#: Logits per argmax block.
ARG_BN = 2048


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass(frozen=True)
class Generated:
    """A continuation and the timing of the part that produced it."""

    tokens: list[int]
    seconds: float
    prefill_seconds: float
    prompt_steps: int

    @property
    def tokens_per_second(self) -> float:
        return len(self.tokens) / self.seconds


class Engine:
    """A loaded Qwen3-1.7B and one captured decode step over it."""

    def __init__(self, ckpt: str | Path, ref_dir: str | Path, *, device: str = "npu",
                 max_new: int = 2048, prompt_room: int = 512):
        ref_dir = Path(ref_dir)
        self.ref = _load_module(ref_dir / "model.py", "ref_model")
        cfg = self.ref.config
        self.cfg = cfg
        self.device = device
        self.dt = torch.bfloat16

        self.H = cfg.hidden_size
        self.HQ = cfg.num_attention_heads
        self.HKV = cfg.num_key_value_heads
        self.D = cfg.head_dim
        self.I = cfg.intermediate_size
        self.V = cfg.vocab_size
        self.L = cfg.num_hidden_layers
        self.eps = cfg.rms_norm_eps
        self.scale = self.D ** -0.5
        self.qkv_n = self.HQ * self.D + 2 * self.HKV * self.D

        self.nsteps = max_new + prompt_room
        self.cap = ((self.nsteps + SPLIT - 1) // SPLIT) * SPLIT
        self.ns = self.cap // SPLIT

        self._pack(ckpt)
        torch.npu.empty_cache()
        self._buffers()
        self._kernels()
        self._capture()

    # ---------------------------------------------------------------- weights
    def _pack(self, ckpt: str | Path):
        """Read the published checkpoint and fuse what one GEMV can serve.

        Every HF ``nn.Linear`` weight is ``(out, in)`` already -- the (N, K)
        a row-block GEMV wants -- so the projections are concatenated, not
        transposed: ``[q | k | v]`` over the output axis and ``[gate | up]``,
        because a decode GEMV's cost is the block count it can fill. Two
        fused reads beat five thin ones.
        """
        import json

        from safetensors import safe_open

        # tensors reach the device through a host staging copy: buffers the
        # device-side allocator hands out directly make the generated
        # kernels' loads read stale slots on some runs (observed on this
        # build; the host-copied path has never reproduced it)
        def to_dev(t):
            return t.to(self.device, non_blocking=False).contiguous()

        ckpt = Path(ckpt)
        index = json.loads(
            (ckpt / "model.safetensors.index.json").read_text(encoding="utf-8")
        )["weight_map"]
        shard_of = {}
        wanted = {
            "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
        }
        for i in range(self.L):
            p = f"model.layers.{i}."
            wanted |= {
                p + "input_layernorm.weight", p + "post_attention_layernorm.weight",
                p + "self_attn.q_norm.weight", p + "self_attn.k_norm.weight",
                p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
                p + "self_attn.v_proj.weight", p + "self_attn.o_proj.weight",
                p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight",
                p + "mlp.down_proj.weight",
            }
        for key in wanted:
            if key in index:
                shard_of.setdefault(index[key], []).append(key)

        tensors: dict[str, torch.Tensor] = {}
        for shard, keys in shard_of.items():
            with safe_open(ckpt / shard, framework="pt") as fh:
                for key in keys:
                    tensors[key] = to_dev(fh.get_tensor(key))

        self.w_embed = tensors["model.embed_tokens.weight"]
        # tie_word_embeddings: the head is the embedding table, in the fp16
        # the vector GEMVs want
        self.w_head = self.w_embed.to(torch.float16)
        self.gamma_final = tensors["model.norm.weight"].contiguous()
        self.layers = []
        for i in range(self.L):
            p = f"model.layers.{i}."
            self.layers.append({
                "gamma_in": tensors[p + "input_layernorm.weight"],
                "gamma_post": tensors[p + "post_attention_layernorm.weight"],
                "gamma_q": tensors[p + "self_attn.q_norm.weight"],
                "gamma_k": tensors[p + "self_attn.k_norm.weight"],
                "w_qkv": torch.cat(
                    [tensors[p + "self_attn.q_proj.weight"],
                     tensors[p + "self_attn.k_proj.weight"],
                     tensors[p + "self_attn.v_proj.weight"]], dim=0
                ).to(torch.float16),
                "w_o": tensors[p + "self_attn.o_proj.weight"].to(torch.float16),
                "w_gu": torch.cat(
                    [tensors[p + "mlp.gate_proj.weight"],
                     tensors[p + "mlp.up_proj.weight"]], dim=0
                ).to(torch.float16),
                "w_down": tensors[p + "mlp.down_proj.weight"].to(torch.float16),
            })
            for key in (
                p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
                p + "self_attn.v_proj.weight", p + "self_attn.o_proj.weight",
                p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight",
                p + "mlp.down_proj.weight",
            ):
                del tensors[key]
        cos, sin = self.ref._generation_rope(self.device)
        self.cos = cos[: self.cap].contiguous()
        self.sin = sin[: self.cap].contiguous()

    # ---------------------------------------------------------------- buffers
    def _buffers(self):
        dev, dt, f32 = self.device, self.dt, torch.float32

        def z(*shape, dtype=dt):
            # host-staged, like the weights (see _pack)
            return torch.zeros(*shape, dtype=dtype).to(dev)

        self.hid = z(self.H)
        self.xn = z(self.H, dtype=torch.float16)
        self.h1 = z(self.H)
        self.xn1 = z(self.H, dtype=torch.float16)
        self.qkv_part = z(TILES["qkv"][2], self.qkv_n, dtype=f32)
        self.o_part = z(TILES["o"][2], self.H, dtype=f32)
        self.gu_part = z(TILES["gate_up"][2], 2 * self.I, dtype=f32)
        self.d_part = z(TILES["down"][2], self.H, dtype=f32)
        self.xg_o = z(self.H, dtype=torch.float16)
        self.xg_d = z(self.I, dtype=torch.float16)
        self.op = z(self.HQ, self.ns, self.D, dtype=f32)
        self.mp = z(self.ns * self.HQ, dtype=f32)
        self.lp = z(self.ns * self.HQ, dtype=f32)
        # the per-group query buffer: rows G..15 stay zero forever, which is
        # what the score GEMM's M=16 padding needs
        self.qp = z(self.HKV, 16, self.D)
        # a zeroed cache is what makes the arithmetic gate enough: a position
        # past the end contributes a zero value with a zero weight
        self.kc = z(self.L, self.cap, self.HKV * self.D)
        self.vc = z(self.L, self.cap, self.HKV * self.D)
        # logits, padded to a whole argmax tile with a -inf tail that never
        # changes: the head writes only the first V entries
        self.nb_arg = (self.V + ARG_BN - 1) // ARG_BN
        self.padv = self.nb_arg * ARG_BN
        self.logits = torch.full((self.padv,), -1.0e30, dtype=f32).to(dev)
        self.bv = z(1, self.nb_arg, dtype=f32)
        self.bi = z(self.nb_arg, 1, dtype=torch.int32)
        # the argmax's loaded tile needs one GM write before its reduce,
        # or the reduce reads the buffer's stale slot (see kernels.py)
        self.flush_arg = z(self.nb_arg, ARG_BN, dtype=f32)
        self.pad_arg = (self.nb_arg + 7) // 8 * 8
        self.pos = z(1, dtype=torch.int32)
        self.ids = z(1, dtype=torch.int32)
        self.inp = z(self.nsteps, dtype=torch.int32)
        self.sam = z(self.nsteps, dtype=torch.int32)
        self.plen = z(1, dtype=torch.int32)

    # ---------------------------------------------------------------- kernels
    def _kernels(self):
        H, HQ, HKV, D, I, V = self.H, self.HQ, self.HKV, self.D, self.I, self.V
        self.k_embed = K.embed(V, H)
        self.k_norm = K.rms_norm(H, self.eps)
        self.k_qkv = K.gemv(H, self.qkv_n, *TILES["qkv"])
        self.k_rope = K.qk_rope_cache(
            HQ, HKV, D, self.cap, self.cap, TILES["qkv"][2], self.eps
        )
        self.k_attn = K.attn_partial(HQ, HKV, D, self.cap, SPLIT, 48, self.scale)
        self.k_o = K.gemv_attn_combine(
            HQ, D, H, *TILES["o"][:3], self.ns, TILES["o"][3]
        )
        self.k_rn_post = K.resid_rms_norm(H, TILES["o"][2], self.eps)
        self.k_gu = K.gemv(H, 2 * I, *TILES["gate_up"])
        self.k_down = K.gemv_silu(
            I, H, *TILES["down"][:3], TILES["gate_up"][2], TILES["down"][3]
        )
        self.k_rn_in = K.resid_rms_norm(H, TILES["down"][2], self.eps)
        self.k_head = K.lm_head(H, V, *TILES["head"])[0]
        self.k_arg = K.argmax_stage(V, ARG_BN, self.nb_arg)[0]
        self.k_sample = K.sample_step(self.nb_arg, self.pad_arg, self.nsteps)

    def _step(self):
        """One decode step, as the sequence of launches the graph records."""
        self.k_embed(self.w_embed, self.ids, self.hid)
        self.k_norm(self.hid, self.layers[0]["gamma_in"], self.xn)
        for i, w in enumerate(self.layers):
            kc, vc = self.kc[i], self.vc[i]
            self.k_qkv(self.xn, w["w_qkv"], self.qkv_part)
            self.k_rope(
                self.qkv_part.view(TILES["qkv"][2], self.qkv_n),
                w["gamma_q"], w["gamma_k"], self.cos, self.sin,
                self.pos, self.pos, kc, vc, self.qp,
            )
            self.k_attn(self.qp, kc, vc, self.pos, self.op, self.mp, self.lp)
            self.k_o(self.op, self.mp, self.lp, w["w_o"], self.xg_o, self.o_part)
            self.k_rn_post(
                self.hid, self.o_part.view(TILES["o"][2], self.H),
                w["gamma_post"], self.h1, self.xn1,
            )
            self.k_gu(self.xn1, w["w_gu"], self.gu_part)
            self.k_down(
                self.gu_part.view(TILES["gate_up"][2], 2 * self.I),
                w["w_down"], self.xg_d, self.d_part,
            )
            # the next layer's input norm, or the norm that closes the stack
            nxt = (self.layers[i + 1]["gamma_in"] if i + 1 < self.L
                   else self.gamma_final)
            self.k_rn_in(
                self.h1, self.d_part.view(TILES["down"][2], self.H),
                nxt, self.hid, self.xn,
            )
        self.k_head(self.xn, self.w_head, self.logits[: self.V])
        self.k_arg(self.logits.view(1, self.padv), self.bv, self.bi,
                   self.flush_arg)
        self.k_sample(
            self.bv, self.bi, self.inp, self.plen, self.ids, self.pos, self.sam
        )

    def _capture(self):
        side = torch.npu.Stream()
        side.wait_stream(torch.npu.current_stream())
        # every launcher binds its stream at construction; repoint them at
        # the capture stream so the launches land inside the graph
        for kern in (
            self.k_embed, self.k_norm, self.k_qkv, self.k_rope, self.k_attn,
            self.k_o, self.k_rn_post, self.k_gu, self.k_down, self.k_rn_in,
            self.k_head, self.k_arg, self.k_sample,
        ):
            kern.launch_stream = side.npu_stream
        with torch.npu.stream(side):
            for _ in range(3):
                self._step()
        torch.npu.current_stream().wait_stream(side)
        torch.npu.synchronize()
        self.graph = torch.npu.NPUGraph()
        # the capture stream must be the stream the launches land on
        with torch.npu.graph(self.graph, stream=side):
            self._step()
        torch.npu.synchronize()
        self._reset()

    def _reset(self):
        self.pos.zero_()
        self.kc.zero_()
        self.vc.zero_()
        self.sam.zero_()

    # ------------------------------------------------------------- generation
    def generate(self, prompt_ids: list[int], max_new: int) -> Generated:
        """Walk *prompt_ids*, then continue for *max_new* tokens.

        Timing covers exactly the steps that produce the continuation: the
        step at ``pos = len(prompt) - 1`` is the first one whose pick is
        kept, so the steps before it are prefill and are not counted.
        """
        pl = len(prompt_ids)
        if pl < 1:
            raise ValueError("decode needs a prompt of at least one token")
        if pl + max_new > self.nsteps:
            raise ValueError(
                f"prompt {pl} + {max_new} new exceeds this engine's {self.nsteps} steps"
            )
        self._reset()
        self.inp.zero_()
        self.inp[:pl] = torch.tensor(prompt_ids, device=self.device,
                                     dtype=torch.int32)
        self.plen.fill_(pl)
        self.ids.fill_(prompt_ids[0])

        torch.npu.synchronize()
        t0 = perf_counter()
        for _ in range(pl - 1):                     # prefill: picks discarded
            self.graph.replay()
        torch.npu.synchronize()
        t1 = perf_counter()
        for _ in range(max_new):                    # the continuation itself
            self.graph.replay()
        torch.npu.synchronize()
        t2 = perf_counter()

        out = self.sam[pl - 1: pl - 1 + max_new].tolist()
        return Generated(out, t2 - t1, t1 - t0, pl)

    def logits_for(self, prompt_ids: list[int]) -> torch.Tensor:
        """The logits after consuming every token of *prompt_ids* -- for checking."""
        pl = len(prompt_ids)
        self._reset()
        self.inp.zero_()
        self.inp[:pl] = torch.tensor(prompt_ids, device=self.device,
                                     dtype=torch.int32)
        self.plen.fill_(pl)
        self.ids.fill_(prompt_ids[0])
        for _ in range(pl):
            self.graph.replay()
        torch.npu.synchronize()
        return self.logits[: self.V].clone()
