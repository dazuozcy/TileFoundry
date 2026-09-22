# Qwen3-1.7B on TileFoundry — tilelang-npuir kernels (Ascend 910B)

Greedy decoding of the published Qwen3-1.7B checkpoint taken end to end on
TileFoundry: every kernel on the path written here in tilelang-npuir
(`target="npuir"`, Developer mode), and the whole decode step — 229 kernel
launches, embedding through the greedy pick — captured into **one torch_npu
graph** and replayed with no host round trip anywhere in the loop.

    ~198 tok/s    one Ascend 910B2C, batch 1, greedy, 2048 new tokens,
                  measured over the whole generation (5.0 ms/token at
                  ctx≈2000; ~228 tok/s at short context; deterministic
                  across runs)

---

## 1. Environment

Nothing here is installed by this directory; it is what the directory was
written and measured against.

| | |
|---|---|
| NPU | one Ascend 910B2C (48 AI cores / 24 cube units, dav-2201) |
| CANN | 8.5.0, `set_env.sh` sourced (`bisheng` on `PATH`) |
| Python | 3.12 |
| `tilefoundry` | the source tree at `../TileFoundry-fork` (on `PYTHONPATH` for `SafetensorsResource`; nothing else reads it) |
| kernel backend | tilelang-npuir at `/home/tilelang/zuochuanuong/tilelang-mlir-dev`, `TILELANG_ASCEND_MODE=Developer` |
| launch mode | `TILELANG_ENABLE_TASKQUEUE=false` — the task-queue path synchronizes the stream after every launch, which makes NPU-graph capture illegal; the direct `rtKernelLaunch` path does not |
| weights | the published `Qwen3-1.7B` checkpoint, 3.4 GB on disk |

## 2. How to run

With the CANN environment sourced, and the two backend variables set —
they are required in every shell, not only the first (a fresh terminal
without them fails before the first token):

    TILELANG_ASCEND_MODE=Developer TILELANG_ENABLE_TASKQUEUE=false \
        python run.py --prompt "详细解释一下昇腾 NPU 上一次矩阵乘法是如何执行的" \
            --max-new-tokens 2048

| variable | why |
|---|---|
| `TILELANG_ASCEND_MODE=Developer` | without it the JIT falls to the expert path (`--disable-hivm-tensor-compile=true`), and every dynamic-origin row copy (`embed`'s table lookup is the first) lowers to elementwise stores — the BiSheng pipeline rejects them with `'hivm.hir.store' op only support copy gm to ub or copy ub to gm or copy ub to ub` |
| `TILELANG_ENABLE_TASKQUEUE=false` | the task-queue path synchronizes the stream after every launch, which makes the NPU-graph capture illegal; the direct `rtKernelLaunch` path does not |

| flag | |
|---|---|
| `--prompt` | the text to continue; required |
| `--max-new-tokens N` | how many tokens to generate, default 2048 |
| `--ckpt` | checkpoint directory, default the published path on this machine |
| `--device` | runtime device, default `npu` |

The first run compiles the tilelang kernels (~1–5 min with a cold
`~/.tilelang` cache; seconds once cached). After changing `kernels.py` or
the environment, clear the cache first — its key normalizes buffer
placement, so a stale binary gets served as a hit. The rate covers
exactly the steps that produce the continuation; walking the prompt is
reported separately, because averaging a short prefill into a long
generation flatters the number.

    run.py                the entry point
    ref_src/              verbatim copy of the shipped `qwen3_1_7b` source —
                          the reference this implementation is measured against
    fast/kernels.py       the tilelang-npuir kernels, one decode step's worth
    fast/engine.py        weight loading, buffers, the NPU-graph capture
    fast/test_kernels.py  the torch statement of every kernel — the interface
                          they must match

### Why it is shaped this way

The authored reference hands each step's key and value back for the caller to
`torch.cat` on. That is the right contract for a reference — it keeps every
shape expressed in `ctx_len` alone — but it means the cache buffer moves every
step, and a graph records addresses. The engine takes the other form the
migrate page names: a cache of fixed capacity whose write window advances,
with the position in a one-element device tensor. Everything a step needs then
has a fixed address, so one capture serves every step of a generation. The
chosen token is written back into the input slot by the last kernel, and while
the prompt still has a token left that kernel feeds that one instead.

Weights are repacked once at load: `q|k|v` become one matrix and `gate|up`
another, converted to **fp16** (see §4), because a decode GEMV is
bandwidth-bound and its cost is the block count it can fill, not its
arithmetic. Two fused reads beat five thin ones.

## 3. Where the time goes

Marginal in-graph cost per decode step, at 2048 context (SS=256 splits,
cap 2304), measured by replaying sub-graphs:

| | per step | rate |
|---|---:|---|
| `gate_up` GEMV ×28 | 1.08 ms | ~1.3 TB/s — at the streaming roofline |
| attention ×28 | 1.01 ms | K/V scan + online softmax |
| `o` GEMV + attention merge ×28 | 0.79 ms | |
| `down` GEMV (+silu) ×28 | 0.71 ms | |
| `lm_head` GEMV | 0.49 ms | ~1.3 TB/s |
| `qkv` GEMV ×28 | 0.47 ms | |
| norms, rope, embed, argmax, sample | ~0.4 ms | |
| **total** | **4.98 ms** | **~200 tok/s** |

The whole model's weights (3.44 GB in bf16, carried as fp16) stream once per
step; 3.44 GB at the measured ~1.3 TB/s is 2.6 ms, so the GEMVs sit near the
memory floor and the remaining time is the attention scan and per-kernel ramp.

## 4. What the Ascend backend fixed about the kernels

Findings from porting the shipped CUDA twin's design onto tilelang-npuir.
Each cost real time to locate; all are cliffs, not gradients.

* **The vector units take fp16/fp32, not bf16.** The cube (`T.gemm`) takes
  bf16, but every vector intrinsic (`vexp`, `vmul`, `reduce_*`, …) rejects it.
  The projections therefore run as **fp16 vector GEMVs** — weights converted
  once at load (exact for bf16 magnitudes), activations rounded to bf16 and
  carried in fp16 — while attention stays on the cube in bf16.
* **The cube's skinny-M load path is slow.** A GEMV shaped `(BN, BK) @ (BK, 1)`
  through `T.gemm` reaches ~80 GB/s no matter the tiling. The same product as
  a *vector* chain — `vbrc` the input across a `(BN, BK)` tile, `vmul`,
  `reduce_sum`, accumulate the per-tile partial in f32 — reaches
  **~1.3–1.46 TB/s** (the d2d copy rate is 1.24 TB/s). Every projection uses
  the vector form.
* **A fixed 48-core grid beats a natural grid.** Launching `N/BN` blocks and
  letting the scheduler serialize them is slower than launching exactly the
  AI-core count and walking tiles serially inside the block. Every GEMV and
  the attention do this.
* **`npuir.enable_auto_multi_buffer=False` is load-bearing.** The pass
  remaps storage slots of loop-carried buffers, and any read after the loop
  goes stale (argreduce's CG-2026-0011, reproduced here). Every kernel
  carries state across its serial tile walk, so all are compiled with it off.
  Buffers are also allocated *inside* the tile loop for the same reason.
* **An elementwise loop with a dtype cast in it runs at scalar speed.** A
  `T.Parallel` row of `bf16(x) * gamma` costs ~165 ns/element; the same chain
  as `vcast`/`vmul` buffer intrinsics is ~100× faster. Every row-shaped
  computation (norms, rope, silu, merges) is written as buffer ops.
* **Global-memory writes go through `T.copy` onto 1D slices or matching-shape
  origin regions.** An elementwise store with a computed 2D index faults the
  AI core; reads with computed indices are fine. Partials are laid out flat
  and addressed with computed offsets on read, `T.copy` on write; 3D buffers
  support rank-reduced origin copies (`T.copy(Op[h, 0, 0], ops)`), which the
  attention-merge uses.
* **Scalar scatters of reduce results read stale.** `Bv[bid] = mx[0, 0]`
  after a `reduce_max` gives zeros on some blocks; staging through a
  one-element region copy does not.
* **Values assigned under `T.If` do not survive the branch** — including
  `vcast` results, but *not* `vmul` results or `T.copy` GM writes. The rope
  kernel picks its gamma per branch and keeps everything else shared.
* **`vrsqrt` is approximate** (0.3 % error); `vsqrt`/`vdiv` are exact. rsqrt
  is spelled `vdiv(1, vsqrt(x))`.
* **A load at a dynamic origin followed directly by a reduce can read a stale
  slot.** The argmax stage touches its loaded tile with a GM write before
  reducing — the same accident that makes the shipped argreduce's debug path
  work. Relatedly, buffers the device-side allocator hands out directly
  (`torch.zeros(device="npu")`) make these stale reads far more likely than
  host-staged ones (`.to("npu")`); the engine stages every buffer and weight
  through the host, and that has not reproduced since.
* **`T.Pipelined` + `vbrc`-of-a-literal**: `T.vbrc(1.0, buf)` with a raw
  Python float fails; `T.vbrc(T.cast(1.0, F32), buf)` does not.

Two deliberate departures from the authored reference, both toward the
published model (the same two the shipped CUDA twin makes):

* the `1/sqrt(head_dim)` factor is applied to the finished f32 score, not to
  q in bf16 first;
* attention probabilities are rounded to bf16 before the V product, which is
  what Hugging Face's own attention does.

One further departure, forced by the hardware: **the projections compute in
fp16** (bf16-rounded activations carried exactly, per-tile partial sums in
fp16, cross-tile accumulation in f32). This carries more mantissa than the
model's own bf16 per product, but less than Hugging Face's f32 accumulation,
and it is what buys the 1.3 TB/s streaming rate.

## 5. Where it stands

Measured at four levels:

1. **Every kernel against a torch statement of the same thing** at production
   dimensions (`fast/test_kernels.py`): all ten pass, exact or within the
   rounding the operation's own precision implies.
2. **Teacher-forced against Hugging Face on the real checkpoint** — 200+
   positions, mixed English/Chinese: **106/108 positions pick the same
   argmax**; the maximum logit deviation on sampled positions is ~5 % of the
   logit scale, which is the fp16-GEMV noise compounding over 28 layers.
3. **Free-running against Hugging Face**: short continuations match
   token-for-token (48/48 on an easy prompt); longer ones drift from HF's
   greedy path after tens of tokens when the fp16 noise flips a near-tie.
   The continuation stays coherent and on-topic through 2048 tokens.
4. The shipped `qwen3_1_7b` authored HIR run through TileFoundry's evaluator
   would decode in the single-digit tok/s range on this part (extrapolated
   from the H200 example's measured 14.8 tok/s, not measured here); this twin
   is on the order of 25–30× that.

### Known gaps, in the order they would pay

- **The attention merge in `o_proj`** (~0.8 ms/step): the per-head
  log-sum-exp chains still run per head; batching them over the block's heads
  with `(HH, NS)`-shaped chains and a `(HQ, NS)` Mp/Lp layout is the next
  structural change.
- **fp16 partial sums**: computing the per-tile product in f32 (a `vcast`
  before the reduce) would close most of the remaining gap to Hugging Face,
  at the cost of doubled on-chip buffers and a smaller `BN`.
- **`tilefoundry check` against the authored HIR** (the optimize page's
   per-function comparison): the reference's signatures (prior cache in,
   grown entry out) are implementable as a twin around these kernels — the
   shipped CUDA example's `twin.py` is the template — but were not wired up
   here; levels 1–2 above stand in for it.
- The prompt echo the greedy pick falls into on the Chinese benchmark prompt
  is the model's own degenerate greedy behavior on that prompt, not a
  decoding bug; an English prompt of the same shape produces a flowing
  explanation.
