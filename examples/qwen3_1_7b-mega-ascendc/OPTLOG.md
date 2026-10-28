# OPTLOG — Qwen3-1.7B single-launch AscendC mega decode step (Atlas 910B)

One `rtKernelLaunch` per decode token (the bisheng `<<<>>>` stub in
`kernel/mega.cpp` -> `run_megak`); all 28 layers + attention over the live
context + the tied LM head run inside that one launch on 24 AIC + 48 AIV
blocks.  This file records the `tilefoundry analyze` numbers for the
authored HIR (`model.py`) and reads the kernel's measured table against
them — the gate-4 record the original session left undone.

## What the analyzer says (`tilefoundry analyze`, ctx_len=1024 / 40959)

    f32 logical flops/step (ctx=1024):   5,526,761,873
    bf16 logical flops/step (ctx=1024):      1,236,992
    f32 logical flops/step (ctx=40959): 17,088,743,073
    largest values (ctx=1024):
      622,362,624   LM head            (2 * 2048 * 151944)
       50,331,648   gate/up GEMV x28   (2 * 2048 * 12288)
       ...          qkv/o/down/score GEMVs per layer

Reproduce (``QWEN3_CKPT`` points ``model.py`` at the published checkpoint):

    QWEN3_CKPT=<ckpt> tilefoundry analyze model.py:Qwen3Mega --compute-cost --dim ctx_len=1024  an-cc-1024.txt
    QWEN3_CKPT=<ckpt> tilefoundry analyze model.py:Qwen3Mega --compute-cost --dim ctx_len=40959 an-cc-40959.txt

Caveats, stated plainly:

* `--memory` / `--roofline` / `--performance` cannot run for Ascend targets
  ("no Facts projection for 'MemoryHierarchyFacts' / 'ParallelCapacityFacts'")
  — same limitation the npuir example hit.  The roofline below is anchored
  on a measured bandwidth instead.
* The HIR casts operands to f32 before every matmul so the interpreter
  reproduces the kernel's numerics; the analyzer prices those as f32 flops.
  The kernel executes the same dots as bf16 cube ops with f32 accumulators,
  so real compute work is ~half the f32 figure.  The step is nowhere near
  compute-bound either way (arithmetic intensity at ctx=40959 is
  17.09 G flop / 8.04 G B ≈ 2.1 flop/B, deep in the memory-bound regime).

## Traffic model (bytes the kernel must stream per step)

    weights (bf16, streamed once per step):
      Wqkv  28*2048*4224*2 = 483.2 MB
      Wo    28*2048*2048*2 = 234.9 MB
      Wgu   28*2048*12288*2 = 1411.1 MB
      Wd    28*2048*6144*2 = 705.5 MB
      Wlm   151936*2048*2  = 622.3 MB
      total                = 3457.0 MB
    cache (K+V, live prefix only):
      2 * 28 * 8 * C * 128 * 2 B = 114,688 * C  (112 KB per context row)
    plus the AscendC kernel's own stage round trips (AIC GEMV slabs -> GM ->
    AIV read-back): ~384 KB per layer-block ≈ 258 MB/token (~7.5%) — the
    npuir kernel fused those; this one pays them (see NOTES).

Measured HBM anchor (this card, this session): large device-to-device copy
reads+writes at 1248 GB/s; the npuir mega kernel sustains 775-797 GB/s
effective on the identical access pattern.  Both are reported below; the
roofline lower bound uses the measured copy figure.

## The table (gate 5, completed: ours / HF torch_npu greedy / roofline)

ours: `run.py --bench ctx` (the session's measurement, median of 8).
HF: cached greedy decode, same card + checkpoint, measured fresh
(`AutoModelForCausalLM`, bf16, 8 timed steps after warmup).
roofline = must-move bytes / 1248 GB/s (measured copy bandwidth).

|    ctx | ours ms/tok | ours tok/s | HF tok/s | vs HF | must-move GB | ours eff. GB/s | roofline ms | roofline tok/s |
|-------:|------------:|-----------:|---------:|-----:|-------------:|---------------:|------------:|---------------:|
|      1 |        8.65 |      115.6 |     60.1 | 1.9x |        3.457 |            400 |        2.77 |             361 |
|    128 |        8.78 |      113.9 |     61.4 | 1.9x |        3.472 |            395 |        2.78 |             360 |
|    512 |        8.73 |      114.6 |     62.0 | 1.8x |        3.516 |            403 |        2.82 |             355 |
|   2048 |        8.84 |      113.1 |     69.3 | 1.6x |        3.692 |            418 |        2.96 |             338 |
|   8192 |        9.69 |      103.2 |     66.4 | 1.6x |        4.397 |            454 |        3.52 |             284 |
|  16384 |       10.90 |       91.7 |     62.8 | 1.5x |        5.336 |            490 |        4.28 |             234 |
|  32768 |       13.94 |       71.8 |     60.2 | 1.2x |        7.215 |            518 |        5.78 |             173 |
|  40000 |       15.04 |       66.5 |     60.2 | 1.1x |        8.045 |            535 |        6.45 |             155 |

Reading it honestly: the kernel runs at 400-535 GB/s effective on must-move
bytes — 52-69% of what the same card sustains on the same pattern (the
npuir kernel's 775-797 GB/s), and 2.3-3.1x above the copy-bandwidth bound.
The gap is the ~400 plane `SyncAll`s that gate fixpipe-output visibility
(double-barrier per gate — see README "hard-won platform rules" #3) plus the
258 MB/token stage round trips; both are identified, hardware-verified
costs, not mysteries.  Removing them is the npuir kernel's answer, not this
deliverable's claim: this kernel's contract is raw AscendC, and the table
above is what it achieves under it.

## The check gate (gate 1-2, completed)

    QWEN3_CKPT=<ckpt> ASCEND_RT_VISIBLE_DEVICES=0 \
    tilefoundry check runtime_model.py:Qwen3MegaRT.mega_step \
        --inputs files:acts/token_ids.pt,acts/cos_cache.pt,acts/sin_cache.pt,acts/pos_ids.pt,acts/scale.pt,acts/k_caches.pt,acts/v_caches.pt \
        --weights ckpt:prepared --dim ctx_len=256 \
        --out output[0] --fn allclose --atol 0.15 --rtol 0.05 --fn rel_l2 --max 0.02 \
        --out output[1] --fn equal \
        --out output[2] --fn allclose --atol 0.3  --rtol 0.05 --fn rel_l2 --max 0.05 \
        --out output[3] --fn allclose --atol 1.0  --rtol 0.05 --fn rel_l2 --max 0.05
    -> PASS (logits rel_l2 0.0073, next_token equal, k_rows rel_l2 0.0048,
             v_rows rel_l2 0.0117)

`acts/` and `prepared/` are symlinks to the npuir example's teacher-forced
activations and HIR-layout checkpoint repack — the same artifacts its gate
consumes; the semantic side (`model.py`) is shared verbatim, so one check
now binds all three: HIR == npuir kernel == AscendC kernel.

---

# Optimization rounds R1..R3 (2026-10, mega_v3.cpp)

Continuation of the baseline above. Bench is `run.py --bench ctx`
(ms/tok, tok/s), gates are `test_mega.py <sl> <lstop>` (torch reference,
all intermediates + argmax). Real-weights decode produces coherent text
(~248 tok/s at ctx~20).

| round | change | ctx=1 tok/s | ctx=40000 tok/s |
|-------|--------|------------:|----------------:|
| base (mega.cpp) | poll+2xSyncAll per gate | 119.9 | 67.6 |
| R1 (mega_v2) | sentinel gates, zero SyncAll | 178.4 | 83.5 |
| R2 | MTE2 ping-pong pipeline (gemv/gemm/lm) | 197.1 | 93.5 |
| R3a | overlapped gate-poll sweeps | 200.7 | 93.5 |
| R3d | chunk-major weight repack | 230.0 | 98.9 |
| R3e | sharded silu (was 24x replicated) | 244.3 | 102.0 |
| R3f | batched V-append RMW | **251.7** | **101.8** |

ms/tok: ctx1 8.34 -> **3.97** (2.10x), ctx40k 14.78 -> **9.83** (1.50x).

Key facts discovered (see smoke3/NOTES.md for the detailed rules):

* Timeline instrumentation works: `GetSystemCycle()` gives a coherent
  cross-core counter; 128B-strided scalar+dcci writes into spare syncws
  words (compact sharing of a 128B line loses writes -- same race as the
  seq area). Driver: `tl_dump.py` on kernel/mega_tl2.cpp.
* The baseline profile was dominated by sync latency, not bandwidth
  (AIC scalar 62% spin). After R1/R2/R3 the layer is ~135us: AIC weight
  stream ~85-100us (BW floor ~52GB/s/AIC), AIV compute ~24us, gate
  handoffs ~30us (dominated by AIC land-skew + poll detect, NOT the 0.3us
  publish itself).
* `FetchEventID()` does NOT allocate -- it peeks and always returns 0;
  the whole kernel's flags run on channel 0 with deep level counters
  (32+ outstanding SetFlag per channel verified). Distinct channels need
  hardcoded ids (verified: SetFlag(a)/WaitFlag(b!=a) hangs, works for a==b
  up to id 5).
* Chunk-major weight repack removed the 17x per-chunk DataCopy issue
  overhead AND the highly-strided HBM pattern (lm read 16x 3KB blocks
  strided 4.86MB apart before).
* R3b (L1 prefetch ring, N=4) is DESIGNED but NOT landed: a hard-to-debug
  hang appeared for the second ring gemv across layers (qkv-only ring
  passes lstop<=2, hangs later; flags verified balanced). The ring code
  lives in kernel/mega_v4.cpp for a future attempt; the A-side decoupling
  (gate-dependent activations) and cross-gemv flag-state are the tricky
  parts to revisit.

Remaining known headroom (measured on the timeline): AIV compute
(~24us/layer; combine ~8us and st3 ~4us are the big items), gate
handoff/skew (~30us/layer), lm gemv ~530us/token (BW-bound), and the
gemv stream itself (~85us/layer, only hideable via prefetch).

---

# Optimization round R4 (2026-10, R3b ring LANDED -> kernel/mega.cpp)

Continuation of the baseline above.  Methodology this round: `msprof op`
hardware counters first, then a targeted fix.  The kernel symbol is
mangled, so profile with the full name, e.g.

    msprof op --aic-metrics=Roofline \
      --kernel-name=_Z5megakPhS_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_iii_mix_aic \
      --output=<dir> --launch-count=10 --warm-up=5 python <driver>.py

and read `PipeUtilization_*.csv` per block.

## What the counters said (ctx=1, R3f baseline = mega_v3.cpp)

Per-AIC (block 0, representative; all 24 identical):

    aic_time   3718us   cube 6.3%   scalar 55.0%   mte1 22%   mte2 60.8%
    aiv_time   3724us   vec  4.0%   scalar 93.5%   mte2 1.5%  mte3 5.9%

Read plainly: **the kernel is latency/serialization-bound, not bandwidth
bound.**  The AIC MTE2 pipe (weight streaming) is idle 39% of the time and
the scalar pipe (poll loops) is busy 55%; the AIVs spin 93% waiting.  MTE2
active BW is already ~1.75 TB/s aggregate (HBM read peak), so the only way
forward is to fill the 39% MTE2 idle with *weight streaming that would
otherwise happen after the gate*.  AIV vector work is a rounding error.

## R4a -- prefetch ring, the R3b idea, fixed (kernel/mega.cpp)

The static weight gemvs' B-side does not depend on the gate, so it can be
streamed into L1 during the gate wait (the AIC is otherwise idle).  Design
(see `ring_push`/`ring_fill`/`gemv_ring` in mega.cpp):

* Ring of `RN=4` x 64KB B1 buffers.  Per gemv the caller runs
  `gemv_begin()` + `ring_fill()` **before** its gate poll, so MTE2 streams
  the first RN chunks while the AIC would spin; `gemv_ring()` then consumes
  the ring and tops it up inline.
* The gate-dependent A-pack is staged in L1 in groups of 8 k-chunks (64KB);
  group 0 is loaded at gate-open, later groups (only the `down` gemv, nk=24)
  reload at their boundary.  This is the load-bearing difference vs the
  first R3b attempt: loading A per-chunk (v3-style ping-pong) *in addition*
  to the ring deadlocks the M pipe; the group-A staging (from mega_v4.cpp)
  does not.
* Ring slot flags use hardcoded distinct event channels 1..4 (both
  MTE2_MTE1 and MTE1_MTE2); the A-group barrier uses channel 5; the v3
  A/cube channels stay on 0.  `FetchEventID` peeks (returns 0) so the ids
  must be explicit.  QUE_MAX_EVENT is 8 on dav-2201.
* The ring's leftover slot-free flags are drained at gemv exit
  (`used = min(total, RN)`), and the drain is skipped when the slab is
  empty (AICs 22/23 for o/down, `nrdLive==0`) -- that unguarded drain was
  the exact R3b hang.

## R4b -- qkv gate narrowing (AIV)

The AIV's stage-3 qk-norm/rope reads only its own head's q/k/v output
columns, produced by a few AIC slabs (q cols `kv*256..+256`, k/v cols
`2048/3072 + kv*128..+128`, AIC = col/176).  The gate now polls only those
AICs instead of all 24, so the AIV no longer waits on the slowest of 24
(removes the ~12us cross-block qkv land-skew from the critical path).

## Results

Kernel-only task duration (msprof op, ctx=1): **3738us -> 3446us (-7.8%)**.
The gain comes entirely from MTE2 idle falling; AIC mte2_time is unchanged
(~2247us) so the ratio rises 60.8% -> 65.2%, and the AIV scalar spin drops
~300us.  Wall-clock, same card/checkpoint as the baseline table:

| | R3f (v3) | R4 (mega.cpp) | vs R3f |
|---|---|---|---|
| ctx=1 tok/s | 251.7 | **277.0** | 1.10x |
| ctx=40k tok/s | 101.8 | **106.1** | 1.04x |
| kernel us (ctx=1) | 3738 | **3446** | 1.08x |

Versus the very first mega.cpp the step is now ~2.4x faster at ctx=1.
Real-weight greedy decode is coherent (269 tok/s at ctx~5).

## Correctness

`test_mega.py` gates PASS at sl=0/lstop=1,28; sl=300/lstop=2;
sl=1000/lstop=1; and the real-weight `run.py` decode is coherent.  Gate
sl=37/lstop=4 reports xn1P/xn2P/xnfP large but **the values are identical to
mega_v3.cpp** (0.0625/0.0625/0.05469) -- the harness tolerance is just
tighter than the bf16 RINT noise at that depth; argmax matches.

## R4c -- tried and rejected: 6-slot ring (kernel/mega_v6.cpp)

Reusing the idle gemm1/gemm2 K/V ping-pong buffers (`b1`/`b1b`) as 2 extra
ring slots (RN=6, same 464KB L1 footprint) hangs at the first gate.  Root
cause not isolated; the aliased slots are written by gemm1/gemm2 on channel
0 and then recycled by the ring on channels 5/6.  Kept as mega_v6.cpp.

## Remaining headroom

Kernel is 3446us vs the ~2247us MTE2-work floor; the residual ~1200us is
gate-wait MTE2 idle the 4-chunk ring cannot cover (L1 limit).  Deeper
prefetch needs either the 6-slot aliasing fixed or an L2-warming pass (a
small reused L1 scratch streaming the next gemv's weights during the wait,
so the gemv reads hit L2).  Both are future work; the AIC's `aic_wait_land`
+ `aic_seq` handoff and the AIV qk-norm/rope stage are the next non-ring
targets.


---

# Optimization round R5 (2026-10, ctx=40000 latency-bound)

Continuation, same methodology (msprof counters + a per-stage timeline, then
one targeted change at a time, gates after each).  New tooling this round:
`kernel/mega_v5tl.cpp` is the current kernel with the R4 `tl_put` markers
re-added (13 AIC events, 13 AIV events + a pass-1 split at ev 16); it builds
to `libmegak_tl.so` and is read back by `/tmp/opencode/tl_dump2.py <ctx>`.
The timeline is what turned each round's next target from a guess into a
number.

## Where the time was (ctx=40000, R4 baseline = mega.cpp head)

msprof: kernel task 9195us; per AIC mte2 busy 73.9%, scalar 44.5%, fixpipe
13.2%; per AIV scalar spin 88-90%.  The timeline (mean/layer, AIC0):

    qkv 22.4 | wait-appends 5.3 | gemm1(scores) 74.0 | wait-softmax 52.6
    gemm2(PV) 61.3 | wait-combine 5.5 | o 6.0 | wait-rms2 10.0 | gu 43.9
    | wait-silu 7.7 | down 22.5      => 312us/layer

So the **softmax bubble was the single largest item** (52.6us, 17% of the
layer): the AIC is idle while the AIV runs its two-pass softmax, further
split by the pass-1/pass-2 marker as 22.2us (max) + 29.0us (exp/probs).

## R5a -- wide-chunk softmax (AIV)

The softmax ran one 256-position tile per iteration (2 GM loads + a
tree reduce), 53 tiles x 2 passes = 106 latency-serialized iterations
(DataCopy round-trip + reduction barrier each).  Rewritten to process
CW=1024 positions (4 tiles) per iteration: the two score rows are loaded
as 1024-wide copies, reduced with `tree_max`/`tree_sum` at width 1024, and
the probs are packed into a 4-tile buffer (`pw`) and written with ONE
`DataCopy` of 16384 elements.  The tail (`sblk % 1024`) falls back to the
old 256 loop.  Numerics are unchanged (same f32 exp, same bf16 RINT land).

    51.2us -> 20.5us per layer (pass1 22.2->7.1, pass2 29.0->13.3)

## R5b -- fixpipe `mSize` (the 16-row write)

Every `Fixpipe` wrote all 16 L0C rows to GM, but the consumers read only
row 0 (weight gemvs) or rows 0,1 (scores, PV) -- an 8-16x redundant GM
write (`aic_fixpipe_time` was 1120us, 12.2%).  Reduced to `mSize=2` for the
weight gemvs and `mSize=3` for scores/PV.

The landing sentinel had to move: it was armed in a *dead* row so the AIV
never read it, and `mSize<16` no longer writes rows 2..15.  Arming row 0
instead **races the AIV's read of the previous layer's region** and
produced NaN at layer 1+.  Fix: use the dead **row 1** for the weight
gemvs (only row 0 is read) and dead **row 2** for scores/PV (rows 0,1
read); both are written by the shrunk fixpipe.  Sentinel row constants
`sr=1`, `sr2=2`.

    fixpipe 1120us -> ~600us; per-layer gemvs qkv 22.4->19.8, gu 43.9->34.5,
    down 22.5->17.3, PV 61.3->56.9

## R5c -- contiguous K cache (host + append + gemm1)

gemm1 read each tile's K as 8 x 8KB chunks strided (SPADG/16)*256 apart
(1.3MB), while gemm2 reads V fully contiguous.  The K cache layout changed
from `[d-block j][s-block i][16s x 16d]` to
`[i-tile][j][i-in-tile][16s x 16d]` so a tile's 128 fractals are one
contiguous 64KB run, still in the `[j][i]` (k-major,n-minor) fractal order
the Mmad's L0B wants -- so gemm1 is now ONE `DataCopy(...,32768)` per tile.
`pack_kc` (test_mega/bench) does the new permute; `model_run` needs no
change (the kernel appends the cache itself).  gemm1 74.0 -> 70.2us.

## R5d -- tried and rejected: L0C double-buffer

Two CO1 accumulators so tile t+1's Mmad overlaps tile t's Fixpipe: **no
change** (gemm1 is MTE2-bound, not fixpipe-chain-bound).  Reverted.

## Results

Wall clock, random weights (`bench.py`-style) and real weights (`run.py`):

| | R4 (mega.cpp) | R5 | |
|---|---|---|---|
| ctx=1 tok/s (real) | 277.0 | 275.2 | = |
| ctx=40k tok/s (real) | 106.1 | **127.8** | 1.20x |
| ctx=40k tok/s (rand) | 108.5 | **133.8** | 1.23x |
| ctx=1 tok/s (rand) | 292.3 | 296.6 | = |
| kernel us ctx=40k (msprof) | 9195 | **7540** | 1.22x |
| AIC mte2 busy (ctx=40k) | 73.9% | 79% | |

The step is now ~1.35x the R4 deliverable at 40k context, still exactly
2.4x the original `mega.cpp` at ctx=1.

## Correctness

`test_mega.py` gates PASS at sl=0/lstop=1,28; sl=300/lstop=2;
sl=1000/lstop=1 (same set as R4; the sl=37/xn*P caveat is unchanged).  The
real-weight greedy token stream is **byte-identical to the R4 lib** for
`"The quick brown fox"` (both degenerate to `...lazy dog. 1000...`, the
model's own greedy behaviour, not a kernel difference), and coherent for
`"The capital of Italy is"` ("...France is Paris. Spain is Madrid...").

## Remaining headroom (ctx=40000, layer now 258us)

The attention chain is now the pole.  MTE2-work floor is ~202us/layer; the
residual ~56us is gate waits: softmax 22.6 (pass1 7 cannot overlap gemm1
because scores land asynchronously), wait-rms2 ~10, wait-silu ~8,
wait-combine ~6, wait-appends ~5.  Next levers, in order of size: (a)
publish probs per chunk and let gemm2 consume as they land (overlaps the
~13us pass2 with the 57us PV stream); (b) chase gemm1's per-chunk landing
from pass 1; (c) shrink the leader combine / rms2 / silu AIV stages.  All
need a finer-grained producer/consumer sync (a monotonic per-cid progress
word rather than the one-shot phase rounds).


---

# Optimization round R6 (2026-10, ctx=40000) -- Mirage-style grain interleave

Follow-on to R5's closing note: replace the one-shot phase barriers around
the attention chain with finer-grained producer/consumer sync so different
operators' *grains* (a 1024-position softmax chunk, a 256-position PV tile)
run concurrently instead of in sequence.

## What msprof says (ctx=40000, before R6)

Per AIC: `aic_mte2_ratio` 84%, `aic_cube_ratio` 7.2%, L1 bw usage 22%,
L2 read hit 8.4%, 368 MB read from main memory / core / launch.  The kernel
moves ~8.6 GB/token of HBM reads in 7.1 ms = **~1.21 TB/s aggregate**.

Bandwidth reference (torch_npu stream probes on the same 910B2C; HBM spec is
1.8 TB/s): a plain f32 read stream converges to ~1.53 TB/s (2-4 GB, so
HBM-resident), a bf16 read stream to ~1.34, a copy (r+w) to ~1.25.  (A 128 MB
probe reads 1.73 -- that is L2, the board's L2 is ~192 MB; bf16 < f32 because
the bf16 reduction kernel, not the memory, is the limit at that size.)

So the kernel runs at **67% of the 1.8 TB/s spec and ~79% of the best
measurable stream**.  MTE2 is by far the busiest unit and compute/L1 are idle,
so it is *memory-path-bound*, but it is **not at the memory ceiling** -- there
is ~20% on the table, and the gap is per-core MTE2 concurrency / dependency
latency / access-pattern efficiency rather than raw HBM throughput.
(The earlier claim in this file that the kernel was "essentially HBM-bound with
the gate waits as the only slack" was based on the 1.31 bf16-stream number and
overstated it; corrected above.)

## R6a -- progress words + streaming PV (the grain interleave)

The softmax was a phase barrier: the AIV computed all 54 probs tiles, wrote a
single `probs done` seq word, and only then did gemm2 start.  Replaced with a
monotonic **progress word** (`PR_PROBS`, a dedicated seq round idx; value
`tag + li*PROG_SPAN + chunk`, `PROG_SPAN=16`, consumed with `>=`): the AIV
publishes each 1024-position probs chunk (4 tiles) the moment it lands, and
gemm2 starts on chunk 0 and consumes every further chunk as it appears.  The
57us PV stream now overlaps the softmax tail instead of waiting for it.  The
AIV's seq MTE3 is ordered behind the probs MTE3, so the word is only visible
once the chunk is actually in GM; `PROG_SPAN < 512` keeps every word strictly
monotonic across launches (a stale token can never satisfy `>=`).

    AIC `wait-softmax` 22.0us -> 9.7us

Same round: gemm2's 16 x 4KB V copies were merged into ONE 32768-element copy
(a 256-position tile is 16 contiguous s-blocks in VcP).  This collapsed the
AIC's MTE2 instruction count (which those tiny copies dominated) but was
**wall-time neutral** -- confirming again that the AIC is not instruction-bound.
Kept (fewer instructions, cleaner).

## R6b -- wider pass-1 (PW=4096)

The AIV's pass-1 max scan ran 1024 positions (4 tiles) per iteration with a
`DataCopy -> WaitFlag -> Muls -> tree_max` round trip each, 14 of them.  Widened
to 4096 positions (16 tiles) reusing the idle probs-pack scratch (`pw` is only
used in pass 2): 4 round trips.

    pass1 max 7.35us -> 4.77us;  AIC wait-softmax 9.7 -> 7.1us

## R6c -- tried and rejected: L2 warmer during gate waits

Idea: during a gate wait the AIC streamed the *next* gemm's region into an L1
scratch and discarded it, so the gemm's reads would hit L2 -- converting the
otherwise-idle gate window into memory traffic.  It worked locally
(gemm2 57.3 -> 53.2us, warm running at ~75 GB/s) but the **layer total did not
change**.  Interpretation (revised after the bandwidth reference above): the
kernel is at ~79% of the achievable stream, so a single core genuinely has
throughput slack in its gate window, but the warm `DataCopy`s occupy the same
per-core MTE2 queue the gemm's copies then wait behind, and the warm's own
latency fills the window it was meant to exploit -- i.e. the limit is the
per-core MTE2 concurrency/dependency depth, not aggregate HBM.  Removed.

Two hazards found while building it (documented for future work): the AIC must
drain any outstanding warmer `DataCopy` before the kernel exits or the core
hangs; and a deferred drain (waited at the *next* warm, layer+1) hangs more
subtly -- drain in the same scope as the issue.

## Results

| | R5 | R6 | |
|---|---|---|---|
| ctx=40k tok/s (rand) | 133.8 | **140.3** | 1.05x |
| ctx=40k tok/s (real) | 127.8 | **134.9** | 1.06x |
| ctx=1 tok/s (rand) | 296.6 | 299.5 | = |
| kernel us ctx=40k (timeline) | 7599.5 | **7208.8** | 1.054x |
| AIC `wait-softmax` (us/layer) | 22.0 | **7.1** | |
| AIV pass1 max (us/layer) | 7.35 | **4.77** | |

Layer is now 240us (AIC0), of which ~34us is gate wait (wait-rms2 9.4,
wait-silu 7.5, wait-softmax 7.1, wait-combine 5.8, wait-appends 4.3) and
~206us is MTE2-issued work.  At 1.21 TB/s the kernel is at 67% of the 1.8 TB/s
spec / ~79% of the best measurable stream, so there is ~20% of real headroom
even though MTE2 is the busiest unit.  Closing it is not about adding more
prefetch windows (R6c) but about raising per-core MTE2 efficiency: fewer /
longer / better-overlapped transfers (the weight-gemv ring is only 4 deep and
the AIC's DataCopy issue is bursty around each gate), and byte reduction.  Note
also that the 24 AIVs each redundantly re-pack and re-write the same
xn1P/xn2P/attnP/xnfP broadcast packs (~6 MB/layer of duplicated HBM writes);
harmless while HBM is at 67% but worth removing if the AIC ever gets close to
the ceiling.

## Correctness

`test_mega.py` gates PASS at sl=0/lstop=1,28; sl=300/lstop=2; sl=1000/lstop=1
(same set as R4/R5; the sl=37/xn*P caveat is unchanged).  The real-weight greedy
token stream is byte-identical to `libmegak_r4.so`/`libmegak_r5.so` for
`"The quick brown fox"`, and coherent for `"The capital of Italy is"`.

--------------------------------------------------------------------------------
## R7 — AIV-plane gu-weight L2 warmer (v8)

The AIV plane is ~97% MTE2-idle (`aiv_mte2_ratio` 3.1-3.5%), while the AIC plane
stalls ~34us/layer on AIV-produced stages.  During the *all-AIV wait for the 22
AIC "o" words* (the stage-6 gate) the AIVs have an independent MTE2 and nothing
to do, so they stream the gu gemv's weight slab -- the very next bulk read --
into L2 and discard it (16KB UB scratch slots, MTE2_V ids 6/7).  Unlike the
rejected R6c (same trick issued from the AIC, which self-blocked on its own MTE2
queue), this uses a different core's MTE2.  The gu slab is 2.1 MB/AIC
(32 x 64KB chunks); chunks 0..3 are the AIC ring's own prefetch, so the AIVs warm
chunks 4..31, split across the two subs (`aiv_warm_gate`).  The scratch is `tPw`
(idle in this window) and the outstanding MTE2_V flags are drained right after
the R+6 publish (`aiv_warm_drain`), off the critical path.

Implementation: `WARM_SLOT = 8192` bf16; helpers `aiv_warm_gate` (poll a range of
AIC words, warm meanwhile, 2 UB slots, ids 6/7, wait-before-reuse at issued>=2)
and `aiv_warm_drain`.  `tWguA` is set from `p_WguP`.

Results:

| | R6 | R7 | |
|---|---|---|---|
| ctx=1 tok/s (rand) | 299.5 | **305.0** | +1.9% |
| ctx=40k tok/s (rand) | 140.3 | 139.8 | = |
| AIC gu gemv (us/layer, 40k) | -- | 25.0 | |

The ctx=1 gain is real (short layers, the AIV is the laggard); at ctx=40k it is
neutral because the window (13us) is shorter than the gu warm (14.5us) and the
warm steals HBM from the concurrent o gemv, which is also on the critical path.

### Rejected: stage-7 down-weight warm (v8b)

Warming the *down* weight (1.18 MB/AIC) in the AIV's silu-wait window was tried
and rejected: sl=0 305 -> 286 tok/s, sl=40k 139.8 -> 136.5.  The silu window is
directly on the critical path (the AIC is waiting for h), so any HBM/AIV-MTE2
contention there delays the h publish one-for-one.  Also note the guard bug that
first caused a deadlock: for AIV cid 22/23 (which skip the warm) `wpend` retained
its stage-6 value and the post-stage-7 `aiv_warm_drain` re-consumed already-
drained flags.  Any flag left outstanding at kernel exit hangs the core -- always
drain before exit and before scratch/event reuse.

## R9 — attempted AIC K/V-path fixes (all rejected, zero effect)

Two well-motivated experiments on the scores gemm (`gemm1`) and PV gemm
(`gemm2`), the largest measured intervals at ctx=40k (72.0 and 66.6 us/layer):

1. **4-deep K prefetch ring** (4 L1 buffers reusing the idle ring slots rb0/rb1,
   prefetch distance 3, A-pack loaded once).  Previously a 2-deep ping-pong.
   Result: `apprel->scoland` 72.00 -> 71.84 us, end-to-end unchanged.
2. **CO1 double-buffer** so `Fixpipe(t)` overlaps `LoadData/Mmad(t+1)`
   (single-CO1 serial chain).  Result: 72.0 us, end-to-end unchanged.

Conclusion: gemm1/gemm2 are **not** prefetch-depth- nor Fixpipe-serialization-
bound.  Note the aggregate check: the final lm gemv reads 622 MB/token at
419.8 us = 1.48 TB/s (**~62 GB/s/core**, near the 1.53 TB/s measurable peak),
whereas gemm1/gemm2 run at ~48-53 GB/s/core.  The 20-25% gap between the ring
gemvs and the K/V gemvs is *not* explained by depth or Fixpipe and remains open;
candidate causes are the K/V read address pattern vs. the pure-streaming weight
reads, and cross-plane interference during the attention window.  (A poll-dcci
change -- dcci only on miss -- was also tried and gave zero.)

## R7 status

`libmegak_r7.so` saved (== `libmegak.so`).  `libmegak_tl.so` re-ported to R7.
`test_mega.py` gates PASS at sl=0/lstop=1,28; sl=300/lstop=2; sl=1000/lstop=1.
--------------------------------------------------------------------------------
## R10 -- prefetch ring deepened 4 -> 6 chunks (v10)

msprof baseline (R7, ctx=40k): AIC mte2 81.7% / mte1 24.4% / scalar 39.3% /
cube 7.1% / fixp 8.2%; main-mem read 48.5 GB/s/core; L2 read hit 13.5% (weights
and KV are cold); AIV scalar 91.6%.  So the kernel is HBM/MTE2-bound and the only
slack is the ~18% MTE2 idle, all of it at gate waits.  The weight-prefetch ring
was only 4 x 64KB = 256KB, so during a gate wait MTE2 stalled once 4 chunks were
in flight.

Fix: RN 4 -> 6 (384KB), at zero L1 cost -- ring slots 4/5 alias the gemm1/gemm2
K/V buffers `b1`/`b1b`, which are idle whenever a weight gemv runs (weight gemvs
and gemm1/gemm2 never overlap).  New MTE2_MTE1/MTE1_MTE2 ids 6/7 (verified free;
only 1..4 were previously used).  `& (RN-1)` -> `% RN` and the drain already uses
`min(total,RN)`, so 6 slots (not a power of two) work.  RN=6 is the ceiling: all
8 MTE2_MTE1 ids are now spoken for (0 gemm1/2, 1-4 & 6-7 ring, 5 evA).

| | R7 | R10 | |
|---|---|---|---|
| ctx=1 tok/s (rand) | 305.0 | **313.7** | +2.9% |
| ctx=40k tok/s (rand) | 139.8 | **142.1** | +1.6% |
| AIC time ctx=40k (msprof) | 7091 | **6973** | -1.7% |
| AIC mte2 ratio | 81.7% | **82.5%** | +0.8pt |
| AIC main read bw (GB/s/core) | 48.5 | **49.4** | |

Same `aic_mte2_instructions` (8711) and `GM_to_L1_datas` (358816 KB): no extra
bytes or ops, purely better overlap.  TL confirms the whole gain is at the
ring-prefetch waits: `rms1rel->qkvland` -1.19, `battrel->oland` -1.98,
`xn2rel->guland` -1.46 us/layer.

### Open lead (not solved this round)

`gemm1` (scores, 71.9us/layer) and `gemm2` (PV, 67.8us/layer) run at ~49-50
GB/s/core while the ring-based gemvs reach ~62 (the final lm gemv: 622MB/token in
419.8us = 61.8 GB/s/core, no gates, no warm).  They are NOT prefetch-depth bound
(R9 4-deep: zero) nor Fixpipe-serialization bound (R9 CO1 double-buffer: zero),
and MTE1 is only 24.7% busy at ~205 GB/s when active, so the chain
LoadData+Mmad should be <0.5us/tile vs the 1.33us observed.  Something specific
to the K/V read stream or its interference caps it at ~49; the obvious next step
is to run gemm1's K side through the ring machinery (as the lm gemv does) rather
than the bespoke 2-deep ping-pong.

--------------------------------------------------------------------------------
## R11 attempt -- BLOCKED by a hardware fault (inconclusive)

Attempted: run gemm1's K-stream through the same 6-deep structure the ring gemvs
use (6 L1 buffers reusing the idle rb slots + CO1 double-buffer), expecting the
~49 -> ~62 GB/s/core lift.  It failed the gates, but on reverting, the
bit-identical, previously-passing `libmegak_r10.so` ALSO failed -- and so did the
saved `libmegak_r6.so`/`libmegak_r7.so` baselines -- with non-deterministic
maxdiffs, while a bare device `torch.matmul` (256x256) now returns wrong,
run-to-run-varying results (maxdiff ~55-70) although `torch.sum` is exact and
npu-smi reports Health OK.

Conclusion: chip 0's **cube pipeline** faulted mid-session (analogous to the
known chip-3 MTE ROB ECC fault; the harness forces chip 0, so there is no
fallback).  The R11 gate failure is therefore INCONCLUSIVE -- it may be the
device, not the code.  R11 is reverted; the deliverable is the verified R10.
RETRY R11 (6-deep K ring for gemm1) once the device is reset/replaced.

--------------------------------------------------------------------------------
## R11 -- WARNING: the 6-deep K-ring gemm1 variant HARMS THE DEVICE

Follow-up after the fault report: a bare device `bf16 matmul` probe
(256x256, vs a CPU fp32 reference; healthy maxdiff ~4.7e-4) was used to audit
every chip.

  chip 0: maxdiff NaN          -> cube FAULTED
  chip 1: maxdiff 2.5e-1       -> cube FAULTED
  chip 2: healthy -> then R11 runs -> maxdiff 5.5e-1 FAULTED
  chip 3: MTE ROB ECC (pre-existing)
  chips 4,5,6,7: maxdiff 4.7e-4  -> HEALTHY

So the earlier "device fault" was NOT pre-existing: running the R11 gemm1
variant (6-deep K ring reusing the rb slots, with or without the CO1
double-buffer) FAULTED chips 0 and 1, and then chip 2 as well, after which every
previously-good library (r6/r7/r10) and even bare torch matmul returned wrong,
non-deterministic results.  R11 deterministically corrupts gemm2's output
(scores PASS, partAcc FAIL) -- almost certainly an out-of-bounds / event-desync
in the gemm1<->gemm2 buffer sharing that trips the MTE ROB (the same fault class
as chip 3).

DO NOT RUN the R11 variant again without first auditing the gemm1 L1/event usage
for OOB; it costs a chip each time.

State: R10 reverted and re-verified on a healthy chip (4): all gates PASS,
314.2 tok/s (ctx=1) / 142.1 tok/s (ctx=40k).  `test_mega.py`'s chip guard was
relaxed to reject only chip 3 (chips 0/1/2 now faulty) -- use
ASCEND_RT_VISIBLE_DEVICES=4 (or 5/6/7) from now on.

--------------------------------------------------------------------------------
## R12 -- profiling audit + wide `pack_out` (landed); most ideas rejected

Continuation with a per-tile cycle probe (a `mega_probe.cpp` build that wraps
each gemm/gemv sub-phase in `GetSystemCycle()` and dumps sums for AIC0 into
`syncws[TL_K+...]`) plus the existing TL build and msprof on chip 4.

### Baseline, ctx=40000 (R10 = mega.cpp before this round), msprof Roofline

    task 6986.8us | AIC per core: mte2 82.5%  scalar 39.5%  mte1 24.7%
                   cube 7.2%  fixpipe 8.3% | main read 49.3 GB/s/core (1183 agg)
                   read 361.5 MB/core, GM->L1 358.8 MB/core
    AIV per core: vec 4.8%  scalar 91.5%  mte2 12.1%  mte3 2.7%

ctx=1: task 3134us, AIC mte2 64.9%, scalar 62.6%, main read 47.8 GB/s/core.

### What the per-tile probe says

* Weight gemvs (`qkv/o/gu/down`) are MTE2-bound at ~60-62 GB/s/core — i.e. at
  the per-core/HBM ceiling; their cube issue (18.6us/layer) and MTE1 (12.9us)
  hide under it.  No headroom there.
* Attention `gemm1`: per 256-pos tile, K copy waits ~0.83us (K=64KB), LoadData
  ~0.28us, and with the per-tile scores Fixpipe the `Mmad` issue stall is
  ~0.87us/tile; removing the Fixpipe drops `Mmad` to ~0.02us and makes the K
  copy the limiter (~0.83us/tile).  So MTE2 and MTE1 already overlap; the
  Fixpipe only costs ~0.07us/tile net (~11us/layer).  `gemm2` (one Fixpipe per
  layer) is the same MTE2 rate.  Both run at ~58 GB/s/core vs 62 for the lm
  gemv.  Attention is *not* prefetch-depth bound (re-confirms R9).
* `aic_wait_land` (fixpipe->L2 visibility self-poll) costs 3-5us per gemv =
  **19us/layer (ctx=1) / 31us/layer (ctx=40k)**.  `SetFlag/WaitFlag<FIX_MTE3>`
  returns in 0.1us, so the latency is the GM write becoming visible, not the
  fixpipe engine.
* AIV `pack_out` emitted the 64KB broadcast pack as 8 x 8KB DataCopies, each
  behind its own V_MTE3 and MTE3_V round trip (6 of the 8 waits are redundant).

### R12 (landed): wide `pack_out`

Build all 4 chunks (32KB) in one pass in the otherwise-idle 32KB argmax scratch
(`tAm`) and emit them with ONE DataCopy; two copies per 2048-wide pack instead
of eight.  Byte-identical, layout-identical (gates PASS).

| | R10 | R12 | |
|---|---|---|---|
| ctx=1 tok/s (rand) | 314.3 | **318.3** | +1.3% |
| ctx=40k tok/s (rand) | 141.8 | **143.1** | +0.9% |
| real greedy decode ctx~5 | 292.5 | **295.8** | +1.1% |
| msprof task ctx=40k | 6986.8 | 6959.5 | -0.4% |
| AIC mte2 / scalar | 82.5 / 39.5 | 83.0 / 39.0 | = |

`test_mega.py` gates PASS at sl=0/lstop=1,28; sl=300/lstop=2; sl=1000/lstop=1
(the known sl=37/lstop=4 caveat is unchanged).  Real-weight greedy text is
byte-identical to R10 for the same prompts.

### Tried and rejected this round (numbers are end-to-end)

1. **Prefetch the next gemv before the current gemv's landing poll** (fill the
   sentinel window with weight streaming): ctx=1 314.7 -> 306.4, 40k 142.1 ->
   141.5.  Total MTE2 work is fixed; moving it into the sentinel window just
   leaves the later gate window idle, and the extra `gemv_begin`/prefill costs
   scalar time.  Reverted.
2. **Per-tile scores Fixpipe batching (4 tiles/one Fixpipe) + double-buffered
   L0C (8 slots, 128KB)**: 314.4 / 142.2 — exactly neutral.  The Fixpipe's
   dependency stall is hidden by the K copy in steady state.
3. **Batched combine partials** (one 24KB partAcc DataCopy + one part_m/l pair
   instead of 6 serialized 512B copies): neutral (314.3 / 142.0).  The AIC's
   combine wait is land skew, not the leader's copy serialization.
4. **mSize 3->2 (scores) / 2->1 (weights), sentinel moved into a live row**:
   NaN at sl=0/lstop=28 — the arm races an AIV read (same failure R5b saw).
   Reverted.
5. **Dynamic 8-slot ring (8 x 48KB over the same 384KB) with one event channel
   per direction**: HANGS (507014).  Reverted; device verified healthy.
6. **`aic_land_fast`**: `FIX_MTE3` wait + `dcci(ENTIRE_DATA_CACHE)` then verify
   the sentinel.  Correct (gates PASS) but exactly neutral — the full-cache
   flush costs as much as the poll.  Reverted.
7. **AIV warmer extensions**: warming next-layer `qkv` during the down wait
   regressed ctx=1 to 308.2; removing the existing gu warmer regressed ctx=1 to
   309.0 (+0.5% at 40k).  The gu warm in the o-wait is the net-best window.
8. **Deferred `MTE3_V` UB-reuse wait in `seq_write`** (dedicated channel, wait
   for the same `sq` slot's copy two calls later, drain at exit): the kernel
   HANGS (507014) at sl=0/lstop=1.  Do NOT defer this flag on a second channel
   without a working probe; reverted, device re-verified healthy.  (The 8-way
   seq-write round trips remain a ~1-2% target for someone with a safe way to
   prove the UB-reuse ordering.)

### Remaining headroom (honest)

At 40k the step moves 8.68 GB of HBM reads in 6.96 ms = 1.24 TB/s aggregate,
84% of the 1.48 TB/s this card delivers on the pure lm-gemv stream.  MTE2 is
the busiest pipe at 83%; the 17% idle is concentrated in the ~34us/layer of
gate + ~31us/layer of landing waits, which the 384KB L1 ring (6 chunks, ~6us)
cannot cover.  Filling them needs *L2-resident* staging (AIV warming or a
paced software prefetch) or a structural change (online/flash softmax to fuse
away the scores round trip, or vector-unit attention) — all beyond a safe
incremental edit; the 8.68 GB itself (3.46 GB weights + 4.59 GB KV + 0.26 GB
stage round trips) is the floor.

## R13 -- attempt 2a: streamed (online) softmax max pass

Goal: attack the one serial stage inside attention.  Per layer the chain was
`gemm1 (all scores) -> R+2 barrier -> pass 1 (whole-slice max) -> pass 2 (exp,
streamed into gemm2)`.  Pass 1 re-reads the whole slice from L2 and can only
start once the *last* score tile has landed.

### Step 0: measured the ceiling before writing anything

Timing-only probe (pass 1 restricted to its last 4096-group, so every accuracy
gate stays untouched because all gates have sblk <= 512): 7.001 -> 6.908 ms at
40k.  So 3 of 4 pass-1 groups cost 93us and the *whole* pass 1 costs ~124us =
**1.8% of the token** -- the hard ceiling for 2a, and an honest bound on 2b too.

### What landed (R13, shipped as libmegak_r13.so)

Keep the two-pass math bit-identical, remove the barrier:
* AIC arms **one dead-row sentinel per 4096-position score group** (g ends at
  tile `min(16g+15, nloc-1)`; the last one IS the old single sentinel), and
  publishes a tagged progress word `SEQ_WANT(24+cid, PR_SCORES=233) =
  tag + li*16 + g+1` 4 tiles (~5us) after that group's last fixpipe, so the
  land poll is off the critical path (`gd == pubG` cursor; the drain publishes
  the tail and its last poll replaces the old all-scores `aic_wait_land`).
* AIV drops the `R+2` gate and waits per group with a new `aiv_wait_prog`
  (`>=` test on the same monotonic count scheme, PROG_SPAN=16 keeps
  `li*16+count < 512`), folding each group's max into the slice max as it lands.
* Pass 2, `part_m`, `part_l`, probs and the combine are untouched.

Why this and not M_guess single-pass: eliminating pass 1 entirely also works,
but at the four gate sizes `sblk <= 512`, so the gates cannot validate the
`M_guess != M_true` case, and `part_m`/`part_l` would stop meaning "slice max"
for sl > 1024.  The exact streamed form keeps every gate and every large-sl
diagnostic meaningful.

### Results (interleaved A/B, 3 trials, best-of-3x30 launches)

| | R12 | R13 | |
|---|---|---|---|
| ctx=1 (sl=0) | 317.5-318.0 | **319.2** | +0.4% |
| ctx=40k | 142.6-142.7 (7.010ms) | **144.0** (6.945ms) | **+0.94%** |
| real-weight decode | 295.8 | 295.8 | text identical |

Captured 65us of the 124us ceiling.  The residual ~60us is irreducible here:
the last group's scores only exist after gemm1's last tile, so its pass-1 work
plus the ~1.5us cross-core wake-up is paid serially no matter what.  Finer
tail groups would recover <0.3us/layer.

### Verification

Gates 0/1, 0/28, 300/2, 1000/1 all PASS; extra **sl=16000 lstop=1 PASS** which
does exercise the in-loop publish (sblk=5376 -> nloc=21, publish at t=19) with
`part_m maxdiff 1.4e-5` / `part_l 0.067` -- i.e. the streamed max is the same
number, no race.

### Judgement on 2b (vector-unit attention)

2a holds, but it captured the whole realizable prize: the remaining attention
side headroom is a ~2.6us/layer tail, and 2b only trades work between pipes
(MTE1 24.7% / fixpipe 8.3% -> AIV vec) while the bound resource is still AIC
MTE2 at 82.5% and the 8.68 GB/token floor is unchanged.  Expected 2b value is
now <1% for a full attention rewrite: NOT recommended.  The 17% MTE2 idle is
the per-layer serial chain (6 handshakes + AIV elementwise stages), which only
M>1 amortizes.

### R13 correction: 2b ceiling was under-estimated (memory counters)

I first argued 2b (vector PV) was worth <1% because the probs round trip
"stays in L2".  The R12 40k `Memory_*.csv` says otherwise:

* AIC per core: main-mem read 361527 KB, L0C->GM 6389 KB.
* AIV per core: main-mem read 21795 KB, **UB->GM 7955 KB** (x48 = 373 MB/token).
* The AIC re-reads ~285 MB/token of that as gemm2's A side.

Logical probs are only 36 MB/token, but the A-side fractal needs 16 rows and
only 2 are live (heads 2kv, 2kv+1), so probs are written and read **7.8x
inflated: ~568 MB/token, ~5% of the step**.  So 2b's real ceiling is the
probs+scores round trip (~7%), not <1%.

Cheaper first step than 2b: have the AIV write probs **compact** (2x256 bf16 =
1 KB/tile) and let the AIC's `LoadData` expand them into the 16-row L0A fractal
with `dstStride` (rows 2..15 may stay stale garbage -- the fixpipe only keeps
rows 0/1 and dead row 2).  Same 7.8x cut on both write and read -> ~568 MB
becomes ~72 MB, ~+5%, with no cube/numerics change.  Not attempted here; it is
the best incremental target left on the attention side.

## R14/R15 -- attempt #2 (AIV polls the fixpipe sentinel directly): REFUTED

Idea: every AIC->AIV gate was two hops (`fixpipe -> L2 -> AIC aic_wait_land ->
AIC aic_seq -> L2 -> AIV poll`).  Hypothesis: let the AIV poll the dead-row
sentinel word itself, deleting the AIC from the loop (6 gates/layer, ~18 hops
total).  Expected 2.5-3.5% @40k, 5-8% @ctx=1.

### Implemented (R14)

* AIC: all 6 `aic_wait_land` + `aic_seq` pairs removed.  The 5 reused sentinels
  (qkv, score groups, partAcc, o, gu) are now armed at the *end of the previous
  layer* and the existing `aic_seq(R+7)` publish is the "sentinels for layer
  li+1 are armed" barrier; `d` is armed after `poll_seq(R+0)` (i.e. after the
  AIVs published rms1, which is after their resid2 read of `d`).
* AIV: new `aiv_land1` / `aiv_land_range` poll the data sentinel directly; the
  L2 warmer polls the `o` sentinels instead of seq words.
* Bring-up bug (worth remembering): the poll must `dcci` *before* reading.  The
  tagged seq words need no such thing (a stale value can never equal `want`),
  but "anything but SENT" is satisfied by the AIV's own cached copy of last
  layer's real data -> silent stale reads at layer 1+.  Fixed by an
  invalidate-then-read entry pass.

All gates PASS with the fix (0/1, 0/28, 300/2 twice, 1000/1, 16000/1).

### Measured (interleaved A/B, best-of-3x30)

| | R13 | R14 (all 6 direct) | R15 (hybrid: o/d kept 2-hop) |
|---|---|---|---|
| ctx=1 | 318.5 | **309.5 (-2.8%)** | 310.1 (-2.6%) |
| ctx=40k | 144.3 | **143.2 (-0.8%)** | 142.9 (-0.9%) |

### Why it loses -- and what it says about the sync graph

The second hop was **never on the critical path**.  `aic_wait_land` runs on the
cube while the AIV is already busy with its own stage, so the two hops overlap:
the AIC detects the landing ~3-5us after the fixpipe, and the AIV's detect adds
~1.5us that is hidden behind the AIV's own work.  What the direct poll changes
is *who pays* for the visibility check: it moves a `dcci` spin loop from the
idle cube's scalar pipe onto the busy AIV's critical path, and adds a 22-word
arm-barrier poll per layer (R15 kept the two 22-word gates 2-hop and still lost
2.6%, so the cost is spread over the ~10 dcci/layer of the cheap gates + the
barrier).

Corollary: the earlier "18 hops x 1.5-2us = the 28-41us/layer bubble" model is
wrong -- the hops overlap.  The bubble is a **dependency stall**: the AIC waits
for the AIV to finish an elementwise stage that produces the next gemv's whole
input vector, so no sync restructuring can remove it.  Only M>1 amortizes it.

### Status

Reverted; `kernel/libmegak.so` == `libmegak_r13.so` (md5 9dd115ae...), R13 gates
re-verified PASS, 144.0 / 318.6 tok/s.  The hybrid source is kept at
`kernel/mega_r15src.cpp` for reference.  Do NOT retry this without a way to make
the AIV-side visibility check free (e.g. a fence the fixpipe can raise).

## R16 -- probs round trip made compact (Nd2Nz expand in the cube)  **+6.0% @40k**

Item #1 of the sync/pipeline list.  The `Memory_*.csv` counters (see the R13
correction above) showed the probs staging was the largest *removable* chunk of
traffic: the A side of gemm2 is a 16-row fractal but only rows 0,1 are live (the
two Q heads of that kv head), so probs were written and read **8x inflated** --
~568 MB of the ~9.1 GB a 40k token moves.

### Why the obvious fix does not work

Storing probs compact and letting MTE1 expand them is impossible: `LoadData2D`
moves whole 512B fractals (`kStep`/`srcStride` are in fractal units, see
`asc/.../load_to_l0a/load_to_l0a_gemv.h`), so the A-side read is irreducibly
8KB/tile.  The expansion has to happen on the **GM->L1** path instead:
`DataCopy(dstL1, srcGM, Nd2NzParams)` is an MTE2 ND->NZ transform that reads
`nValue x dValue` elements and writes them into the NZ layout *without touching
the rows it does not fill*.

### Change

* AIV: `hbw` is now filled in the compact interleaved layout
  `[tile][head0 256][head1 256] bf16` (8 `Cast` of 256 instead of 2 of 1024; the
  old 8 `strided_rows` disappear), and probs are written as **4KB per 1024-chunk
  instead of 32KB** (tail path: 1KB/tile).  ~297 MB -> ~37 MB per token.
* AIC: `nd2nz_probs()` -- `Nd2NzParams(ndNum=1, nValue=2, dValue=256,
  srcDValue=256, dstNzC0Stride=16, dstNzNStride=1)`.  `dstNzC0Stride` is in
  C0_size (16 bf16) units, so 16 -> 256 elements between consecutive k-blocks,
  exactly the A-fractal stride; rows 2..15 of the fractal are left stale (finite
  garbage from the previous tile/K data, and row 2 only ever feeds the
  sentinel, which just needs `!= SENT`).  Read ~285 MB -> ~36 MB.
* No host/pack change: `probs2P` is simply addressed as `(cid*NTM + t) * 512`.

### Results (interleaved A/B, best-of-3x30)

| | R13 | R16 | |
|---|---|---|---|
| ctx=1 | 317.8 | **319.9** | +0.7% |
| ctx=40k | 144.3 (6.930ms) | **153.0** (6.535ms) | **+6.0%** |
| real-weight decode | 295.8 | 297.9 | text identical |

395us saved at 40k vs the 496 MB predicted / measured -> 397us at the old
1.25 TB/s: the byte accounting was exact.  Effective aggregate is now
8.6 GB / 6.535 ms = **1.32 TB/s (88% of the 1.5 TB/s achievable)**, up from 1.24.

### Verification

Gates 0/1, 0/28, 300/2, 1000/1 and **16000/1** all PASS with maxdiffs *identical*
to R13 (e.g. partAcc r0 0.005681 at 16k, attnP 0.01562 at 0/1) -- the layout
change is numerically exact, as expected (rows 0,1 are written identically; the
padding rows are discarded).  `libmegak_r16.so` is the deployed binary.

### Note for whoever continues

The same 16-row-broadcast waste exists for `q2P` (4KB/layer/cid) but it is
L2-resident and negligible.  The probs were the only one on the HBM path.

## R17 -- attempt #2 (cheaper PR_PROBS hand-off): NEUTRAL / REFUTED

The probs streaming hand-off publishes `tag + li*PROG_SPAN + chunk` from the AIV
once per 1024 positions (14x/layer at 40k); gemm2 consumes it at grading
granularity.  Two forms were tried, both on top of R16.

### (a) Precomputed word table, publish without the flag pair -- BROKEN

All progress words are loop-invariant per layer, so they can be built once
(`Duplicate` x nchp into a 16x16 int32 UB buffer) and each publish becomes a
bare `DataCopy(gseq[slot], sqp[i*16], 16)` with no `Duplicate`/`V_MTE3`/
`MTE3_V` around it.  All gates: PASS with the precompute in place but the old
`seq_write` publish, **FAIL** (partAcc/attnP = 1e31 garbage, i.e. gemm2 gated
open early) as soon as the publish itself is the bare copy -- with or without a
trailing `MTE3_V` wait.

So the flags around each publish are not overhead, they are the guard band that
makes the progress word *reach L2 after* the chunk's probs burst does:
`MTE3_V` = "copy retired", and retiring is not visibility.  This is the same
lesson as R14: any attempt to shave the guards on a GM-visibility hand-off buys
latency back by trading correctness.  Do not retry without a real fence.

### (b) Keep seq_write, publish every 2048 instead of 1024 -- NEUTRAL

Only change: the publish condition/value and the AIC's `(t + 1) / 8 + 1`.
Gates PASS (0/1, 0/28, 300/2, 1000/1).  A/B at 40k, 3 interleaved trials:

| | R16 | R17 (2048-grain) |
|---|---|---|
| ctx=40k | 152.8 | 152.8 |

Exactly neutral.  Halving the publish count gives nothing back, which means the
14 `seq_write`s were **already fully hidden**: the AIV produces a 1KB probs
chunk in ~0.6us while gemm2 consumes the matching 16 V-tiles in ~20us, so the
producer is ~8x ahead of the consumer and its flag stalls have slack.  Note this
does NOT contradict R14 (which lost 2.8%): there the `dcci` was added to the
*poll* path, which is on the critical path; the *publish* path is not.

### Status

Reverted to R16 (`libmegak.so` md5 c237d091..., gates re-verified PASS,
153.0 / 319.9 tok/s).  Source snapshots: `mega_r16src.cpp` (current),
`mega_r13src.cpp`, `mega_r15src.cpp`.  The precomputed-table variant is kept at
/tmp/opencode/r17_try.cpp for reference only -- it is incorrect.

## R18/R19 -- de-duplicating the AIV plane (and the L2 question behind it)

### The question that started it

"weights are read once per token, so their L2 occupancy is useless and only
evicts useful data -- can we bypass L2 for them?"  Answer: **not on this chip**.
`GlobalTensor::SetL2CacheHint` / `CacheMode::CACHE_MODE_DISABLE` exists, but the
address-encoded implementation (`L2CacheAlter`, addr bit>=60) is compiled only
for `__NPU_ARCH__ == 3101 || 5102`; dav-c220 (2201) never defines it.

And the premise targets the wrong data: the weight stream's L2 *hit rate* is
already 0, so it costs no hits, and all HBM traffic crosses L2 anyway (a bypass
adds no bandwidth).  The read the L2 does hurt is the **score matrix** (2.6 MB
per layer, evicted by 168 MB/layer of KV) which is re-read **4x** per token --
pass1 and pass2, each executed by *both* AIV subs of the group.

### R18: sub1 skips stage 4

Every output of the softmax stage (probs2P, part_m/l, the R+3 word, the combine)
is `sub == 0`-only, so sub1's pass1+pass2 was pure dead work: ~146 MB/token of
HBM reads.  One-line guard.  Gates PASS.  Result: ctx=1 +0.3%, **40k -2.2%**.

The loss is not the de-dup: with sub1 no longer spending time on the softmax it
*arrives at the o-gate earlier*, so the gu L2 warmer (which warms while polling
that gate) runs to its full range -- 12 MB/core of extra weight reads, 616
MB/token, which at 40k is thrown away again by the KV stream.

### R19: + gate the gu warm on `sl`

`if (sl < 4096) aiv_warm_gate(...); else aic_gate_poll(...)`.  Gates PASS
(0/1, 0/28, 300/2, 1000/1, 16000/1), real decode 300.1 tok/s.

| | R16 | R19 |
|---|---|---|
| ctx=1 | 319.7 | 320.2 |
| ctx=40k | 152.9 | **153.2** |

### The important finding

Removing 146 MB (1.5% of the token's bytes) bought only **+0.2%** of time, and
the profile already had AIC mte2 at 80%.  So at 40k the megakernel is **no
longer bandwidth-bound -- it is chain-bound**.  Consequences:

* the L2-bypass idea, even if the hardware had it, would have been worth ~0;
* further byte reduction (e.g. #3's weight padding, 0.4%) is worth less than its
  byte share, so it is not worth the pack/ring refactor;
* removing the gu warm at long context is worth ~0.5% (it was 2.4% relative to
  R18 only because R18 had inflated the warm window);
* the residual 13% @40k / 25% @ctx=1 is the per-layer dependency chain -- the
  AIC must wait for an AIV elementwise stage that produces the next gemv's whole
  input vector.  Only M>1 (batch / speculative verify) amortizes it.

`libmegak_r19.so` is deployed (`libmegak.so` == it, md5 b94003ca...).  Sources:
`mega_r19src.cpp` (current), `mega_r16src.cpp`, `mega_r13src.cpp`,
`mega_r15src.cpp`.

## R20 -- drop the dead "scores done" publish

R13 replaced the whole-slice scores barrier with per-group `PR_SCORES`
publishes; the old `aic_seq(SEQ_WANT(24+cid, R+2))` "scores done" word has had
no consumer since.  Removed (one scalar store + dcci per layer per AIC).
Gates PASS; 40k 153.4/153.1/153.6 vs 153.3/153.5/153.2 -> neutral, as expected
(the cost was a few ns).  Kept because it removes a misleading invariant.

## R21/R22 -- killing the softmax max pass, and the visibility window it hid

Idea: softmax is shift invariant, so `exp(x)` with a GLOBAL shift of 0 is as
correct as `exp(x - m_slice)` as long as nothing overflows f32 (needs x > 88) or
underflows bf16 (x < -87).  Measured worst scaled score on the real model over
sl = 1 / 4096 / 16384: **27** (`/tmp/opencode/probe_mx.py`), i.e. 60 e-folds of
headroom.  That would delete pass 1 entirely: 2.6 MB/layer of score traffic plus
the reductions, ceiling measured at 1.8% in the R13 probe.

R21 (pass 1 deleted) did exactly that and looked great -- then `sl=300` failed
**1 run in 6**, with `xn1P` wrong by 2-5% and everything downstream of it.

The first reading was wrong.  Restoring pass 1 (R22) to reintroduce the "settle"
margin did *not* help (3/8 still bad), and neither did:
* a whole-D-cache `dcci` fence on the AIC after every land poll (5/6 *worse*),
* ~2us of dummy work before the AIV's o-gemv read (3/6 still bad).

Because with `lstop=1` the first failing check is `xn2P` while `attnP`/`o_out`
pass, the corruption is a **uniform ~10% scale error**, not a stale tile.  A
uniform scale error means the scalar `den` is wrong.  And `den` comes from
`tPartL`, which R21 read *immediately after* the R+3 gate:

    producer: put_scalar16(part_l) ... then seq_write(R+3)
    consumer: poll(R+3) -> DataCopy(part_l) -> read

So a part_l MTE3 does not have to be visible to another core by the time the
follow-up flag word is -- the same flag/data gap R17 found, one buffer over.
A stale slice total moves `attn` by a *constant factor* for the whole group,
which the loose `t5` tolerance in `attnP`/`o_out` absorbs (attn ~2, t5 ~0.07)
and `xn2P`/`xn1P` then expose.

R23 fixed it at zero cost by reading `part_l` only *after* the partAcc
accumulation, i.e. giving it the ~1-2us the old `part_m` load used to occupy:

| | 40k | ctx=1 |
|---|---|---|
| R20 | 153.3 | 320.5 |
| R23 (combine restructured, pass 1 still there) | 153.7 | 322.0 |

8/8 clean at sl=300 (vs 3/8 failures before), and 0/1, 0/28, 300/2, 1000/1,
16000/1 all PASS.

R24 then removed pass 1 (the original goal) on top of the fixed combine:
**8/8 clean at sl=300**, 3/3 clean at 400/1000/3000/16000, full gate suite PASS,
real decode 300.0 tok/s.

| | R20 | R24 | |
|---|---|---|---|
| ctx=40k | 153.2 | **155.7** | +1.6% |
| ctx=1 | 320.3 | **322.1** | +0.6% |

Details that matter:
* probs in `probs2P` are now `bf16(exp(x))`, unnormalised, one global shift;
  combine sums the live slices' `partAcc` and divides by the summed `part_l`.
  Dead slices are skipped on `sblk == 0` (the old code skipped on `w == 0`,
  where `w = exp(m_s - mg)` was 0 exactly for dead slices).
* `part_m` is still published (from a cheap in-pass 2x-copy + `tree_max`) purely
  so `test_mega.py` keeps verifying the score values end to end.
* the gate refs for `part_l`/`partAcc` moved from `exp(x - max)` to `exp(x)`.
  That is a **reference-form change, not a tolerance change**; `MEGA_MAXSHIFT=1`
  selects the old form so pre-R24 binaries are still testable.
* the `sl=3000 lstop=1` `xnfP` marginal failure is pre-existing: R20 shows
  0.1523/0.1099/0.0938 against a 0.095 tolerance.

**Lesson for everything below**: every publish->read pair here has a ~1-2us
data-visibility window hidden behind the flag, and the current schedule covers
it with accidental slack.  A uniform scale error in `attnP`/`o_out` is invisible
to their tolerances -- always re-check a speedup with repeated `sl=300` runs, not
just once through the suite.

## R25 -- host side: the per-token round trip dominated ctx=1

`ab1.py` yet again saved us from a wrong model: 300 tok/s of "decode" at ctx=1
is 3.33 ms, but the kernel is 3.11 ms -- **~0.2 ms/token of host overhead**,
6%.  Breakdown per token (`model_run.py:step`): `xEmb.copy_`, `partAccP.zero_`,
`logits.fill_(-1e30)` (12.58 MB!), two *strided, synchronous* `.cpu()` copies
(~60-70us each) and a CPU argmax.

* the 12.58 MB fill cannot go away -- it is also a **race mask**: the kernel's
  per-span argmax scans a full `SPAN`, and with the mask removed it reads
  logits the lm gemv has not made visible yet (deterministic wrong argmax
  `134491` at every sl and every layer).  But only **row 0** is ever read (rows
  1..15 are fixpipe dead rows), so the fill is 786 KB instead of 12.58 MB.
* the two strided `.cpu()` calls became two *contiguous* `.cpu()` calls of the
  whole (replicated) 6 KB partial buffers; the strided pick is then free.

Real-weight decode, 24 new tokens: **300.0 -> 309.8 tok/s** (+3.3%);
40 new tokens: 308.4 tok/s.  `test_mega.py` gets the same treatment (its
`launch()` had the same fill), which is why the A/B numbers move too.

Two more round trips turned out to be pure launch overhead, not bytes:

* `xEmb.copy_` (a 4 KB H2D of one embedding row) cost ~24us/token.  The full
  embedding table is already resident, so the kernel's `xEmb` argument is now
  `emb.data_ptr() + token * H * 2` -- the row pointer itself.  No kernel change
  is needed (it is just a base pointer) and the copy disappears.
* `torch.argmax` on the 24x4 CPU partials plus the `view()[:,0].view()` chain
  cost ~35us of torch dispatch.  The reduce is now numpy on the already-copied
  buffer (~1us).
* the 40-argument `ctypes` list and the stream handle are built once, not per
  token.

Measured with the real 1.7B weights (`model_run.py`):

| | before R25 | after |
|---|---|---|
| decode tok/s, ctx=1 | 300.0 | **315.2** (+5.1%) |
| E2E ms/token, ctx=40000 | (6.73 est.) | **6.502** (153.8 tok/s) |

Per-token host overhead went 0.22ms -> 78us.  The rest is 2 synchronous D2H +
the pre-kernel launches; further work there is ~40us at best, i.e. 0.6% at 40k.

### State after R25
Kernel = R24 (`libmegak_r24.so`, md5 e986ba57...), source `mega_r24src.cpp`
(md5 f2020786...).  Gates: 0/1, 0/28, 300/2, 1000/1, 16000/1 PASS, `sl=300`
repeated 8/8 clean.

Final A/B (idle box, best of 3x30, harness identical, so this is the number the
later rounds should be compared against):

| | R19 (session start) | R24/R25 |
|---|---|---|
| ab1 ctx=40k | 153.3 | **156.2** (+1.9%) |
| ab1 ctx=1 | 319.9 | **325.0** (+1.6%) |
| real decode, ctx=1 | 300.0 | **315.2** (+5.1%) |
| real decode, ctx=40k (E2E) | - | **153.8** |

## R26 (not done) -- why the streaming items are on hold

The remaining ranked item was "stream the remaining gates" (per-slab qkv land,
per-chunk silu->down, PR_SCORES at 1024).  Deferred, for two measured reasons:

1. the weight gemvs share one `gemv_ring`, so a per-chunk K hand-off means
   polling inside that shared walk -- not a localized change; and
2. R21/R22 showed every new publish->read pair needs a ~1-2us slack budget on
   the consumer side, so each of these would have to be validated with repeated
   `sl=300` runs, and the wins are individually small (skew removal, not
   throughput).  If it is picked up, do the silu->down one first and check
   `sl=300` x8 before trusting it.

## R27 -- making the single-pass softmax model independent (carried shift)

R24's single-pass softmax is correct for *any* input only because the scaled
scores happen to peak at 27 on Qwen3; the code itself says "shift = 0", which
overflows f32 (x > 88) on a model with larger logits.  Fixing that was the one
open correctness item.

Why not a real online softmax: the running accumulator lives in **L0C** (one
`partAcc` per slice, written by exactly one Fixpipe per slice) and there is no
L0C scalar rescale.  Rescaling per chunk would mean ~8700 extra Fixpipe/L0C
round trips per token.  What R24 actually needs is not a rescale but a *safe*
shift: `exp(x - sh)` is exact for ANY `sh` (num and den carry the same
`exp(-sh)`), so `sh` only has to stay within ~88 of the max.

Design that landed (each AIV owns one `(head, slice)`, 3 slices per head):

* the three slices of a head MUST share one shift -- a plain-sum combine is
  only valid if the partials are commensurable (this was the first, failed,
  attempt: per-slice carried shifts made `sl=300` fail by a uniform factor);
* `li > 0`: `sh` = the value `part_m` holds for `(head, slice 0)`, i.e. the max
  that slice published on the **previous layer**.  One 64-float `DataCopy` (it
  covers both heads of the pair), no flag needed -- a whole layer of slack, so
  none of R22's flag-vs-data window applies;
* `li == 0`: no history exists, so every core *measures* slice 0's max (one
  extra pass over slice 0 only, ~2.5us/token @40k, ~11us @ctx=1).  All three
  cores read the same GM region, so they get bit-identical values and the
  single-pass exp can start immediately afterwards.

Two things that cost real debugging time:

* `dcci(...SINGLE_CACHE_LINE...)` + a raw scalar `gpartm[i]` read **faults the
  AIV** (`retCode 0x26`, `core id 28`) on dav-2201 -- and the slot stride makes
  the 2nd head's address 64B- but not 128B-aligned.  A plain `DataCopy` +
  `MTE2_S` + `GetValue` (the same idiom stage 5 already uses) works first try.
* a vector `Exp` anywhere inside stage 5 (the combine) still faults, so the
  LSE-merge variant (`w_s = exp(sh_s - mg)`) is off the table.  One shift per
  head is what keeps the combine Exp-free.

Gate support: `part_m` publishes the layer max (that is the carry) except on the
**last** layer, where it publishes the shift actually used -- the only way the
portable harness can rebuild `part_l`/`partAcc`, which are not shift invariant.
`test_mega.py` gained three explicit modes: default (R27+, read `part_m`),
`MEGA_SHIFT0=1` (R21..R25, global shift 0), `MEGA_MAXSHIFT=1` (pre-R21, slice
max).

Generality proof (new hook, zero cost when off): build with
`-DMEGA_SCORE_BUMP=200` (bisheng line, plus `Adds` hooks on both exp paths) and
run with `MEGA_SCORE_BUMP=200`.  Softmax is shift invariant, so the token must
not change -- but a shift-0 kernel *must* overflow (`exp(227)`), which is
exactly what the control build does.  Result @ `0/1`, `0/28`, `300/2`,
`1000/1`: **PASS with the bump, `nan` everywhere for a forced-shift-0 twin of
the same source**.  That is the property R24 could not state.

Gates: `0/1`, `0/28`, `300/2` (x8, then x4 on the final binary), `1000/1`,
`3000/1`, `16000/1` -- all PASS.  `3000/1`'s pre-existing marginal `xnfP`
failure is gone (it was the shift-0 range, not a race).

Cost, honestly: the layer-0 bootstrap + one `part_m` read per layer is **-0.3%
(40k 156.2 -> 155.9, ctx=1 325.0 -> 323.9, real decode 315.6 -> 315.0, E2E
40k 6.502 ms/token unchanged at 153.8)**.  It is
paid once per token; making it conditional on a usable carried value would
remove it but turns the first token's correctness into a host contract ("zero
the scratch buffer"), which is a worse trade than 0.3%.

Perf is otherwise identical to R24 (the combine, the score pass and the probs
path are untouched), so `libmegak_r27.so` = correctness fix only.

## R28 — grouped down-input gates (2026-10-27)

Starting state was the existing, uncommitted R27 source and binary; both were
preserved as `kernel/mega_baseline_20261027.cpp` and
`kernel/libmegak_baseline_20261027.so`. Source SHA256
`bc2aac3dfb2de6cb4acae96ba6336f578de506634f53a9f76bcb3950b9f73de4`,
binary SHA256 `9884bf3ec88567b66f9e4de6b0a634242decb22cfae899997ccaeabfef8275ad`.
The other pre-existing modified files were preserved.

### Profiling method and baseline diagnosis

NPU 4, 910B2C, CANN 8.5.0, measured frequency 1800 MHz. Sandbox device access
fails; hardware runs require execution outside the filesystem sandbox.
`msprof op` requires the **complete mangled kernel symbol**, not `main` or
`megak` (both were tried and yielded no data). It also rejects a writable
output directory with group/other write permission; output directories here
have mode 0700. No system or shared-directory permissions were changed.

Reproduce the final four captures with:

```bash
ASCEND_RT_VISIBLE_DEVICES=4 python profile_suite.py
python summarize_profiling.py > profiling/summary.json
```

`profile_suite.py` invokes `msprof op --aic-metrics=Roofline`, 10 launches,
5 profiler warmups, full `_Z5megak..._iii_mix_aic` symbol, on
`profile_mega.py --ctx {1,40000} --lib <baseline-or-final>`. Each capture
contains exactly 10 `OpBasicInfo` records. This fixed-position harness uses
real weights and initially zero KV history; it is a controlled timing load,
not a claim of valid natural-language prefill at 40k. The Python wall time
inside an instrumented process includes profiler overhead and is **not**
the kernel duration: all device numbers below come from the CSVs.

Initial 40k baseline: mean **6398.50 us** (10 samples). Final same-session
baseline/final captures are below; absolute clocks drift slightly, so also
use the interleaved A/B runs rather than comparing captures across time.
Across all AICs the initial baseline has MTE2 85.0%, cube 7.84%, scalar 40.7%;
AIV scalar 94.3%. Block 0 alone has MTE2 ~88%. This points at input streaming
and inter-stage latency rather than cube arithmetic throughput.

### Six experiments

| Candidate | Reference checks | Measurement / decision |
|---|---|---|
| Down waits for groups of 8 SiLU chunks, just before each A load | PASS | Keep: repeated A/B win |
| Q loaded into both L1 parities once | PASS | No further reproducible win; revert |
| 32-bit ring cursors (maximum 528 chunks) | PASS on stress/timing runs | No consistent improvement across both lengths; revert |
| Q retained in L0A throughout the score loop | PASS, including 28 layers at 40k | Slightly slower than grouped down alone; revert |
| Refill weight ring immediately after L1 slot release, before cube issue | PASS, including 28 layers at 40k | No further reproducible win; revert |
| Down A groups of 4 chunks | **FAIL at sl=0, one layer** (`d_out` maxdiff 0.1472, `xnfP` 0.125) | Reject immediately; no performance claim |

The retained change does not change dot-product order, BF16 rounding,
softmax, packing, weights, launch ABI or host code. Previously every AIC
waited for all 24 SiLU producers before loading its first 8 K chunks. Now
`gemv_ring` optionally polls only producers 0..7 before loading group 0,
8..15 before group 1, and 16..23 before group 2. Later producers can finish
while earlier chunks run. All other GEMVs keep their existing behavior.
The failed 4-chunk experiment illustrates why a finer granularity cannot
be accepted based on a throughput measurement alone on this implementation.

### Final device captures (10 samples each)

| Context | R27 mean | R28 mean | Latency reduction |
|---|---:|---:|---:|
| 1 | 3081.773 us | 3068.889 us | **0.418%** |
| 40000 | 6419.225 us | 6394.586 us | **0.384%** |

40k aggregate AIC MTE2 utilization increases 84.686% -> 85.694%; cube stays
7.81% -> 7.84%. Short-context MTE2 increases 66.307% -> 67.627%.
Raw reports: `profiling/final-{baseline,final}-ctx{1,40000}/`;
parsed statistics: `profiling/summary.json`.

Interleaved synthetic-weight A/B (6 rounds, 30 launches per round, reversing
binary order each round) confirms the direction. In the last three-way run:

| Context | Baseline median | Group-8 median | Reduction |
|---|---:|---:|---:|
| 1 | 3.084529 ms | 3.051741 ms | 1.063% |
| 40000 | 6.417643 ms | 6.393250 ms | 0.380% |

These are **harness wall times**, not interchangeable with profiler device
durations. Logs: `profiling/compare-early-refill.log`,
`profiling/compare-q-l0a.log`, `profiling/compare-synthetic-warm.log`.
Earlier real-weight fixed-context timings are in `compare-stream-down.log`
and `compare-cache-q.log`; their long-context token-stability assertions
failed even for baseline, so they are not end-to-end correctness evidence.
Repeating a 256-token packed history did not remove that baseline issue.

### Precision and stability

* `profiling/gates-stream-down.log`: original gates 0/1, 0/28, 300/2 x8,
  1000/1, 16000/1, all PASS.
* `profiling/gates-final.log`: **70 additional gates**, all PASS:
  0/1 x32, 300/2 x32, 0/28 x2, 16000/1 x2, 39999/28 x2.
  Uses the existing op-by-op Torch reference and existing tolerances;
  no tolerances in `test_mega.py` were changed.
* `profiling/compare-decode.log`: real checkpoint, fresh caches, three
  prompts (two English, one Chinese), **64 generated tokens per prompt**;
  baseline and final token sequences match exactly for all three.
* Long-context repeated synthetic launches show baseline's own logit drift
  (~0.08–0.10 max absolute difference), including near-tie argmax changes.
  The A/B harness records this; its additional drift guard is 0.2, distinct
  from and not a replacement for the independent reference gates. These
  results do not establish bitwise determinism or long-context real-model
  sequence equivalence. Such baseline behavior remains an accuracy risk to
  investigate before further aggressive synchronization changes.

All six variants and logs remain available for reproduction. Current
`kernel/mega.cpp` and `kernel/libmegak.so` contain only the retained group-8
optimization. R28 reaches a measured local plateau for these experiments;
**no claim of a global performance limit** is made. More work should start
with the long-context stability issue and per-stage timing, rather than
reducing producer/consumer slack again without stronger visibility checks.

## R29 — compact GEMV inputs, paired GU producers, streamed LM argmax (2026-10-27)

Continues from R28, preserved as `kernel/mega_r28src.cpp` and
`kernel/libmegak_r28.so`. Existing staged work was preserved; nothing was
committed or staged by this round.

### Diagnose before changing the schedule

Added compile-time `MEGA_TIMELINE` timestamps in the extra sync workspace.
Only AIC 0/8/16 and AIV 0/sub0 are traced; each owns whole 128-byte lines.
Normal builds compile out all timestamp operations. The selected library
exports host-only `megak_layout()` metadata so `model_run.py` and
`test_mega.py` select matching packing automatically. Explicit environment
settings that disagree with a new binary are rejected before launch.
Legacy libraries retain the old environment override mechanism.

`phase_profile.py` captures timestamp differences and device events. Its
microsecond estimates are intrusive diagnostics, not performance results.
The first device event returned 0.16 us on the initial run; that invalid
startup sample is excluded from `phases-summary.json`, and the script now
rejects startup durations below 100 us. Valid counter estimates are around
50 MHz. `phases.log` retains the original record for audit.

Representative AIC 0 means across layers/samples:

| Stage, us/layer unless stated | R28 ctx=1 | R29 ctx=1 | R28 ctx=40000 | R29 ctx=40000 |
|---|---:|---:|---:|---:|
| Wait for RMS1 input | 6.81 | 4.72 | 6.13 | 4.00 |
| Wait for RMS2 input | 7.44 | 5.78 | 7.95 | 6.03 |
| Down GEMV including grouped input waits | 17.06 | 16.04 | 19.64 | 18.48 |
| Entire LM GEMV, us/token | 405.99 | 409.69 | 406.71 | 411.74 |

Baseline 40k QK/score-publish ~59.2 us/layer and PV ~50.2 us/layer.
Baseline argmax tail is only ~6–7 us/token. The evidence favors shortening
input preparation/hand-off, not trying to speed up the cube arithmetic or
expecting a large standalone argmax gain. Streamed argmax overlaps most of
its work with LM, but adds producer progress checks, so LM itself need not
get faster. Its diagnostic AIV interval now includes the stream wait and
is labelled accordingly.

### Implemented and retained

1. **Compact GEMV A inputs.** Each 256-element input chunk stores two
   contiguous bf16 ND rows (512 elements) instead of the old 16-row
   broadcast fractal (4096 elements). One MTE2 Nd2Nz copies/expands eight
   chunks into L1. Row 0 carries the value and row 1 remains valid for the
   existing landing sentinel; rows 2..15 are never consumed as outputs.
   RMS1/RMS2/final packs emit eight compact chunks in one 8KB transfer;
   attention and SiLU producers use the same compact layout. Existing GM
   allocation sizes and launch parameter list remain unchanged.
2. **Paired GU layout.** Each AIC produces its own `[gate256, up256]`
   slab. SiLU waits for that single producer instead of the two original
   gate/up slabs. CPU/checkpoint and random-test weight packers are updated;
   the op reference unpermutes GU outputs before comparing them.
3. **Streamed final argmax.** LM publishes four per-span progress values
   after checking the actual end-of-chunk logits sentinel. AIV consumes
   each 2048-logit chunk as it becomes ready instead of waiting for the
   complete span. The tile remains 96 and the previous final completion
   publish is retained. No unchecked ready-word-only publication is used.

The changes preserve bf16 values, the K reduction order for each output,
and the existing softmax algorithm. No quantization, batching, or speculative
execution is involved.

`kernel/build.sh` accepts an optional output filename and compile settings:
`MEGA_TIMELINE` (default 0), `MEGA_LM_TILE` (96), `MEGA_COMPACT_A` (1),
`MEGA_GU_INTERLEAVE` (1), `MEGA_LM_STREAM` (1). Normal `bash kernel/build.sh`
builds the selected configuration. Debug libraries use separate filenames.
`MEGA_LIB` selects the runtime binary before checkpoint packing.

### Rejected experiments

* **LM N=128:** all reference gates pass, but ctx=1024 and 8192 regress.
  In the four-way run, ctx=8192 was 3.847 ms vs baseline 3.760 ms. The
  tile is not selected despite reducing the number of N iterations.
* **Next-stage GU/down prefetch before the preceding GM landing:** gates
  pass, but all four context lengths regress slightly (~12–15 us/token).
  Restored the existing stage order.

Standalone and combination comparisons are in `bench-stage-stream.log`,
`bench-compact.log`, and `bench-combined-gu.log`. Reversing library order
on alternate rounds limits drift. Reference checks, rather than repeated
output equality, decide whether a candidate passes the numerical gate.
The timing harness records drift warnings separately after the R28 control
itself exceeded its original 0.2 diagnostic threshold at ctx=8192; no
reference tolerances were relaxed.

Last four-way interleaved run, medians of six rounds x30 launches:

| Context | R28 | Compact only | Compact + streamed argmax | + paired GU (selected) |
|---|---:|---:|---:|---:|
| 1 | 3.058632 ms | 2.920019 ms | 2.915063 ms | **2.892764 ms** |
| 1024 | 3.247869 ms | 3.107894 ms | 3.103896 ms | **3.082683 ms** |
| 8192 | 3.759349 ms | 3.617345 ms | 3.614000 ms | **3.594165 ms** |
| 40000 | 6.389681 ms | 6.231510 ms | 6.227459 ms | **6.204671 ms** |

These are synthetic-reference harness wall times, not profiler device times.
The selected configuration reduces latency by about 5.42%, 5.09%, 4.40%,
and 2.90% respectively. Independent-process Roofline results are recorded
below when the final capture completes.

### Accuracy evidence and unresolved baseline behavior

* Individual tile, prefetch, compact, streamed, and paired-GU variants passed
  the existing Torch op-by-op gates, including 39999/28. The selected build
  additionally passed **70 stress gates** in `stress-final.log`: 0/1 x32,
  300/2 x32, 0/28 x2, 16000/1 x2, 39999/28 x2. Tolerances are unchanged.
* Three real-checkpoint prompts, 64 generated tokens each, match exactly
  between baseline, baseline repeat and selected build after scratch state
  is defined. `isolated-*.json` and `isolated-suite.log` run each binary in
  its own process, with matching packing and one library per process.
* **4096-token real-prefix generation is still not deterministic in R28.**
  Even two independent processes running the same R28 binary disagree
  (first difference at generated token 35 in the final isolated test).
  The candidate also differs, so long-prefix sequence equivalence is NOT
  established. The isolated suite intentionally exits with a failure and
  its logs are retained; it is not counted as a passing accuracy test.
* Reusing multiple libraries in one process also produces baseline-control
  differences. Fixed-prefix teacher-forcing diagnostics additionally show
  differences involving the `-1e30` mask value in both controls/candidates
  (`teacher-final.log`). This must not be dismissed as only near ties.
  Root cause remains unresolved; these runs do not certify long-context
  real-model accuracy.
* Runtime scratch is initialized once to zero at construction rather than
  left uninitialized. This removes an initial-state confounder observed
  on short prompts, without adding per-token zeroing, but does NOT claim
  to fix the long-prefix baseline problem. Comparison programs explicitly
  reset scratch between independent prompts.

The arithmetic/reference gates and short-prompt checks support the selected
layout optimizations. They do not prove full long-context equivalence;
R28 remains available for comparison. Resolve the baseline stability problem
before relying on more aggressive synchronization changes or claiming a
performance/accuracy endpoint.

### Final independent-process Roofline verification

All four captures contain 10 records, with 5 profiler warmups. These are
`OpBasicInfo` device durations; profiler-instrumented Python wall time is
not used. `profile_mega.py` selects the binary before importing the runtime,
so baseline and final checkpoint packing both match their own binary.

| Context | R28 mean | R29 mean | Latency reduction |
|---|---:|---:|---:|
| 1 | 3060.134 us | 2904.298 us | **5.092%** |
| 40000 | 6398.972 us | 6211.561 us | **2.929%** |

At 40k AIC MTE2 utilization rises 85.56% -> 87.42%; at ctx=1 it rises
67.84% -> 70.79%. Reported aggregate main-memory reads fall from 8602.81
MiB to 8384.06 MiB per 40k token; UB->GM falls 132.48 -> 51.54 MiB.
This corroborates the compact-layout traffic reduction independently of
the interleaved timing harness.

Raw captures: `profiling/r29/roofline-{baseline,final}-ctx{1,40000}/`.
Statistics: `profiling/r29/roofline-summary.json`. Sequential verification
completed in `verification-suite.log`; it covers the 70 op reference gates,
phase diagnostics, and the four captures. The separate isolated long-prefix
sequence test failed as described above and is not hidden by this completion.

Reproduce the selected build and checks:

```bash
bash kernel/build.sh
ASCEND_RT_VISIBLE_DEVICES=4 python validate_final.py
ASCEND_RT_VISIBLE_DEVICES=4 python bench_variants.py --mode bench \
  --variants libmegak_r28.so:96:0:0 libmegak.so:96:1:1
```

For a fresh Roofline run, use `profile_r29_suite.py` with a fresh output
directory (the script refuses to mix new data into an existing capture).
Use `bench_variants.py`, not the older same-layout comparison scripts,
when mixing R28/R29 layouts.

Final source SHA256: `4b17b4ccbeb24a5ce6fea6d99cf9c3541fba420cb4e87e6c94fb951cc987e7c5`.
Final binary SHA256: `8bc48201f39fa9bbf99fa92b6afc436e2ed7ab89ddbc630ddc113212f1356005`.
Snapshots: `kernel/mega_r29src.cpp`, `kernel/libmegak_r29.so`.



## R30: long-context accuracy closure (2026-10-27)

The validated default keeps compact GEMV A, interleaved gate/up weights and
streamed LM argmax. No quantization, batching or accuracy-tolerance relaxation
is used. Release snapshots are `kernel/mega_r30src.cpp` and
`kernel/libmegak_r30.so`; the latter is identical to default `libmegak.so`.

### Root causes and changes

- The default 96-thread **CPU RoPE cosine path was not reproducible**. Twelve
  fresh processes had identical inverse frequencies and frequency tensors but
  different cosine tables. Two calls in one process could even disagree
  between the two supposedly identical halves. One captured pair differed in
  1,163,534 elements, by up to 0.0001511574. Independent decoders were therefore
  consuming different inputs. `model_run.make_rope_tables()` keeps FP32
  frequencies, evaluates scalar FP64 libm sin/cos once, rounds to FP32 and
  duplicates each result. Eight fresh processes now produce identical tables;
  max `abs(cos²+sin²-1)` is 1.1921e-7. This is startup work, not decode work.
- Hardware `Rsqrt` approximation introduced RMSNorm error before BF16 rounding.
  **Two scalar Newton steps** recover FP32 accuracy. Neither the earlier 16k
  terminal error above 2% nor the one-step-Rsqrt maximum-length error of 2.192%
  was waived. The final build passes the original relative-L2 <= 0.02 / cosine
  >= 0.9998 thresholds, including the maximum-length boundary.
- A GU dead-row marker did not certify the row SiLU actually consumes. A
  single-layer reproduction captured identical QKV, attention, normalization
  and final GU tensors but 128 incorrect SiLU values. GU now checks a live
  row-0 marker for every 128-column tile. Empty down slabs on AICs 22/23 also
  acknowledge hP chunks 8..23 before the next layer can rearm those markers.
- `FIX_S` orders destination completion before scalar landing checks;
  `S_MTE2` orders sentinel stores/cache cleans before following work.
  Score/PV ping-pong uses distinct event IDs: `FetchEventID` peeks without
  allocating. GEMV drains its ring before those channels are reused.
- Combine/softmax UB reloads wait for vector readers; V_S/S_V protect scalar
  readback and scratch reuse; grouped down-A reloads wait for MTE1. Denominator
  publication checks GM readback. Carried maxima use alternating history
  banks to protect preceding-layer readers.
- Direct ctypes launch bypasses torch_npu's host queue. Production and test
  launchers synchronize the current stream after queued clears and before
  launching. Merely obtaining the raw handle is insufficient.

Broad extra vector barriers, KV-append readbacks and disabling GU interleave
were investigated but are not production changes. Final accuracy/timing
controls use **the same production source, precision and synchronization**,
with only compact-A, GU interleave and LM streaming disabled. Their complete
logits match the release bitwise. Earlier conservative-vector-fence controls
were also used during diagnosis; failed intermediate experiments remain in
`profiling/r30/` and are not counted as final passes.

### Final accuracy evidence

For each natural-prefix case, three independent processes (control, control
repeat, release) run three short prompts plus a real-checkpoint long prefix,
then generate 128 tokens per prompt. All token IDs, every FP32 vocabulary
logit and the **complete packed K/V buffers** match byte for byte. 4k/16k ran
on devices 7/8; the 40k processes ran on devices 10/11/9. Device 4, and later 5,
became occupied by other processes; their historical timings are not used for
final performance comparisons.

| Prefix / max live context | CPU-check position | Max operator-local relative L2 | Independent terminal logits relative L2 | Terminal cosine |
|---:|---:|---:|---:|---:|
| 4,096 | 4,223 | 0.00009680 | 0.01368172 | 0.99994087 |
| 16,384 | 16,511 | 0.00033745 | 0.01039816 | 0.99997061 |
| 40,000 | 40,127 | 0.00028441 | 0.00631288 | 0.99998081 |
| 40,960 max | 40,959 | 0.00029407 | 0.00943002 | 0.99995565 |

The CPU validator reads original checkpoint weights, not packed weights. Each
case enforces 308 operator-local checks (11 x 28), an independently propagated
full-trajectory final norm/logit check, equal greedy argmax, and eight strict
same-input intermediate-byte repetitions. Intermediate full-trajectory
diagnostics are retained separately from enforced operator-local checks.
Records: `profiling/r30/rsqrt2-accuracy-{4096,16384,40000}/` and
`profiling/r30/final-accuracy-summary.json`.

Additional final-release checks:

- `rsqrt2-cached-max40960/summary.json`: four fresh processes extend saved real
  40k KV by **833 steps**, through position 40959 / live context 40960. All
  tokens and complete vocabulary logits are bitwise identical. The final
  boundary also independently passes CPU validation (`final-cpu-max40960.json`).
- `final-random-gates.log`: 70 standard random-weight gates, an additional
  maximum-position gate, and 32 strict full-depth maximum-position repeats.
- `final-alternating.log`: 4,096 full-depth calls on real 40k KV, alternating
  four tokens and comparing every full vocabulary vector bitwise.
- `final-short.log`: 128 repeated short-prefill/128-token generations, checking
  every complete vocabulary vector bitwise and validating argmax by direct
  ACL D2H readback.
- `rope-scalar-processes.log`: eight independent complete-table hashes and
  rotation-unit-circle checks pass.

### Final performance

`final-bench.log` alternates control/release order across six rounds of
30 timed launches on device 7. Every sampled output matches the control
bitwise. These are launcher wall times, including clears and synchronization.

| Context | Same-precision control median ms | Release median ms | Latency reduction |
|---:|---:|---:|---:|
| 1 | 3.11719 | 2.97551 | 4.54% |
| 1,024 | 3.29345 | 3.15438 | 4.22% |
| 8,192 | 3.81946 | 3.68353 | 3.56% |
| 40,000 | 6.44234 | 6.29530 | 2.28% |

Fresh **Roofline** captures of the final two-step-Rsqrt build use device 6,
15 distinct-tag launches, discard the first five and retain ten steady samples.
These device-kernel times are separate from the launcher wall times above.

| Context | Control mean us | Release mean us | Latency reduction |
|---:|---:|---:|---:|
| 1 | 3105.248 | 2936.361 | 5.44% |
| 40,000 | 6440.606 | 6265.543 | 2.72% |

Raw records: `profiling/r30/roofline-rsqrt2-*`; collected summary:
`profiling/r30/final-roofline-summary.json`. Application replay is intentional:
the kernel's monotonic tags cannot safely be reused by kernel replay. The CLI
warm-up is zero; the five discarded warm-up launches are real distinct-tag
application launches. The final release's mean AIC MTE2 occupancy is 69.66%
at context 1 and 86.29% at 40k; streaming weights/KV remains the dominant cost.
AIV scalar ratios include synchronization polling, not just arithmetic.

Reproduction (select an idle healthy device; profiling creates a fresh label):

```bash
bash kernel/build.sh
MEGA_COMPACT_A=0 MEGA_GU_INTERLEAVE=0 MEGA_LM_STREAM=0 \
  bash kernel/build.sh mega.cpp libmegak_r30_control.so
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 ASCEND_RT_VISIBLE_DEVICES=6 \
  python long_accuracy_suite.py
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 ASCEND_RT_VISIBLE_DEVICES=6 \
  python bench_variants.py --mode bench --strict \
  --variants libmegak_r30_control.so:96:0:0 libmegak.so:96:1:1
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 ASCEND_RT_VISIBLE_DEVICES=6 \
  python profile_r30_suite.py
```

Final source SHA256: `a6bc5c2b884b7a2b15feef96e36a82cb54e1b2c2a0d4d88aeba9c343934b3737`.
Final binary SHA256: `355ee25d56368e96c61b58db2be9c8fe19b8a79fbf0f5ab1e1ea02b6bc6b77be`.


## R31 — dedicate sub1 AIVs to GU prefetch (2026-10-28)

Fresh profiling of the R30 release preceded selection of the retained change.
No quantization, batching, arithmetic-order, weight-layout or host-ABI changes.
Baseline source and binary were saved as `profiling/r31/mega.baseline.cpp` and
`kernel/libmegak_r31_baseline.so`. Experiments use separate libraries; the
production `kernel/libmegak.so` is rebuilt from the selected source.

### Diagnosis and retained change

The baseline ctx=1 Roofline reports mean AIC MTE2 occupancy 69.35%, cube
7.63%, and AIV scalar 94.70%. At ctx=40000, MTE2 occupancy is 86.66%.
The source still replicated norm, RoPE, residual, SiLU, argmax and gate
polling on sub1, although only sub0 publishes model state. Sub1's useful
work is GU weight warming for sl<4096. It now runs that prefetch loop alone,
draining both UB/event slots after each layer. At larger contexts it returns.
The existing seq-word protocol has no full-plane barrier requiring sub1.
Sub0 arithmetic and its gates remain unchanged. Less redundant computation
and polling is the likely explanation for the gain; Roofline ratios alone
do not distinguish those contributions.

### Iteration results

Every benchmark alternates library order each round and checks complete
vocabulary logits bitwise against the baseline. Failed or neutral candidates
are retained only as diagnostic source snapshots/libraries and logs.

| Experiment | Result / decision |
|---|---|
| Retain all eight K=2048 activation chunks in 64KB L0A across N tiles | Reference gates pass; 8–13 us slower across four contexts; reject |
| Bounded ring indices instead of 64-bit modulo | No consistent improvement; reject |
| Seven 48KB ring slots for N=96, six 64KB slots for GU | Small short-context fluctuation, long-context regression; reject |
| Sub1 dedicated GU prefetch worker | Reference gates and bitwise checks pass; retain |
| KV CACHE_MODE_LAST hint on retained version | No consistent improvement; reject |
| Remove sub1 entirely, including GU prefetch | Short contexts regress by about 60–70 us; reject |
| Delay sub1 prefetch until all QKV producers finish | ctx=1 regresses about 40 us; reject |
| Skip the first six GU chunks instead of four and rebalance warm ranges | ctx=1024 regresses about 25 us; reject |

Subsequent refinements found no additional stable gain. This is a measured
local plateau over the tested schedules, not proof of a global hardware limit.

### Final device Roofline

Both libraries use real checkpoint weights and a fixed-position, initially
zeroed KV prefix as a controlled timing workload. These captures do not
represent a natural 40k prefill. Short-context captures use device 6 and
long-context captures device 8; each pair stays on the same device. The
profiler collected 15 distinct-tag application launches, discarded the first
five and retained ten samples. Kernel replay cannot safely reuse this
kernel's monotonic launch tags. Python timings during profiling include
instrumentation and are not kernel latency.

| Live context | Baseline mean us | Selected mean us | Latency reduction |
|---:|---:|---:|---:|
| 1 | 2946.932 | 2836.823 | 3.74% |
| 40,000 | 6253.722 | 6205.542 | 0.77% |

Raw CSV locations and all steady samples: `profiling/r31/roofline-summary.json`.
The initial abbreviated-symbol capture selected no kernel and was terminated;
it is excluded. Its process exited before any retained baseline samples.

Final launcher timings (device 7, ten alternating rounds x60 launches;
medians include clears and synchronization):

| Live context | Baseline ms | Selected ms | Latency reduction |
|---:|---:|---:|---:|
| 1 | 2.975461 | 2.903743 | 2.41% |
| 1,024 | 3.152642 | 3.101041 | 1.64% |
| 8,192 | 3.678037 | 3.615065 | 1.71% |
| 40,000 | 6.288116 | 6.235019 | 0.84% |

### Accuracy and release checks

- `gate-worker.log`: both libraries pass the existing random-weight,
  per-operator gates at short/medium/long contexts and full depth at 40k.
- `accuracy-4096.log`: three short prompts plus a real 4096-token prefix,
  128 generated tokens, baseline/repeat/selected in independent processes.
  All tokens, full vocabulary-logit hashes and complete packed K/V hashes
  match bitwise. The selected version also passes independent CPU reference
  validation and eight strict intermediate-state repetitions at position 4223.
- `accuracy-max40960/summary.json`: four independent processes extend saved
  real KV at position 40127 by 833 steps through position 40959. All tokens
  and every FP32 vocabulary logit are bitwise identical to baseline.
- `alternating.log`: 1024 full-depth calls on real 40k KV, alternating four
  input tokens, compare every full vocabulary vector bitwise; all pass.
- `bench-final.log`: every sampled output matches baseline bitwise.
- `build-final.log`: production build passes. The production library and
  profiled/validated `libmegak_r31_worker.so` have identical SHA256 hashes.
- `git diff --check`: passes.

Reproduce a capture (use ctx=40000 for the long case, and give each capture
its own fresh output directory):

```bash
bash kernel/build.sh
ASCEND_RT_VISIBLE_DEVICES=6 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONPATH=. \
msprof op --aic-metrics=Roofline --replay-mode=application \
  --kernel-name=_Z5megakPhS_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_iii_mix_aic \
  --output=/home/tilelang/zuochuanuong/prof/mega-asc-r31-reproduce \
  --launch-count=15 --warm-up=0 \
  python debug/profile_mega.py --ctx 1 --iterations 10 --lib libmegak.so
python profiling/r31/summarize.py
```

The summary script reads the four recorded capture directories, requires
15 records per capture and discards the first five. Snapshots and per-candidate
build/gate/benchmark logs are under `profiling/r31/`.

Final `kernel/mega.cpp` SHA256: `bfb8d9872ea3a2644bd2ba214a2a53b8798cf9ab7880dfec4dc858b15220388b`.

Final `kernel/libmegak.so` SHA256: `b3c8374bd209b723ad81a476e28ac3db08acd665cfec6a66a2fbf46675371dd1`.


## R32 — implement and measure pipeline-bubble reduction proposals (2026-10-28)

**No R32 candidate is retained.** Production source and `libmegak.so` are
identical to the R31 release at the start of this round. All implementations,
build logs, gates, alternating timings and profiler summaries are preserved
under `profiling/r32/`. This round does not establish that the remaining
bubbles are unavoidable; it rejects these particular implementations.

### Implementations and accuracy

1. **L0C double buffering:** alternate output tiles between two 16KB
   accumulator regions and FIX_M event IDs 0/1. A buffer is waited only when
   reused, and both outstanding events are drained before return. GM output
   landing checks and arithmetic order remain intact. Existing reference
   gates pass, including full depth at 40k; sampled complete logits are
   bitwise identical. Snapshot: `mega.l0c.cpp`; logs: `gate-l0c.log`,
   `bench-l0c.log`.
2. **GU K=128 L0A/L0B double buffering:** retain K=256 GM/L1 packing, select
   two contiguous halves via LoadData, and alternate the L0 destinations.
   Initial implementations trigger hardware L0 read/write conflicts.
   Moving both A buffers to separate 32KB halves does not resolve correctness:
   device 9 reports conflicts; device 10 initially matches but then fails
   GU/h/down/final-norm gates on a repeated launch. Changing the ring refill
   placement still reports L0B conflict on device 7. These are unresolved
   implementation/synchronization failures, not evidence that hardware
   double buffering is unsupported. No timing from these variants is used.
   Snapshots/logs: `mega.gu128-*.cpp`, `gate-gu128*.log`.
3. **GU serialized control:** single-destination K=128 with explicit
   PIPE_ALL fences, in addition to per-transfer and per-Mmad events, passes
   existing reference gates. Its complete sampled logits match the baseline
   bitwise, but latency increases substantially. It is a diagnostic control,
   not a successful double-buffer implementation. Snapshot:
   `mega.gu128-fenced.cpp`; `gate-gu128-fenced.log`, `bench-gu128-fenced.log`.
4. **Attention QK/PV group interleave:** for layers after layer 0 and slices
   with multiple 4096-position groups, issue QK0, QK1, PV0, QK2, PV1, etc.
   Preserve the PV accumulator in a separate L0C region across QK groups.
   Layer 0 retains the original score bootstrap order to avoid waiting for
   softmax before its shared shift can be computed. The group prologues,
   drains, score publication and probs progress gates remain explicit.
   Reference gates and complete-logit bitwise benchmarks pass, but 40k
   latency regresses. Snapshot: `mega.attention.cpp`; generator:
   `make_attention.py`; `gate-attention.log`, `bench-attention.log`.
5. **Lighter FIX overlap:** keep a single accumulator and defer FIX_M wait
   until after the next tile's initial L0 load, immediately before Mmad.
   Gates and complete-logit bitwise comparisons pass; no stable speedup.
   Snapshot: `mega.fixoverlap.cpp`; corresponding gate/bench logs.

### Alternating A/B launcher measurements

Device 7; eight rounds x60 timed launches per variant, except the fenced
GU control uses six rounds x60. Order reverses each round. Values are medians
including clears/synchronization, and each row uses its own paired baseline.
Positive change means slower. Normal candidates cover contexts 1/1024/8192/
40000; the clearly slower GU control covers 1/40000.

| Candidate | Context | Paired baseline ms | Candidate ms | Latency change |
|---|---:|---:|---:|---:|
| L0C double | 1 | 2.907704 | 2.913670 | +0.205% |
| L0C double | 40,000 | 6.245473 | 6.245885 | +0.007% |
| Attention interleave | 1 | 2.904242 | 2.910956 | +0.231% |
| Attention interleave | 40,000 | 6.236413 | 6.285867 | +0.793% |
| Deferred FIX wait | 1 | 2.885864 | 2.890691 | +0.167% |
| Deferred FIX wait | 40,000 | 6.238917 | 6.243781 | +0.078% |
| GU K128 fenced control | 1 | 2.900284 | 3.256418 | +12.279% |
| GU K128 fenced control | 40,000 | 6.254462 | 6.496548 | +3.871% |

All successful candidate timing samples report zero full-logit difference.
Failed asynchronous GU variants have no accepted performance results.

### Fresh device Roofline

Application replay, 15 distinct-tag samples, first five discarded, ten
steady samples retained. Short-context pair uses device 6; long-context
pair device 8. Fixed-position timing harness uses real checkpoint weights
and initially zero KV history, rather than natural long prefill. Complete
CSV paths, durations and pipeline ratios: `roofline-summary.json`.

| Candidate | Context | Baseline mean us | Candidate mean us | Latency change |
|---|---:|---:|---:|---:|
| L0C double | 1 | 2840.694 | 2842.720 | +0.071% |
| Attention interleave | 40,000 | 6208.956 | 6269.703 | +0.978% |

The first L0C profiler invocation had a mistyped mangled kernel name and
selected no kernel; it is excluded. `profile_one.py` derives the exact symbol
from the successful baseline capture to avoid manual transcription.

The measurements do not support the earlier expectation that these bubble
reductions would improve end-to-end latency. A plausible explanation is that
L0C destination writes are already cheap (two live rows), while extra buffer/
event bookkeeping offsets saved waits. Attention interleave additionally
breaks the continuous KV stream with group drains/prologues; the existing
implementation already overlaps softmax with the score tail. These are
interpretations, not isolated per-instruction causal measurements.

### Reproduce and release state

```bash
# Rebuild a diagnostic implementation, preserving the production library.
bash kernel/build.sh ../profiling/r32/mega.l0c.cpp libmegak_r32_l0c.so
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONPATH=. \
  python debug/bench_variants.py --mode bench --strict --rounds 8 --iterations 60 \
  --variants libmegak_r32_baseline.so:96:1:1 libmegak_r32_l0c.so:96:1:1
# Use a fresh --label for every capture.
python profiling/r32/profile_one.py --ctx 1 --lib libmegak_r32_l0c.so \
  --label reproduce-l0c --device 6
python profiling/r32/summarize.py
```

No real-checkpoint long-trajectory accuracy campaign was run for candidates
that failed correctness or regressed in paired timing; none is released.
The default source was restored byte for byte; the production library was
never overwritten during experiments. The previously generated R31 pipeline
diagrams therefore still describe the current production implementation.
`git diff --check` passes.

Restored `kernel/mega.cpp` SHA256: `bfb8d9872ea3a2644bd2ba214a2a53b8798cf9ab7880dfec4dc858b15220388b`.

Restored `kernel/libmegak.so` SHA256: `b3c8374bd209b723ad81a476e28ac3db08acd665cfec6a66a2fbf46675371dd1`.

## Restore TileFoundry CLI check for the R31 runtime twin (2026-10-28)

The production kernel and library are unchanged by this restoration:
source `bfb8d9872ea3a2644bd2ba214a2a53b8798cf9ab7880dfec4dc858b15220388b`,
binary `b3c8374bd209b723ad81a476e28ac3db08acd665cfec6a66a2fbf46675371dd1`.
The old `runtime_model.py` had fallen behind the production runtime. It now
reads `megak_layout()`, packs chunk-major weights and paired GU rows, stages
K caches in contiguous 256-position tiles (and reads the appended rows with
the same layout), allocates the history workspace, defines scratch state,
masks logits padding and synchronizes queued work before the raw launch.
The device-0-only assertion is removed; select an idle healthy device.

`acts` and `prepared` are restored relative symlinks to the neighboring npuir
example's saved activations and HIR-layout checkpoint. Only those data are
shared; the CLI candidate remains this directory's AscendC library. The new
`check.sh` keeps every original tolerance and supports explicit data locations
through `MEGA_CHECK_ACTS`, `MEGA_CHECK_WEIGHTS`, and `MEGA_CHECK_CTX`.

Reproduce the original gate (device 6 was healthy and idle for this run):

```bash
QWEN3_CKPT=/home/tilelang/zuochuanuong/weights/qwen3_1_7b \
ASCEND_RT_VISIBLE_DEVICES=6 \
  bash check.sh --json profiling/check-r31/check-pos255.json
```

All 28 layers, original fixture pos=255 / ctx_len=256, CLI exit 0:

| Output | Comparison | Result |
|---|---|---|
| logits | allclose 0.15/0.05; rel-L2 <=0.02 | PASS; rel-L2 0.0071062322 |
| next token | equal | PASS; token 323 |
| K rows | allclose 0.3/0.05; rel-L2 <=0.05 | PASS; rel-L2 0.0039665004 |
| V rows | allclose 1.0/0.05; rel-L2 <=0.05 | PASS; rel-L2 0.0113508666 |

Additional full-depth CLI cases at pos0 (ctx256), pos256 and pos257 (ctx512)
pass every original predicate. Extended rows in the latter fixtures are zero;
these are controlled layout boundaries, not natural-prefix generation.
CPU tests additionally check pack/read roundtrips, unwritten-row zeros and an
independent device-address oracle at prefixes 0,1,15,16,255,256,257,511.

An extra fixture repeats the saved 256-token KV history to fill ctx512 and
checks pos511 with a nonzero second K tile. It **fails** V-row allclose with
max_violation 0.2450195551 (V rel-L2 0.0161847345 still passes); logits, token
and K rows pass. The R30 baseline `libmegak_r31_baseline.so` produces exactly
the same reported output metrics and fails the same predicate. This is not
counted as a pass or dismissed because the fixture is synthetic: arbitrary-KV
coverage under the original tolerances remains incomplete. No production
kernel, HIR reference or tolerance was changed to hide it.

All CLI JSONs, the failing control, CPU layout log, fixture variants and the
hash/coverage summary are in `profiling/check-r31/`. This work restores the
established CLI gate; it does not constitute a fresh long-prefix accuracy or
performance campaign.
