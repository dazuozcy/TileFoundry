# OPTLOG — Qwen3-1.7B single-launch mega decode step (Ascend 910B2C)

One `rtKernelLaunch` per decode token (verified: `generate_npu_wrapper_src` in
`tilelang/jit/jit_npu.py` emits exactly one launch site; `run.py` calls the
kernel once per step). All 28 layers + attention over the live context + the
tied LM head run inside that one launch on 24 aicore blocks (cid = sid*8+kv).

## What the analyzer says (`tilefoundry analyze`, ctx_len=1024)

    f32 logical flops/step:  5,526,761,873
    bf16 logical flops/step:      1,236,992
    largest values:
      622,362,624   LM head            (2 * 2048 * 151944)
       50,331,648   gate/up GEMV x28   (2 * 2048 * 12288)
       ...          qkv/o/down/score GEMVs per layer

Caveat, stated plainly: the HIR casts operands to f32 before every matmul so
that the interpreter reproduces the kernel's exact numerics (bf16 inputs, f32
cube accumulator, no intermediate bf16 landing). The analyzer prices those as
f32 flops. The kernel executes the same dots as bf16 cube ops with f32
accumulators, so the real compute work is ~half the f32 figure. The step is
nowhere near compute-bound either way (see roofline below), so this does not
change any conclusion.

`--memory`/`--roofline` cannot run for Ascend targets yet ("no Facts
projection for MemoryHierarchyFacts"); the roofline below is anchored on
measured step times instead.

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

Arithmetic intensity at ctx=40959: 5.53e9 f32-flop / 8.18e9 B ≈ 0.68 flop/B —
deep in the memory-bound regime; the cube idles however it is priced.

## Measured (`kernel/test_mega.py L=28 TIME=...`, 20 reps after 3 warmups)

Current (after the MTE2 pipelining pass, see below), L=28 harness numbers:

|    ctx | ms/step |  tok/s | must-move GB | eff. GB/s |
|-------:|--------:|-------:|-------------:|----------:|
|     64 |    3.86 |  259.4 |         3.46 |       897 |
|   1024 |    3.99 |  250.6 |         3.58 |       897 |
|   4096 |    4.37 |  228.7 |         3.93 |       899 |
|   8192 |    4.92 |  203.3 |         4.40 |       894 |
|  13824 |    5.59 |  179.0 |         5.05 |       903 |
|  27648 |    7.61 |  131.4 |         6.65 |       874 |
|  40959 |    9.45 |  105.9 |         8.18 |       866 |

Before the MTE2 pipelining pass (same machine):

|    ctx | ms/step |  tok/s |
|-------:|--------:|-------:|
|     64 |    4.31 |  232.1 |
|   1024 |    4.43 |  226.0 |
|   4096 |    4.85 |  206.1 |
|  13824 |    6.19 |  161.6 |
|  27648 |    8.39 |  119.2 |
|  40959 |   10.42 |   95.9 |

Every context length now runs at ~866-903 GB/s effective (was 775-797).

## MTE2 pipelining pass (2026-10-20)

Profiling showed the kernel is memory-bound with the cube-side MTE2 pipe
only ~51% busy (the rest is copy<->gemm serialization bubbles, sync latency
and small-load stalls).  Changes, all in `kernel/mega.py`:

1. **Double-buffered weight tiles + cross-round prefetch** in the qkv, gu,
   down and lm_head GEMVs (o stays single-buffered: the 512 KiB L1 (cbuf)
   budget does not fit a fifth pair).  Two `l1_b` buffers alternate per kt
   parity; the kt+1 tile streams while the cube consumes kt, and the last
   kt of each nt round prefetches the next round's first tile.  A-tiles
   (activations) are double-buffered the same way.
2. **Prefetch the first B tile before the activation flag wait** in all
   double-buffered GEMVs (weights do not depend on the handshake; K/V cache
   tiles deliberately do NOT cross fb+9/fb+5 because the appended row is
   covered only by those flags).
3. **Attention K/V double-buffering**, enabled L1-neutrally by dropping SLT
   256 -> 128 (the 64 KiB K/V tiles halve to 32 KiB so the pair costs the
   same as the old single).
4. `nc_gu` 128 -> 64 and `nc_lm` 128 -> 64 to fit the double buffers in L1
   (gu: 8 rounds of 64 instead of 4 of 128, still exact; lm: pad stays ~0).

Measured effect: `aic_mte2_ratio` 0.51 -> 0.65 (short ctx) / -> 0.75 (long
ctx); aggregate read bandwidth 768 -> ~852 GB/s (sl=511) and -> ~1108 GB/s
(sl=13823); step time -10..-12% across the whole bench table (see above).
Gates re-passed: L=2 random, SEQ teacher-forced S=4096 (10 checkpoints).

Dead ends / traps learned:

- **K/V head-dedup** (share each K/V tile between the two q heads of a kv
  group) is a NET LOSS: it halves K/V bytes but forced SLT 128 -> 64 to fit
  L1, and the doubled attention-iteration count costs more than the saved
  traffic (measured slower at every ctx; reverted).
- **`T.gemm` operands must be whole L1 buffers, never slices** with a
  nonzero offset (`l1_q[1:2, :]` compiled fine but produced silently wrong
  scores -- caught by the random gate).
- **Cross-scope L1 buffer references are rejected** (an L1 buffer allocated
  in one `T.Scope("Cube")` cannot be read in a later one), so per-GEMV
  double buffers cannot be shared across scopes.
- cbuf is 512 KiB/block; the allocator only partially overlays scopes, so
  the total live footprint decides what fits (the o GEMV's double-buffer
  pair lost the budget fight).
- Hoisting the activation row to one load per GEMV (P0-2 as originally
  planned) is blocked by the sliced-gemm-operand trap; the per-kt A copies
  are instead hidden inside the double-buffer pipeline.

Optimization backlog, in expected-value order:

1. ~~Mid-context slice imbalance~~ — fixed (see above): runtime thirds split
   of the live prefix; all gates re-passed (SEQ S=40960 18 checkpoints, check
   gate, HF 64/64), table above.
2. ~~Launch-sync tax~~ — resolved differently: the launcher is now the
   ported upstream PR #178 (caching-allocator buffers, async non-taskqueue
   launches, no per-launch sync). Measured effect on greedy decode: none
   (+-0.05 ms; the argmax read forces a per-step sync regardless). The async
   property is banked for pipelined patterns (back-to-back launches now
   overlap host prep with device execution, 4.378 ms/launch at S=768). The
   old forced workspace env is dropped (kernel declares none; saves a
   pointless 786 KB/launch allocation).
3. ~~MTE2 pipelining~~ — done (this pass, ~-11% step time).
4. **Prefill batching** — prefill is one launch per prompt token (4.7 ms/tok
   at short ctx after the slice fix); a batched prefill variant would cut wall
   time for long prompts, though it is outside the decode kernel's contract.
5. **o-GEMV double buffering / deeper K-lookahead** — blocked by the 512 KiB
   cbuf budget; needs a fundamentally different tile split to free ~90 KiB.
6. **Weight repacking to contiguous K-major tiles** — deprioritized: with
   the pipelines in place the instantaneous per-block MTE2 rate already sits
   near the per-block fair share, so the strided 1 KiB rows are no longer
   the binding constraint (the remaining ~35% MTE2 idle is inter-stage sync
   latency and the single-buffered o GEMV).

## Baseline comparison (greedy, same checkpoint)

|    ctx | ours ms/tok | ours tok/s | HF transformers (torch_npu) tok/s | speedup |
|-------:|------------:|-----------:|----------------------------------:|--------:|
|     64 |        4.47 |      223.6 |                              70.0 |    3.2x |
|   1024 |        4.79 |      208.9 |                              68.0 |    3.1x |
|   8192 |        7.67 |      130.5 |                              42.4 |    3.1x |

Greedy token stream: 64/64 tokens identical to HuggingFace greedy on the
same prompt ("The capital of France is", `compare_hf.py`).

End-to-end: README prefill (3413 tok) + 2048 generated tokens, steady
5.06 ms/tok = 197.8 tok/s as context grows 3413 -> 5461 (`run.py
--prompt-file ... --max-new-tokens 2048`); before the slice fix this was
6.24 ms/tok = 160.2 tok/s.

## P0-4 combine-fold + P2 merged MLP-tail scope (2026-10-21)

Two follow-up passes on the bulk-synchronous overlap theme:

1. **P0-4: fold the attention combine into the vector stage.**  The leader
   AIV now computes `num = sum_j w_j * acc_j` (column-broadcast `vmul` +
   `T.reduce(dims=[0])`, bf16 landing discipline preserved) and `/den` in
   one vector pass.  The whole gemm3 cube scope, its `acc16/wrow16/num/den`
   GM staging buffers and two rendezvous flags are gone.
2. **P2 (phase 1): o + gate_up + down share ONE cube scope.**  A single
   explicit L1 ring (`l1_a0/l1_a1` + `l1_b0/l1_b1` at the max nc = 88) with
   per-GEMV `l0_c` accumulators; o is double-buffered for the first time,
   and each GEMV's two first weight tiles stream BEFORE its flag wait
   (fb+15/fb+11/fb+13), so the MTE2 pipe stays busy through the combine
   tail, rms2 and silu stages.  Two compiler traps surfaced (see NOTES
   traps 19/20): the shared accumulator had to be split per GEMV, and the
   ring needed a dynamic trip-1 wrapper loop to keep its allocation
   anchored inside the scope.

Measured (L=28, same protocol as the table above):

|    ctx | before (ms/tok) | P0-4 | P0-4+P2 | delta vs before |
|-------:|----------------:|-----:|--------:|----------------:|
|     63 |            3.88 | 3.78 |    3.66 |          -5.7 % |
|   1023 |            4.01 | 3.90 |    3.78 |          -5.7 % |
|   4095 |            4.39 | 4.28 |    4.16 |          -5.2 % |
|  13823 |            5.61 | 5.50 |    5.37 |          -4.3 % |

Gates: L=2 SL sweep PASS, L=28 SL sweep PASS, SEQ S=1024 synthetic PASS
(max rel 0.0230), SEQ S=1024 REAL teacher-forced PASS (max rel 0.0473 at
pos=1, identical to the pre-change baseline).

## P1-1 dual-AIV work split (2026-10-21, third opt session)

The two aiv subblocks ran every vector stage fully redundantly (both compute
and both GM-store identical values).  A spike (see NOTES trap 21) proved the
block-level `sync_block_wait` on the cube side passes only after BOTH aivs
set the flag -- a true rendezvous -- so splitting work by `subid` needs no
extra barrier as long as both aivs keep their per-stage `sync_block_set`.
Splits landed in `kernel/mega.py`:

* **S4 combine** (leader blocks): `for gh in range(2)` -> `gh = subid`.
  Profiling showed the leaders' aiv_vec time is ~2.7x the non-leaders'
  (162 us vs 61 us per launch at short ctx) because of this stage; halving
  it is the biggest single item.
* **S2** (qk-norm+rope): the two q heads split by `subid`
  (`qh_ = kv*2 + subid`); the shared k/v path (norm, rope, cache append)
  stays redundant -- its ~1.1 head-units cannot split below d=128.
* **S3 softmax**: `for gh in T.serial(2)` -> `gh = subid`; each aiv produces
  its own head's probs row + m/l partial.
* **S5'b silu**: split by `ii` halves (`i0 = subid * (ii//2)`), elementwise.
* **argmax**: `nch=4` chunks split 2+2 across the aivs; the old
  `if subid == 0` store guard is gone (each aiv writes disjoint slots).

NOT trapped: alloc extents must be compile-time expressions of baked params
(`ii // 2` inline); a loop-body `half_ii = ii // 2` assignment becomes a TIR
let-var and the dynamic UB extent SIGFPEs the allocator (NOTES trap 22).

Measured (L=28 test harness, 20 reps, A/B 3 rounds):

|    ctx | before (ms/tok) | P1-1 | delta |
|-------:|----------------:|-----:|------:|
|     63 |           3.643 | 3.573 | -1.9 % |
|  13823 |           5.370 | 5.283 | -1.7 % |
|  27647 |           7.380 | 7.303 | -1.1 % |

`run.py --bench ctx` (median of 5, same-session A/B):

|    ctx | base ms/step | P1-1 | delta |
|-------:|-------------:|-----:|------:|
|     64 |         3.67 | 3.56 | -3.0 % |
|   1024 |         3.79 | 3.66 | -3.4 % |
|   2048 |         3.92 | 3.80 | -3.1 % |
|   4096 |         4.16 | 4.04 | -2.9 % |
|   8192 |         4.68 | 4.57 | -2.4 % |
|  13824 |         5.38 | 5.26 | -2.2 % |
|  27648 |         7.36 | 7.25 | -1.5 % |
|  40959 |         9.16 | 9.06 | -1.1 % |

msprof (`--aic-metrics=Roofline`, 10 launches, same protocol as the 2026-10-20
baseline):

| run | aic_mte2_ratio | aiv_vec_ratio | aiv_mte2_ratio | aiv_scalar |
|-----|---------------:|--------------:|---------------:|-----------:|
| base @short    | 0.711 | 0.026 | 0.079 | 0.096 |
| P1-1 @short    | 0.721 | 0.019 | 0.063 | 0.076 |
| base @sl=13823 | 0.738 | 0.037 | 0.073 | 0.095 |
| P1-1 @sl=13823 | 0.748 | 0.023 | 0.060 | 0.077 |
| base @sl=27647 | 0.739 | 0.037 | 0.069 | 0.095 |
| P1-1 @sl=27647 | 0.750 | 0.023 | 0.058 | 0.076 |

aic_mte2_ratio +1pp everywhere (0.75 reached at long ctx); aiv_vec -28..-37%.

Gates: L=2 SL sweep PASS (the sl=128 near-tie argmax flip is the documented
synthetic noise, trap 14), L=28 SL sweep PASS (incl. 13695/6/7 slice
boundary), SEQ S=1024 REAL teacher-forced PASS (19 ckpts), SEQ S=4096 REAL
teacher-forced PASS (13 ckpts, max rel 0.0062).

### Dead ends measured this session (do not retry without new evidence)

* **P0-3 (K/V prefetch across fb+9/fb+5)**: depth-1 and depth-2 variants both
  measured 0.5-2.6 % SLOWER at every ctx.  The cube's MTE2 queue is FIFO:
  a pre-wait V0/K0 lands ahead of the post-wait critical loads (probs row /
  l1_q) and delays the first gemm by ~one tile transfer.  The fb+5 window
  vanishes at long ctx (gemm1 outlasts S3, so the flag is pre-set) and the
  window is shorter than a tile transfer at short ctx, so the prefetch only
  reorders the FIFO.  Contrast with pre-wait WEIGHT prefetch (P2 ring): that
  wins because the weights are not needed by the first post-wait gemm.
* **P0-2 lm_head depth-2 prologue**: +1 % slower.  The `(kt+nt)>0` guard it
  requires costs ~100 ns/iteration and lm_head has 396 iterations/launch;
  the ~1.6 us one-time saving does not cover it.  qkv's P0-2 was skipped for
  the same reason (guard cost + FIFO, and S1's ~2-3 us window is already
  covered by the existing 1-tile prologue).
* **P1-2 (narrowing PIPE_ALL)**: the barrier drains the aiv's pipes at the
  layer boundary, but the only outstanding pipes there are MTE3 (x store)
  and V -- MTE2 is already idle, so a narrower barrier drains the same set.
  ~zero gain, real race risk (UB overlay WAR across the loop backedge).
* **dynamic `jr` loop bound in gemm1/gemm2** (`T.serial(nloc)` instead of
  `T.ceildiv(step, slt)`): neutral -- the dead iterations cost nothing
  measurable.  Reverted (kept the static-bound form).
* S3 softmax arithmetic cannot be narrowed to the live extent: vbrc/vmul/
  reduce require static extents (dynamic-extent regions only work on T.copy).

## P2-narrowcast: dropping two provably-redundant plane barriers (2026-10-22)

The TODO's "P2 可选" asked for fine-grained flags to replace the fb+1 big
barrier by kv-group narrowcast.  Analysis of the eight per-layer plane
barriers showed (NOTES traps 24/25):

* A **literal** narrowcast is not buildable: FFTS INTER rendezvous are
  whole-plane (24 cubes / 48 aivs, spike 3k/3f), and GM seq-word polling
  (the AscendC twin's mechanism) is inexpressible -- every npuir codegen
  FATALs on `WhileNode`.
* But two rendezvous guard **non-global** data and can go:
  1. **fb+2 after the KV append (S2 end).**  gemm1's inputs are all
     same-block: the slice owner appends row sl itself (fb+9 covers), and
     every block of a kv group rewrites its own q_buf rows with identical
     values.  Removed; each cube now starts gemm1 as soon as its own aivs
     finish S2.
  2. **fb+1 after gemm2 (partials).**  The S4 leader reads only its own kv
     group's part_m/l/acc.  The cube now relays fb+6 right after its own
     gemm2 fixpipe (no 24-cube rendezvous); the existing fb+2 at S4 entry
     is the carrier (arrivals gated per block by fb+6, ordered on PIPE_MTE3
     after the aiv's own stores -- covers the sblk<=0 neutral-partial path
     too).  The freed cube flows straight into the o-scope Wo prefetch,
     overlapping slower blocks' gemm2 tail.

Per-layer plane barriers: 8 -> 6.  The surviving six are genuine all-block
exchanges (qkv_out sliding window, o_out/gu_out/d_out full activation rows,
attn_flat from all eight leaders).

Value-neutrality verified by GM dumps: part_acc / attn_out / o_out / logits
bitwise identical vs the pre-change kernel on the same scenario (the sl=255
and sl=13696 sweep flips are pre-existing synthetic near-tie noise, trap 26
-- the baseline flips identically).

Measured (L=28 harness, 20 reps, 3 A/B rounds):

|    ctx | before (ms/tok) | after | delta (avg) |
|-------:|----------------:|------:|------------:|
|     63 | 3.55-3.58 | 3.48-3.51 | **-1.7 %** |
|   1023 |     3.66-3.67 | 3.57-3.60 |     -2.2 % |
|   4095 |         4.05 | 3.95-3.98 |     -2.0 % |
|  13823 |         5.27 | 5.17-5.20 |     -1.7 % |
|  27647 |     7.27-7.29 | 7.17-7.22 |     -1.3 % |

msprof (`--aic-metrics=Roofline`, 10 launches):

| run | aic_mte2_ratio | aiv_vec_ratio | aic_time |
|-----|---------------:|--------------:|---------:|
| before @short    | 0.7225 | 0.0188 | 3445 us |
| after  @short    | 0.7293 | 0.0189 | 3419 us |
| before @sl=27647 | 0.7367 | 0.0251 | 3706 us |
| after  @sl=27647 | 0.7491 | 0.0255 | 3657 us |

Gates: L=2 SL sweep PASS (0/1/33/255/1023), L=28 SL sweep PASS (the two
near-tie flips reproduce identically on the pre-change kernel), SEQ S=1024
REAL teacher-forced PASS (8 ckpts, max rel 0.0092), SEQ S=4096 REAL
teacher-forced PASS (10 ckpts, max rel 0.0061).
