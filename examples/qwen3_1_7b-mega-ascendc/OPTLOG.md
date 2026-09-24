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
