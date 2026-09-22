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

## Measured (`run.py --bench ctx`, median of 5, sync included)

|    ctx | ms/step |  tok/s | must-move GB | eff. GB/s |
|-------:|--------:|-------:|-------------:|----------:|
|     64 |    4.46 |  224.3 |         3.46 |       776 |
|   1024 |    4.58 |  218.3 |         3.58 |       781 |
|   2048 |    4.69 |  213.0 |         3.69 |       787 |
|   4096 |    5.00 |  200.2 |         3.93 |       786 |
|   8192 |    5.57 |  179.7 |         4.40 |       790 |
|  13824 |    6.34 |  157.8 |         5.05 |       797 |
|  27648 |    8.54 |  117.0 |         6.65 |       778 |
|  40959 |   10.56 |   94.7 |         8.18 |       775 |

Every context length now runs at the same ~775-797 GB/s effective rate: the
kernel sits at the achievable HBM roofline across the whole range.

Before the dynamic-slice fix the mid-range ran at 509-654 GB/s (10.00 ms at
ctx=13824, 7.72 ms at 8192). The cause was not barrier overhead but
load spreading: with fixed sid*step windows only the slices below the live
length stream cache rows, and a single block sustains only ~32-36 GB/s, so 8
active blocks reached just ~510 GB/s aggregate (16 blocks: ~655; 24: 775).
Splitting the live prefix into runtime thirds (SLT-aligned) keeps all 24
blocks streaming at every context length; the split degenerates to the old
fixed windows at full context. Measured gains: +58% tok/s at ctx=13824, +39%
at 8192, +21% at 4096; ends unchanged (they were already at roofline).

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
3. **Prefill batching** — prefill is one launch per prompt token (4.7 ms/tok
   at short ctx after the slice fix); a batched prefill variant would cut wall
   time for long prompts, though it is outside the decode kernel's contract.

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
