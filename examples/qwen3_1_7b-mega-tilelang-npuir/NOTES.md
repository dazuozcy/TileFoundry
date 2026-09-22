# Qwen3-1.7B mega-kernel on TileFoundry (npuir Expert mode)

Project: whole greedy decode step (embed gather, 28 layers, final norm, tied
lm_head, argmax partials) in ONE kernel launch per token, on one Ascend 910B.

Layout constants (mega_shapes in kernel/mega.py): NB=24 blocks (1 aic + 2 aiv),
NSL=3 seq slices, STEP=13824 (at S=40960), SPADG=41472, SLT=256, KTILE=256,
NQKV=4096, NQKV_PAD=4224 (24x176 contiguous slabs, 2x88 cube tiles), o/down
NC=88 (pad 2112), gu NC=128x4 (12288 exact), lm SPAN=6331 (VPAD=151944),
SPANP=8192, NCH=4, CH=2048, kpad=16. cid = sid*8 + kv (kv = cid%8,
sid = cid//8). Weights: /home/tilelang/zuochuanuong/weights/qwen3_1_7b.

## Status (2026-09-22)

Kernel CORRECT end-to-end: teacher-forced 40960-step run over real README text,
15 checkpoints across the whole context range (0,1,2,63,1023,4095,8464,
13823/4, 27647/8, 32767, 40957/8/9) all MATCH the fp32 reference with logits
rel 0.005-0.047. Timing (post-fix, per-launch sync): 4.45ms/224.5 tok/s @sl=64,
5.28ms/189.5 @2048, 7.75ms/129.1 @8192, 9.90ms/101.0 @13824, 10.15ms/98.5
@27648, 10.56ms/94.7 @40959.

## Load-bearing infrastructure fix (tilelang launcher memory leak)

The npuir launcher (tilelang/jit/jit_npu.py, generate_npu_wrapper_src) leaked
ONE rtMalloc'ed syncBlockLock (lock_num=131072 -> exactly 1 MiB for this
kernel) on EVERY launch in the non-taskqueue path; the taskqueue path leaked
the lock too (only workspace was freed). After ~8464 launches the device
allocator exhausts and rtMalloc fails SILENTLY -> the launch becomes a no-op
(frozen outputs); the last launch before exhaustion can run with a broken lock
init (observed once as: all 28 layers appending IDENTICAL k rows, i.e. the
layer loop replaying li=0). Fix applied in the dev tree
(tilelang-mlir-dev/tilelang/jit/jit_npu.py): after rtKernelLaunch the launcher
now does rtStreamSynchronize + rtFree(syncBlockLock) (+ rtFree(workspace) on
the inline path) in BOTH taskqueue modes. Backup: /tmp/opencode/jit_npu.py.bak.
After the fix the leak is 0 B/launch and the 8464 corruption disappears.
Kernel caches (.tlcache) had to be cleared so launchers recompile.

Cost: every launch now synchronizes (no pipelining across launches) ~0.3-1.4ms
slower per step. For greedy decode this is the realistic mode anyway (the host
needs the argmax each step).

## Traps learned today (add to the trap list)

1. LAUNCHER LEAK (above): any long-running decode (>8k launches) on the stock
   dev tree breaks mysteriously. Symptom: outputs freeze; occasionally one
   corrupt launch near the boundary. Diagnosis: torch.npu.mem_get_info() delta
   per launch.
2. Device module registration cap: only ~4-5 rtDevBinaryRegister'ed kernels
   per process work; further JitKernel_NPU objects get dead function handles
   and their launches silently no-op. NEVER compile many kernel variants in
   one process; bisect depth by other means.
3. Random-KV-history tests are NOT a correctness signal at full context: the
   model amplifies 1-ULP input noise chaotically on such off-manifold states
   (measured: 1 bf16 ULP on the embedding -> rel 0.058 logits divergence after
   8 layers). Validation must be teacher-forced (on-manifold history) or vs
   transformers.
4. Qwen3 residual stream has legit |x| up to ~600-2500 (outlier channels);
   do not mistake large magnitudes for corruption. Judge only vs reference.
5. Buffers: stage every GM buffer through the host (torch.zeros(...).to(npu));
   direct device allocations invite stale reads (known hazard).
6. The kernel jit decorator now carries
   pass_configs={"npuir.enable_auto_multi_buffer": False} (kept; the pass is
   implicated in stale-slot reads of loop-carried accumulators generally).

## Validation harness (kernel/test_mega.py)

- SL mode (default): per-sl single calls with random history; OK for small sl,
  meaningless at full context (trap 3).
- SEQ=1 mode: teacher-forced sequential decode over the checkpoint README
  text; checkpoints compare kernel logits vs fp32 ref_decode conditioned on
  the kernel's own cache. THIS is the correctness gate. CKPT env selects
  checkpoint positions; DBG=1 dumps per-layer k-row errors on failure.
- TIME=sl,sl,... mode: per-step wall time at given positions.
- Env for all runs: source CANN set_env, then
  PYTHONPATH=/home/tilelang/zuochuanuong/tilelang-mlir-dev/
  TILELANG_NPU_COMPILER_PATH=.../3rdparty/AscendNPU-IR/build/bin
  TILELANG_ASCEND_MODE=expert TILELANG_ENABLE_TASKQUEUE=false
  TILELANG_ASCEND_WORKSPACE_SIZE=32768
  LD_LIBRARY_PATH=/home/tilelang/miniconda3/envs/tilefoundry/lib:
  $ASCEND_HOME_PATH/lib64:.../torch_npu/lib TORCH_DEVICE_BACKEND_AUTOLOAD=0
  and /home/tilelang/miniconda3/envs/tilefoundry/bin/python (3.12, the only
  interpreter that loads both npuir and torch_npu).

## Deliverables complete (session of 2026-09-22)

All four TODO items landed; the gates all pass.

### Files

- `gen_model.py` -> `model.py` (~3.1k lines): the TileFoundry Module HIR of
  the whole decode step. 28 unrolled layers; 24-way N splits for
  qkv/o/gu/down/head; 8-way KV-head mesh for attention. `sys.setrecursionlimit`
  shim at import (the checker's AST walk recurses per unrolled layer).
- `runtime_model.py`: the RuntimeModule twin. `mega_step` drives the actual
  kernel; `set_weights` is the only place HIR layouts -> kernel GEMV slabs
  (transposes + pad-strip). MEGA_S env picks the padded context (768 for the
  check, 40960 for decode). ONE kernel variant per process (module cap).
- `prepare_weights.py` -> `prepared/model.safetensors`: checkpoint re-laid-out
  into the HIR's declared weight names/shapes for `--weights ckpt:prepared`.
- `dump_acts.py` -> `acts/*.pt`: coherent teacher-forced activations (ctx=256,
  kernel-built cache) for `--inputs files:...`.
- `probe_twin.py`: HIR-vs-twin per-layer k/v diff (the debugging tool that
  isolated both the rope sign bug and the bf16-landing bugs).
- `run.py`: greedy decode CLI (--prompt/--prompt-file/--max-new-tokens/--seed/
  --greedy, --bench ctx|steps). `compare_hf.py`: vs HuggingFace greedy.
- `OPTLOG.md`: analyze numbers + traffic model + measured roofline table.

### Gate results

- `tilefoundry check runtime_model.py:Qwen3MegaRT.mega_step --inputs files:...
  --weights ckpt:prepared --dim ctx_len=256 ...` -> PASS
  (logits rel_l2 0.0062, argmax equal, k-rows rel_l2 0.0044, v-rows 0.0106).
- Greedy vs transformers (torch_npu): 64/64 tokens identical.
- 1 launch/step: the generated launcher has exactly one rtKernelLaunch site
  (non-taskqueue path); run.py calls it once per token.
- tok/s: 223.6 @ctx=64 ... 94.5 @ctx=40959; HF baseline 70.0/68.0/42.4
  @64/1024/8192 -> 3.1-3.2x. 2048-token run: steady 160.2 tok/s
  (ctx 3413->5461).

### New traps (this session)

5. tf.rope applies rotate_half itself (negation inside); the kernel's tables
   carry the sign (sin row = cat(-s, s)). When staging the published
   convention into the kernel, NEGATE the first half of each sin row; cos
   passes through. Symptom if wrong: layer-0 k-row err ~10, v-row exact.
6. The HIR must dot in f32 (cast operands before tf.matmul): a bf16 matmul
   lands its OUTPUT bf16, which the kernel never does (scores, qkv, o, gu,
   down, head all accumulate f32). Symptom: ~4% systematic divergence growing
   per layer. Layer-0 k/v rows bit-exact is the success signal.
7. `for t in tile(C, BLK)` slices are STATIC 256-windows: they fail at
   ctx < 256 and on the tail tile (Slice window exceeds axis). Whole-context
   ops with symbolic C work everywhere (arange accepts DimVar lengths).
8. HIR mesh topology level names must be target-supported: AscendTarget
   accepts npu/cta/thread; "aic" parses and checks but `analyze` rejects it.
   Use Topology("cta", 24).
9. `analyze --memory/--roofline` is unavailable for Ascend targets (no
   MemoryHierarchyFacts projection); --compute-cost works.
10. Residency: every multi-input op needs its concrete operands in one
    residency; reshard into "smem" before mixing gmem params into mesh-local
    math (scale, live mask, transposed caches all needed this).
11. `--expected` on `check` is a saved-OUTPUT file, not a reference model;
    the reference is the HIR interpreter itself, the candidate is the twin.
12. The chaos floor: HIR-interpreter vs kernel differ by hardware-vector-op
    rounding (vexp/vsqrt vs torch) which amplifies ~1.5x/layer through the
    residual stream. At ctx=256 real weights: k/v rows max-element err
    ~0.12/0.45, rel_l2 ~0.004-0.011. Bounds at ~2.5x that floor still catch
    real bugs (the rope bug was err ~10 / rel_l2 ~0.5+). Documented in the
    check command in runtime_model.py's docstring.

### Standing env (unchanged)

source CANN set_env; PYTHONPATH=/home/tilelang/zuochuanuong/tilelang-mlir-dev/
TILELANG_NPU_COMPILER_PATH=.../3rdparty/AscendNPU-IR/build/bin
TILELANG_ASCEND_MODE=expert TILELANG_ENABLE_TASKQUEUE=false
TILELANG_ASCEND_WORKSPACE_SIZE=32768
LD_LIBRARY_PATH=/home/tilelang/miniconda3/envs/tilefoundry/lib:
$ASCEND_HOME_PATH/lib64:.../torch_npu/lib TORCH_DEVICE_BACKEND_AUTOLOAD=0;
interpreter /home/tilelang/miniconda3/envs/tilefoundry/bin/python.

### Optional follow-ups

- Mid-context slice imbalance (see OPTLOG backlog: empty-slice early-out).
- Launcher lock-pool fix to remove the per-launch sync tax.
- Batched prefill variant.

## Launcher: PR #178 ported (2026-09-22, second session)

Replaced my local leak fix with the port of upstream PR #178
(tile-ai/tilelang-mlir-ascend#178, 3 patches) onto this tree:

- lock buffer: per-launch `rtMalloc`/`rtFree` ->
  `at_npu::native::allocate_workspace` (torch_npu workspace pool, stream-aware)
- workspace: `rtMalloc` -> `at::empty(kPrivateUse1)` (caching allocator);
  inference default 32768 -> 0, override moved before wrapper generation
- stream resolved at call time (`_get_current_raw_stream`, mirroring
  triton-ascend) instead of captured at kernel build
- non-taskqueue path no longer force-syncs per launch (fully async)
- npu_utils.so build race fix (per-invocation tmpdir output)

Validation (canonical gates, NOT hand-rolled smokes -- see trap 13):
- SEQ S=40960 REAL=1, 15 checkpoints incl. 8464 and slice boundaries: PASS
  (both with forced ws=32768 and with ws=0)
- SEQ S=10240 with ws=0 (PR inference): PASS -> the mega kernel needs NO
  workspace (its .so declares no `_infer_workspace_shape_function`); the old
  standing env `TILELANG_ASCEND_WORKSPACE_SIZE=32768` is DROPPED (it made the
  launcher allocate a pointless 786KB/launch)
- `tilefoundry check` gate: PASS (identical numbers; the launcher does not
  change numerics)
- run.py greedy: coherent, 221.6 tok/s at short ctx
- leak: ~0 B/launch (was 1,048,576); back-to-back async launches at S=768:
  4.378 ms/launch (host prep fully overlapped)

Performance verdict (honest): per-token wall time is UNCHANGED (+-0.05ms
noise) across the whole 0..40960 bench table -- greedy decode must read the
argmax after every step, so the per-step sync happens anyway and the removed
double-sync/rtMalloc churn is in the noise. My earlier "0.3-1.4ms sync tax"
estimate was wrong (it compared a differently-measured loop). The real wins
are hygiene and capability: zero driver-level alloc churn, no fragmentation,
TRUE async launches (back-to-back host prep overlaps device execution --
relevant for any future pipelined/batched-prefill use), and the workspace
waste removal. The mid-context slice imbalance remains the one big
performance item (OPTLOG backlog #1).

13. Debugging trap of the session: a hand-rolled smoke harness (reference
    weights reconstructed by permutes + checkpoint logits read AFTER the
    loop) produced a deterministic rel~1.4 "failure" that sent the bisect
    (lock source, workspace source, sync, stream) down a garden path -- every
    variant "failed" identically because the REFERENCE was wrong, not the
    launcher. The canonical seq_test gate arbitrated: everything passed. Rule:
    when a bisect produces IDENTICAL failure signatures across supposedly
    causal variants, suspect the measurement first.

Launcher backups: /tmp/opencode/jit_npu.py.bak (original, leaky),
/tmp/opencode/jit_npu.py.myfix (my sync+free fix), jit_npu.py.purePR (the
ported PR, now live). Standing env: same as before MINUS
TILELANG_ASCEND_WORKSPACE_SIZE.

## Dynamic attention slices (2026-09-22, third session)

The mid-context dip (509-654 GB/s at ctx 8k-27k) was load spreading, not
barrier overhead: a single block sustains ~32-36 GB/s of cache streaming, and
with fixed sid*step slice windows only the slices under the live length
stream -- 8 active blocks at ctx<STEP, 16 at ctx<2*STEP. Fix (kernel/mega.py):

    thirds = ceil(ceil(sl+1, nsl), slt) * slt      # runtime, SLT-aligned
    s0 = sid * thirds; e0 = min(sl+1, s0+thirds)

plus the new-token owner condition `sl < s0 + thirds` (was s0+step). The LSE
combine is partition-independent, so nothing else changes; at full context
thirds == step (the old windows). SLT alignment keeps s0 tile-aligned, which
also keeps the tail tile inside the padded cache (unrounded thirds can read
past SPADG at small S).

Results: +58% tok/s @13824 (100 -> 158), +39% @8192, +21% @4096; every ctx
now at 775-797 GB/s. All gates re-passed: SEQ S=40960 (18 ckpts incl.
13823/4/5), SEQ S=10240 (12 ckpts), tilefoundry check, HF greedy 64/64,
2048-token run 197.8 tok/s steady.

14. Synthetic-mode noise: after the re-partition, one random-history position
    crossed the harness's rel 0.05 line (0.0554 @sl=1023) with argmax still
    matching and normal k_row_err -- a different LSE-partition rounding path,
    not a bug. Real-weights teacher-forced SEQ (rel 0.004-0.010 at the same
    region) is the arbiter; synthetic FAIL + argmax MATCH = suspicious noise,
    verify on-manifold before bisecting.
