# Qwen3-1.7B mega-kernel on TileFoundry (npuir Expert mode)

Project: whole greedy decode step (embed gather, 28 layers, final norm, tied
lm_head, argmax partials) in ONE kernel launch per token, on one Ascend 910B.

Layout constants (mega_shapes in kernel/mega.py): NB=24 blocks (1 aic + 2 aiv),
NSL=3 seq slices, STEP=13696 (at S=40960), SPADG=41088, SLT=128, KTILE=512,
NQKV=4096, NQKV_PAD=4224 (24x176 contiguous slabs, 2x88 cube tiles), o/down
NC=88 (pad 2112), gu NC=64x8 (12288 exact), lm NC=48 (SPAN=6331, VPAD=151944),
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

## MTE2 pipelining (2026-10-20 session)

Profiled `main_mix_aic` (msprof op): the kernel is memory-bound with the
cube-side MTE2 pipe only ~51% busy -- the rest is copy<->gemm serialization
bubbles (single L1 tile buffers), sync latency, and small activation loads.
Fix landed in kernel/mega.py:

- qkv/gu/down/lm_head GEMVs: two L1 B-buffers (and two A-buffers) alternate
  per kt parity; tile kt+1 streams while the cube consumes kt, with a
  cross-round prefetch of the next nt round's first tile at kt == nkt-1.
  The first B tile is prefetched BEFORE the activation flag wait (weights do
  not depend on the handshake).  The buffer parity requires nkt even
  (h//ktile = 4, ii//ktile = 12 -- holds).
- attention K/V double-buffering, enabled L1-neutrally by SLT 256 -> 128
  (the K/V tile pair then costs the same 64 KiB as the old single).  K/V
  prologue loads stay behind fb+9/fb+5 (the appended row is only covered by
  those flags -- do NOT prefetch K/V across the wait).
- nc_gu 128 -> 64 (gu 8x64, still exact) and nc_lm 128 -> 64 to fit the
  double buffers in the 512 KiB cbuf; the o GEMV stays single-buffered (its
  pair lost the budget fight).

Results (L=28 harness, ms/step): 4.31 -> 3.85 @sl0, 4.43 -> 3.99 @1023,
4.85 -> 4.37 @4095, 6.19 -> 5.59 @13823, 8.39 -> 7.61 @27647,
10.42 -> 9.45 @40959  (~-11% everywhere).  msprof: aic_mte2_ratio
0.51 -> 0.65 (sl=511) / -> 0.75 (sl=13823); aggregate read 768 -> ~852 GB/s
(short) / ~1108 GB/s (long).  Gates re-passed: L=2 random (S=768, 7 sl),
SEQ teacher-forced S=4096 REAL=1 (10 checkpoints).

### New traps (this session)

15. **`T.gemm` operands must be whole L1 buffers, never slices.**  A sliced
    operand with a nonzero offset (`l1_q[1:2, :]`, a BufferRegion) compiles
    cleanly but is silently miscompiled (the offset appears to be dropped):
    the K-head-dedup attempt computed wrong scores until the two q rows were
    moved into separate full buffers.  This also blocks hoisting the GEMV
    activation row to one load + per-kt slices.
16. **Cross-scope L1 references are rejected.**  An L1 buffer allocated in
    one `T.Scope("Cube")` cannot be referenced in a later one (compile-time
    diagnostic), so double-buffer pairs cannot be shared across GEMV scopes.
17. **cbuf (L1) is 512 KiB and the allocator only partially overlays**
    disjoint scopes: the deciding metric is the total live footprint, not
    per-scope max.  Doubling all five GEMVs needs ~736 KiB; the o pair
    (smallest weight) was the one to drop.
18. **K/V head-dedup is a net loss here.**  Sharing each K/V tile between
    the two q heads of a kv group halves K/V bytes but forced SLT 128 -> 64
    to fit L1; the doubled attention-iteration count (and smaller tiles)
    costs more than the saved traffic -- slower at every ctx.  Reverted.

### New traps (P2 merge session, 2026-10-21)

19. **`PlanAndUpdateBufferAllocationLocation` is stateful and can DROP
    allocations.**  The pass latches an internal `need_extract` flag the
    first time a gemm accumulator (`initC=(kt==0)`) gets its LCA hoisted to
    the layer-loop level; from then on every later buffer placement goes
    down a fragile loop-depth arithmetic path, and lm_head's `l1_a1/l1_b1`
    Allocate nodes silently vanished, so codegen died with
    `cannot find variable l1_a1` inside the lm_head kt loop (the var name
    is the shared hint -- the failing scope was NOT the merged one).
    Kernel-side fix: one accumulator per GEMV (`l0_co/l0_cg/l0_cd`) so no
    accumulator's uses span the whole merged scope.  Disabling the pass
    (`tl.enable_plan_and_update_buffer_allocation: False`) is NOT an
    option: without its sinking the cbuf/UB allocators overflow.
20. **Buffers whose LCA is a whole scope get hoisted to the enclosing
    loop, and the MLIR allocator anchors liveness at the alloc site.**
    The shared o/gu/down ring landed at the `for li` top, pinning its
    ~176 KiB across the entire layer (concurrent with qkv + attention) ->
    cbuf 528 KiB > 512 KiB.  Fix: wrap the three GEMVs in a dynamic
    trip-1 loop (`for _r in T.serial(T.min(sl + 1, 1))`) so the ring's
    LCA becomes an inner ForNode; Simplify cannot fold the bound and the
    allocator again sees per-phase lifetimes.

### New traps (P1-1 session, 2026-10-21)

21. **`sync_block_wait` on the cube side is a true block rendezvous: it
    passes only after BOTH of the block's aiv subblocks call
    `sync_block_set`** (spike: a variant where only aiv0 sets the flag hangs
    forever; both-set completes).  Cube→aiv flags broadcast (one cube set
    satisfies both aivs' waits).  This is what makes P1-1's subid work
    splits safe with no extra barrier -- each aiv sets the shared flag after
    its own half, and the cube proceeds only when both halves are done.
    (The AscendC twin deliberately avoided FFTS for multi-setter gates --
    "a wait passes after ONE set" for its mode-2 cross-core flags -- but
    that is the INTER-core direction; the same-block aiv→cube INTRA flags
    behave as a 2-party rendezvous here.)
22. **A loop-body assignment (`x = <int expr>`) becomes a TIR let-var, and
    using it as a `T.alloc_ub` extent creates a DYNAMIC-extent UB buffer
    that SIGFPEs the allocator** (host-side crash, empty log; use gdb-less
    bisection).  Fix: keep alloc extents as inline expressions of baked
    params (`ii // 2`, not a `half_ii` variable).  Also: `T.vbrc`/`T.vmul`/
    `T.reduce` require STATIC extents -- only `T.copy` tolerates dynamic
    region extents (so the S3 softmax arithmetic cannot be narrowed to the
    live slice length).
23. **Pre-wait cache-row prefetch (P0-3) is a FIFO reorder, not a win.**
    The cube MTE2 queue is in-order: a V0/K0 tile issued before a flag wait
    lands ahead of the post-wait critical loads (probs row, l1_q) and delays
    the first gemm by ~its own transfer time whenever the wait window is
    shorter than the transfer (short ctx) -- and at long ctx the flag is
    already set when the cube arrives (gemm1 outlasts S3), so there is no
    window to fill.  Measured: 0.5-2.6 % slower at every ctx for both
    depth-1 and depth-2 variants.  Pre-wait WEIGHT prefetch (P2 ring) wins
    by contrast because weights are not consumed by the first post-wait
    gemm.  The same FIFO+guard accounting killed the lm_head depth-2
    prologue (P0-2): its `(kt+nt)>0` guard costs ~100 ns x 396 iterations,
    more than the ~1.6 us saving.

### New traps (P2-narrowcast session, 2026-10-22)

24. **The plane-barrier lattice has provably-redundant members; audit
    per rendezvous which data is truly cross-block.**  Two of the eight
    per-layer plane barriers guarded nothing global: (a) fb+2 after the
    KV append -- the slice owner appends row sl itself (fb+9 covers it),
    and q_buf rows are rewritten identically by every block of the kv
    group, so gemm1 has only same-block inputs; (b) fb+1 after gemm2 --
    the S4 leader reads only its own kv group's part_m/l/acc, and the
    already-existing fb+2 at S4 entry is a sufficient carrier: every
    aiv arrives there behind its own cube's fb+6 (i.e. after its own
    gemm2 or its own S3-Else stores), and arrival on PIPE_MTE3 orders
    it after the aiv's own stores.  Removing the two barriers is
    bitwise value-neutral (verified by GM dumps) and passed all gates.
    Lap-safety is preserved by the surviving four cube-plane barriers
    per layer (a block cannot lap another around a shared flag id).
    The truly-global barriers that must stay: qkv (S2 reads a sliding
    producer window), o/gu/down (every aiv reads the full activation
    row), attn_flat (every cube reads all eight leaders).
25. **A literal kv-group narrowcast cannot be built with the current
    primitives.**  INTER FFTS rendezvous are whole-plane (all 24 cubes
    or all 48 aivs -- spike 3k/3f), no subset mode exists, and the
    AscendC-twin style GM seq-word polling is inexpressible: every
    npuir codegen (`codegen_npuir{,_api,_dev,_api_a5,_dev_a5}.cc`)
    FATALs on `WhileNode`, so data-dependent spin loops don't compile.
    If a subset rendezvous is ever needed, the backend first needs a
    lowering for bounded spin + scalar GM polling.
26. **The synthetic gate has pre-existing run-to-run argmax flakiness
    at near-ties** (trap 14 generalizes): at sl=255 with a 0.0026 top-2
    gap the SAME baseline binary flipped kernel tokens across runs
    (15881 vs 132059) with matching rel (~0.035-0.040).  Before
    suspecting a new sync bug, re-run the baseline -- and verify
    value-neutrality directly by dumping GM intermediates (part_acc,
    attn_out, logits) for both versions and diffing bitwise.

### P1-1 dual-AIV split landed (2026-10-21)

Splits: S4 combine `gh = subid` (the biggest item -- leaders' aiv_vec was
2.7x non-leaders'), S2 q-heads by subid (k/v redundant), S3 softmax
`gh = subid`, S5'b silu by ii halves, argmax chunks 2+2.  Measured
(bench ctx): -1.1 .. -3.4 % across 64..40959; aic_mte2_ratio +1pp
(0.711->0.721 short, 0.739->0.750 @27647); aiv_vec -28..-37 %.  Gates:
L=2/L=28 SL sweeps, SEQ S=1024 + S=4096 REAL teacher-forced all PASS.

Remaining backlog (2026-10-21 update)

- o-GEMV double buffering: DONE via the P2 merged MLP-tail scope (the
  explicit ring made it footprint-neutral; see OPTLOG 2026-10-21).
- P0-3 (K/V prefetch across flags): DEAD END, measured (trap 23).
- P0-2 depth-2 prologues (qkv/lm_head): DEAD END for lm_head (measured
  +1 %); qkv skipped by the same FIFO+guard accounting (trap 23).
- P1-1 (dual-AIV split): DONE (this session).
- P1-2 (narrowing PIPE_ALL): analyzed as ~zero gain (MTE2 is already idle
  at that boundary), race risk -- not pursued.
- P2-optional (fine-grained flags): DONE (2026-10-22) as barrier economy
  -- fb+2#1 and fb+1#2 removed (traps 24/25); literal narrowcast blocked
  by backend (no While, plane-wide FFTS).
- P2 phase 2: pull qkv/attention into the same merged cube scope so their
  weight/K/V prefetches also stream across stage boundaries.
- Weight repacking to contiguous K-major tiles: deprioritized (the instant
  per-block MTE2 rate is already near fair share; remaining idle is
  inter-stage sync latency).
- Prefill batching (outside the decode kernel's contract).
