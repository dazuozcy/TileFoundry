# Qwen3-1.7B decode as ONE AscendC mix-kernel launch per token

Raw-AscendC re-implementation of the npuir mega-kernel: the entire decoder
step -- 28 layers, full-context attention, LM head -- executes inside a
single `KERNEL_TYPE_MIX_AIC_1_2` kernel launch (24 cube + 48 vector blocks)
on one Atlas 910B. The device binary is compiled by bisheng
(`kernel/build.sh`) and launched through the bisheng `<<<>>>` host stub
(`run_megak`) via ctypes; on top of it sits the standard TileFoundry pair:
`model.py` (the mega-step HIR, shared verbatim with the npuir example) and
`runtime_model.py` (its runtime twin, one `run_megak` launch per step),
gated by `tilefoundry check` -- see OPTLOG.md for the gate, the
`tilefoundry analyze` numbers, and the roofline table.

## The original task prompt (verbatim)

The task as it was given to the session that built this kernel (Chinese,
unabridged):

```
在 TileFoundry 上跑通 Qwen3-1.7B 的真实解码，并把它做快。
权重与配置：/home/tilelang/zuochuanuong/weights/qwen3_1_7b
硬件：一张昇腾 Atlas 910B（CANN 8.5 环境已 source；torch 与 torch_npu 可用）。
Kernel 后端：AscendC。
手写 __global__ __aicore__ 的 C++ kernel，bisheng 编译（-xasc，910B 就是 dav-2201），<<<grid, nullptr, stream>>> 直发 launch。
AscendC 的 API 真相在已 source 的 CANN 里：$ASCEND_HOME_PATH/x86_64-linux/ascendc/include/，
从 kernel_operator.h 读起——basic_api/interface/ 是搬运、cube 与同步的底层接口
（DataCopy、LoadData、Mmad、Fixpipe、SyncAll、CrossCoreSetFlag / CrossCoreWaitFlag……），
highlevel_api/lib/ 是向量内建和算子族（broadcast、matmul、reduce、softmax、silu、topk……）。
逐个头文件读，不猜 API。

关于 TileFoundry 的一切，问 tilefoundry 命令——不要问人，不要去别处找。
关于 AscendC 的一切，只读 CANN 的头文件和相关例子。模型本身归你自己研究。

环境已准备好，tilefoundry / torch_npu / bisheng（CANN 8.5）均已装好。
动手前先跑通这条工具链的冒烟测试。
不要安装、升级或替换任何包。

只做文本解码路径——batch=1、seq_len=1 那条分支。

### 一个形状：mega decode

host:   一步一次 launch。一个 kernel、一个常驻 grid，走完整个 decode step。
        不用 torch_npu graph 把一串 launch 录起来回放——是真的一个 kernel。

device: 同一个 kernel 里走完一步：
            for layer in 0 .. L-1:
                for stage in 这一层要过的 stage:
                    do(stage)
                    barrier()        # 一个 stage 结束在一次同步上，不在一次 return 上
            lm_head

prompt 的 token 也一个一个走这同一条路。权重一份、状态一份——不拷贝、不重排。

### 三个约束

一——HIR 本身就得是 mega 的，而且与运行时实现同形。整个 decode step 是一个程序；
   一个 stage 的边界是一次 mesh 级的 barrier，不是一次 @func 边界。

二——kernel 走 AscendC 的低层表面。显式的搬运与 buffer（DataCopy / DataCopyPad、
   TPipe.InitBuffer、TQue / TBuf——铺在哪级存储说出来：VECTIN / VECTOUT 是 UB，
   A1 / B1 / C1 是 L1，A2 / B2 / CO1 是 L0）、显式的向量内建（BroadCast / Mul / Exp /
   ReduceSum / ReduceMax 这一族）、cube（Matmul，或更裸的 LoadData + Mmad + Fixpipe）、
   显式的同步（SetFlag / WaitFlag、CrossCoreSetFlag / CrossCoreWaitFlag、SyncAll）——
   搬运、时序、buffer 的安排是你写出来的，不是编译器或运行时替你排的。一次搬运
   在哪里发出、谁等谁、数据什么时候落地，都说出声来：跨 stage 的预取只有这么说
   才说得出来。

三——“快”要在整个范围上成立，不是在某一点上。交一张从 0 到 40960（这个模型的
   max_position_embeddings）的解码 tok/s 表，每个长度一个数，每个都要站得住——
   短上下文好看、长上下文垮掉的实现不算完成。

   怎么让整张表站得住，归你定：某个 stage 是不是要拆几个变体、运行时按什么选、
   边界放哪里——量了再定。但把决定的依据写下来：试过哪些、各自在哪些长度上量到
   多少、为什么留下的是留下的那个。

### 三步

这是最省功夫的顺序：

    1. 先把整个 decode step 写成 authored HIR；从 analyze --performance 拿到一个数
    2. 在 HIR 里探分布与数据流——placement（哪层 mesh 负责哪个轴、常驻 grid 怎么铺）、
       stage 怎么切、哪些能并行、每个值落在哪级存储、搬运发生在哪里——直到那个数不再降
    3. 然后才碰 kernel（逐算子的小 kernel 先行验工具链与数值不算越步，
       但 HIR 的数字推到底之前不开始调优它）

第 2 步每次改动留一行：改了什么，数字从多少到多少。进第 3 步之前，把最后一个数
连同 roofline 下限一起报告。analyze 会给搬运的字节计价——在一个不吃带宽的 stage 上
跟着它跑是白费功。两个数对不上就直说，不要硬凑。

### 交付物
    model.py          TileFoundry Module：那个 mega decode step 的 HIR
    runtime_model.py  它的孪生；kernel 从这里被调用
    kernel/           AscendC 源码与构建/launch 胶水——一条命令从零重建
    run.py            入口：python run.py --prompt "..." --max-new-tokens 2048
                      打印续写文本和一个 tok/s 数；也支持 --seed 和 --greedy

### 完成标准
按顺序——前一关不过，后面行的数字都不算。
| | 标准 | 什么算过 |
|---|---|---|
| 1 | 正确 | check 说了算——不手写对比、不自选容差；覆盖整个模型。greedy token 与 transformers 逐 id 一致，跑几十步 |
| 2 | 两侧同形 | check 比较的那对孪生里，语义侧的 decode step 是一个 mega 程序；stage 边界和 barrier 在哪里，你说得出来 |
| 3 | 一次 launch | 每步 launch 数 = 1，并连同 op-by-op 路径的同口径数字一起报告 |
| 4 | analyze 推到底 | 第 2 步的记录在那里；最终数字连同 roofline 下限一起报告 |
| 5 | 那张表 | 稳态 tok/s，从 0 到 40960 每个长度一个数，三行并排：你的、torch_npu 上 transformers 贪心解码的、roofline 下限（下限 = 该长度下每 token 实测字节 ÷ 实测 HBM 带宽）。基线在同一张卡、同一份权重上自己量——不要引用别处的数字；测前先热身 |
| 6 | 选择有据 | 表里任何一处，你都能说出现在这个版本为什么是它：试过哪些、各自量到多少。只报最终一个不算 |

最后一关，从仓库外运行、打印续写文本、并报告覆盖整个生成的 tokens-per-second——
在足够长的生成上量，2048 个新 token、2000 字以上：
    python run.py --prompt "详细解释一下昇腾 NPU 上一次矩阵乘法是如何执行的" \
        --max-new-tokens 2048
```

## Files

- `model.py` -- the mega decode step as authored HIR (`Qwen3Mega.mega_step`,
  TileFoundry Module targeting `huawei.ascend910b2c`).
- `runtime_model.py` -- its `@runtime_module` twin: packs the HIR-layout
  weights into the kernel's NZ-fractal slabs, stages the KV caches, and
  serves `mega_step` with ONE AscendC launch.
- `kernel/mega.cpp` -- the mega kernel (~1000 lines, heavily commented).
- `kernel/build.sh` -- plain bisheng build (NEVER `-cce-enable-mix`) ->
  `libmegak.so`.
- `test_mega.py` -- op-by-op validation harness vs torch (random weights).
  ALL gates PASS: sl/lstop = 0/1, 0/28, 100/1, 100/28, 5000/1, 5000/28,
  20000/28 (argmax exact in all).
- `model_run.py`, `run.py` -- real-weight greedy decode CLI + bench.
- `bench.py` -- kernel-only launch bench.
- `OPTLOG.md` -- check gate, analyzer numbers, traffic model, roofline table.
- `SESSION_REPORT.md` -- measured time breakdown of the bring-up session
  (from the opencode session log) with phase analysis.
- `acts`, `prepared` -- symlinks to the npuir example's teacher-forced
  activations and HIR-layout checkpoint repack (check inputs).

## Check gate (from this directory)

```
QWEN3_CKPT=/path/to/qwen3_1_7b ASCEND_RT_VISIBLE_DEVICES=0 \
tilefoundry check runtime_model.py:Qwen3MegaRT.mega_step \
    --inputs files:acts/token_ids.pt,acts/cos_cache.pt,acts/sin_cache.pt,acts/pos_ids.pt,acts/scale.pt,acts/k_caches.pt,acts/v_caches.pt \
    --weights ckpt:prepared --dim ctx_len=256 \
    --out output[0] --fn allclose --atol 0.15 --rtol 0.05 --fn rel_l2 --max 0.02 \
    --out output[1] --fn equal \
    --out output[2] --fn allclose --atol 0.3  --rtol 0.05 --fn rel_l2 --max 0.05 \
    --out output[3] --fn allclose --atol 1.0  --rtol 0.05 --fn rel_l2 --max 0.05
```

PASS: logits rel_l2 0.0073, next_token equal, k_rows rel_l2 0.0048,
v_rows rel_l2 0.0117 -- the HIR interpreter is the reference, the AscendC
launch the candidate, at the npuir gate's documented tolerances.

## Run

### 1. Environment

```sh
conda activate tilefoundry
```

- Weights: HF checkpoint at
  `/home/tilelang/zuochuanuong/weights/qwen3_1_7b` (`WDIR` in
  `model_run.py` -- edit it there for a different location; Qwen3-1.7B,
  rope base 1e6).

### 2. Build the AscendC kernel

```sh
cd kernel && ./build.sh           # -> libmegak.so  (~40-60 s)
```

which is exactly:

```sh
bisheng --npu-arch=dav-2201 -std=c++17 -xasc \
    -I$ASCEND_HOME_PATH/x86_64-linux/asc -I$ASCEND_HOME_PATH/include \
    -L$ASCEND_HOME_PATH/lib64 -lruntime -lascendcl -lplatform -lc_sec -ldl -lm \
    -fPIC --shared mega.cpp -o libmegak.so
```

`libmegak.so` carries the `.aicore_binary` section (the auto-identified
MIX kernel) and the `run_megak` host launch stub; `build.sh` greps both
out of the ELF before printing OK. NEVER add `-cce-enable-mix` (doubled
`_mix_aic_mix_aic` symbols -> `rtDevBinaryRegister` fails 107000 -- see
the platform rules below). `./build.sh [src.cpp]` accepts an alternative
source file (default `mega.cpp`).

### 3. run.py

```sh
# greedy decode: tokens echo live; ends with prefill + decode stats
ASCEND_RT_VISIBLE_DEVICES=0 python run.py --prompt "详细解释一下昇腾 NPU 上一次矩阵乘法是如何执行的" --max-new-tokens 2048

# prompt from a file instead
ASCEND_RT_VISIBLE_DEVICES=0 python run.py --prompt-file input.txt --max-new-tokens 512

# the context sweep behind the performance table above
ASCEND_RT_VISIBLE_DEVICES=0 python run.py --bench ctx
```

Options (`run.py` is a thin wrapper over `model_run.py`'s CLI -- the two
are interchangeable): `--prompt` (default "The capital of France is"),
`--prompt-file`, `--max-new-tokens` (default 64; prompt + generation must
fit 40960 = max_position_embeddings), `--bench ctx`. Greedy is the only
decode mode; the whole 28-layer step is ONE kernel launch per token.

## Correctness evidence

- `tilefoundry check` PASS (above): HIR == AscendC kernel, all 4 outputs.
- Teacher-forced argmax vs HuggingFace greedy: **162/162 positions** over
  400 tokens of Chinese text (the only diffs anywhere are exact bf16 ties,
  gap 0.000).
- Free-running greedy is deterministic (8/8 identical runs) and matches
  HF's trajectories up to exact ties. HF itself loops/degenerates on long
  greedy generations of the base model -- so does this kernel (faithfully).

## Performance (28 layers, real weights)

|   ctx | ms/tok | tok/s | HF tok/s | must-move GB | eff. GB/s | roofline ms |
| ----: | -----: | ----: | -------: | -----------: | --------: | ----------: |
|     1 |   8.65 | 115.6 |     60.1 |         3.46 |       400 |        2.77 |
|   128 |   8.78 | 113.9 |     61.4 |         3.47 |       395 |        2.78 |
|   512 |   8.73 | 114.6 |     62.0 |         3.52 |       403 |        2.82 |
|  2048 |   8.84 | 113.1 |     69.3 |         3.69 |       418 |        2.96 |
|  8192 |   9.69 | 103.2 |     66.4 |         4.40 |       454 |        3.52 |
| 16384 |  10.90 |  91.7 |     62.8 |         5.34 |       490 |        4.28 |
| 32768 |  13.94 |  71.8 |     60.2 |         7.22 |       518 |        5.78 |
| 40000 |  15.04 |  66.5 |     60.2 |         8.04 |       535 |        6.45 |

HF baseline: cached greedy decode on the same card/checkpoint, measured
fresh. Roofline = must-move bytes / 1248 GB/s (measured d2d copy
bandwidth). The kernel runs at 52-69% of the rate the same card sustains
on the same pattern (the npuir kernel's 775-797 GB/s); the gap is the ~400
plane `SyncAll`s gating fixpipe visibility plus the ~258 MB/token stage
round trips (see below and OPTLOG.md).

## AscendC on Atlas 910B: the complete bring-up record

The kernel is ~1000 lines; getting them correct took **13.5 hours of
active session time, of which ~10 hours was discovering platform
semantics** (measured from the session log; the phase breakdown and the
reduce-it-next-time proposals are in session.md). This section is
the durable copy of everything the platform taught us. Every claim below
was verified on hardware (chip 0, CANN 8.5.0, dav-2201) by a minimal
probe before it was trusted; the probe sources live in
`/tmp/opencode/smoke3` (84 `.cpp` files + run scripts -- an ephemeral
path, which is why this section is self-contained).

### The hard-won platform rules (quick reference)

1. Compile WITHOUT `-cce-enable-mix`; auto-identify via the
   `KERNEL_TYPE_MIX_AIC_1_2` global. Launch via the `<<<>>>` stub.
2. FFTS mode-2 cross-core flags are broadcast-increment: a wait passes
   after ONE setter -- useless for all-24-AIC gates.
3. AIC fixpipe output is NOT in L2 when FIX_M fires (async FIX-unit path);
   an AIC core-cache flush does not help. Every AIV read of fixpipe output
   goes through poll + SyncAll; under real-weight L2 pressure even a single
   barrier is not enough -> DOUBLE SyncAll per gate (deterministic).
4. Host DMA readback of fixpipe buffers can see stale DDR: consume results
   via AIV-written buffers (part_max/part_idx) or bridge with a torch op.
5. ccec hoists MTE2 load issues past PipeBarrier: reload loops MUST
   ping-pong their UB buffers.
6. `Cast(f32->f32, CAST_NONE)` corrupts data on dav-2201.
7. Fixpipe `dstStride` is the full output row stride; tree_sum/tree_max
   destroy their tensor (land Casts before reduces).

### Method: probe first, then build

- Every primitive question got a minimal standalone probe: one `.cpp`, one
  question, an exact compare against a torch reference -- never a change
  inside the mega kernel first. 84 probes total; the load-bearing ones are
  indexed at the end of this section.
- The kernel itself was debugged with throwaway dbg scripts that dump
  intermediates to GM and compare per-op against test_mega.py's torch
  reference (~19 iterations). The discipline that paid: one build+run
  cycle must decide between ALL live hypotheses -- a bisheng compile of
  the kernel costs 40-60 s, so each dump covers every candidate
  explanation, not one.
- Always run hang-prone binaries under `timeout`: sync bugs HANG, and a
  hung process holds the NPU context until killed.
- Probes are code too, and they bit back: p15vis alone carried 3 self-bugs
  (SyncAll participant count, an OOB dump buffer, wrong row math) before
  its answer was trustworthy. Budget for probe debugging.

### Toolchain and the mix launch path

- bisheng must compile the mix kernel WITHOUT `-cce-enable-mix`: with the
  flag the .so carries DOUBLED `_mix_aic_mix_aic` symbols and
  `rtDevBinaryRegister` fails with error 107000. Auto-identification
  works via the global
  `auto __enable_feature_for_compile_default = KERNEL_TYPE_MIX_AIC_1_2;`
  plus cube ops (TPipe/TBuf/DataCopy) in the AIC branch; this emits the
  single-suffix `_mix_aic`/`_mix_aiv` pair.
- Launch through the bisheng `<<<>>>` host stub compiled into the .so
  (`extern "C" run_megak`): internally `rtDevBinaryRegister(magic 0x43554245, .aicore_binary)` + `rtFunctionRegister(plain kernel name)`
  + `rtKernelLaunchWithFlagV2(rtArgsEx_t)`. A hand-rolled legacy
    `rtKernelLaunch` HANGS on mix kernels. `blockDim=24` yields 24 AIC +
    48 AIV blocks.
- The FFTS base address comes from `rtGetC2cCtrlAddr()` and is ONLY valid
  after a device context exists -- allocate any NPU tensor first (it
  returns 0 pre-context). The kernel guards
  `if ((uint64_t)ffts != 0) SetSyncBaseAddr((uint64_t)ffts);`.
- A `constexpr` helper (SEQ_WANT) produced wrong values on the device
  until made an `__aicore__ inline` function -- a ccec constant-folding
  quirk that cost the first gate attempt.
- `ld_dev()` is NOT a GM load (it crashes on the cube). Direct
  `*(volatile __gm__ T*)` reads work, paired with `dcci`.
- `get_block_idx()` silently returns wrong values on AIC; the API is
  `GetBlockIdx()`.
- Environment: chip 3 (the default `ASCEND_RT_VISIBLE_DEVICES`) has MTE
  ROB ECC hardware faults -- check `npu-smi` and always use chip 0.

### Synchronization semantics (what each primitive actually does)

- FFTS mode-2 cross-core flags are broadcast-increment: every set adds +1
  to a level counter on EVERY core; a wait passes as soon as the level
  moves. With 24 AIC setters, the AIVs proceed after the FIRST AIC sets
  -- fine for single-setter gates, useless for all-24-AIC gates. (This
  misread -- inherited from an early probe conclusion -- is what forced
  the sync redesign; the 12-probe `mx_*` mode x pipe matrix plus `mixv`
  nailed the real semantics.)
- AIC -> AIV notify: `CrossCoreSetFlag<2, PIPE_FIX>(id)` on the AIC,
  `CrossCoreWaitFlag<2, PIPE_S>(id)` on the AIV. Works with an EMPTY fix
  pipe; all 48 AIVs may wait on one flag; ids are 4-bit (0-15) and
  reusable once all waiters consumed (12 rounds on 2 alternating ids
  verified).
- AIV -> AIC notify: NO cross-core flag mode works on dav-2201 -- the
  full mode x pipe matrix (`mx_0..mx_3` x `PIPE_{FIX,MTE3,S}`) hangs or
  crashes. Use GM flag polling instead: the AIV writes a flag word to GM
  (SetValue + S_MTE3 event + DataCopy); the AIC polls with
  `dcci(p, SINGLE_CACHE_LINE, CACHELINE_OUT)` + a compiler barrier + a
  volatile read. Each round MUST use a DIFFERENT address -- re-polling a
  rewritten address hangs.
- Scalar->MTE3 (`S_MTE3`) event ids are ONE-SHOT per kernel launch:
  reusing an id for a second Set/Wait pair hangs. Cycle ids (a pool of 4
  verified); `V_MTE3` id 0 reuse is fine.
- Plane barrier: `SyncAll<false>(gmWs, ubWs, 48)` -- the `isAIVOnly=true`
  default HANGS in a mix kernel. GM workspace = 8 int32 x 48 blocks,
  host-zeroed ONCE at allocation; counters are monotonic, so multi-round
  reuse works without re-zeroing.
- The final protocol (under which all gates pass): AICs publish
  completion as seq rows 24..47 of the sync workspace by direct store +
  `dcci(CACHELINE_OUT)` (the `kfc_comm` pattern, probe p14aicw); AIVs
  write rows 0..23. A consumer polls its one assigned slot
  (`24 + bid%24`), then joins a plane `SyncAll<false>` -- the barrier both
  aggregates the 24 AIC confirmations and covers the fixpipe L2-landing
  delay (next subsection). The seq tag starts at 512 so launch 0, round 0
  cannot collide with the host-zeroed workspace.

### The fixpipe/L2 visibility problem (the hardest bug class)

- AIC fixpipe output is NOT in L2 when the FIX_M event fires -- the write
  still sits in the FIX unit's async path. Probe p15vis measured it
  directly: a bare poll-then-read saw **1318/1440 stale reads**;
  poll + SyncAll saw **0/1440** (the barrier's delay covers the landing);
  an AIC `ENTIRE_DATA_CACHE` flush does NOT help (1313/1440 stale).
  Consequence: every AIV read of AIC fixpipe output goes through poll +
  plane SyncAll.
- Under real weights (622 MB of Wlm streamed per step) the landing window
  widens beyond a single barrier: roughly 1 launch in 6 flipped a token
  (non-deterministic argmax). FIX: DOUBLE `SyncAll` after every poll gate
  (8 sites in mega.cpp) -> 8/8 identical free-runs and 162/162
  teacher-forced argmax vs HF. Cost: ~2.8 ms/launch (7 gates x 2 barriers
  x 28 layers, ~400 plane SyncAlls) -- the dominant, fully identified gap
  to the roofline table above.
- Host DMA readback of fixpipe-written buffers can ALSO see stale data:
  dirty L2 lines are not snooped on D2H through the bisheng launch path
  (the stock runtime apparently does cache maintenance that the stub
  path lacks). The first prefill readback showed -1e30 sentinel values
  while the AIVs read coherent data. Workarounds: consume results via
  AIV-written buffers (`part_max`/`part_idx` are MTE3-written and
  host-visible), or bridge with a torch op (`logits.clone()`) before
  `.cpu()`.
- Identified but NOT implemented: sentinel gates -- `dcci`-poll a word
  INSIDE the fixpipe output until it leaves a per-launch host-prefilled
  sentinel value. Guaranteed visibility without any barrier; estimated
  ~3 ms/launch recovered.

### Compiler behaviors that corrupt data (ccec)

- ccec HOISTS MTE2 load issues PAST `PipeBarrier` (software pipelining).
  A UB buffer reloaded in a loop while vector ops still read it receives
  the NEXT iteration's data -- the symptom is a +1-iteration shift in the
  dependent op. `PipeBarrier` does NOT stop it; `SyncAll` does (its
  internal PIPE_ALL drain). The fix for reload loops that cannot take a
  SyncAll: ping-pong the UB buffers (`sc0/sc0b`, `ggu/ggub`, `uup/uupb`,
  `acc/accb`, argmax `lgc/lgcb`/`idxc/idxcb` in mega.cpp).
- `Cast(f32->f32, CAST_NONE)` returns garbage on dav-2201. `f32->bf16`
  (RINT) and `bf16->f32` are fine. Where a copy was intended, use direct
  GM loads per region instead (probe p13norm).
- `tree_sum`/`tree_max` DESTROY their tensor in place (the reduce folds
  by halving adds). Land any Cast/Copy of the data BEFORE the reduce --
  the probs bug was `Cast(hb, ex0)` issued after `tree_sum(ex0)` writing
  tree partials into the probs plane.
- Fixpipe `dstStride` is the FULL output row stride. The LM-head gemv
  wrote a (16, NB*SPANP) buffer with `dstStride = SPANP`: row 0 was fine,
  rows 1+ overwrote neighbouring cids' spans (cid 0 correct, cid >= 1
  garbage).
- gemv B-side fractal indexing must use the PADDED pack width
  (`NF = nPad/16`), not `nGlobal/16`: the host-side pack pads rows, which
  adds n-fractals per k-block on the device side.

### The A-side fractal broadcast (probe11, verified exact vs torch)

AIV builds the fractal pack_a layout in UB from plain x (2048 -> 32768
elements):

```cpp
CopyRepeatParams cp;  // units = BLOCKS (32B = 16 el for bf16)
cp.dstStride = 1; cp.srcStride = 0; cp.dstRepeatSize = 8; cp.srcRepeatSize = 0;
for (int j = 0; j < KF; j++)   // KF = K/16
    Copy(pack[j*256], x[j*16], (uint64_t)128, (uint8_t)2, cp);
```

- This is the official `TwoDimBroadCastLastDimAlign220` pattern
  (broadcast_v220_impl.h): per call, 2 repeats x 8 blocks, `srcStride=0`
  re-reads the same source block -> 16 consecutive dst blocks = fractal j.
- Strides are in BLOCK units; mask=128 (count mode) = 8 blocks/repeat
  (16-bit elements).
- `BLOCK_MODE_VECTOR` DataCopy (GM->L1) does NOT broadcast on dav-2201
  (probe9); `Brcb` (16-bit) sends each source ELEMENT to one 16-el dst
  block -- neither is usable for block broadcast.
- AIV sub-block 0 writes the pack alone (halves GM write traffic); the
  cube reads it GM -> L1 -> `LoadData` rt=16 ss=1.

### Validation/reference subtleties (what the test harness had to learn)

- The kernel's LSE partials (part_l / partAcc) are UNNORMALIZED per-slice
  exp sums: the reference must accumulate in the same LSE form, not a
  softmax.
- The argmax result ci is the SPAN-GLOBAL index (into idxTab); the
  absolute vocab id is `tok*SPAN + gi` -- there is no per-row CH offset.
- The K/V append of step sl lives at `(sl//16, sl%16)` of the LAST
  layer's cache block -- the block ring wraps per layer.
- Attention scores must be compared AFTER the `1/sqrt(d)` scale: the
  kernel's raw dp plane is pre-scale.
- With random weights, argmax near-ties (< 0.2 logit gap) are numerics
  coin-flips, not bugs: the harness accepts either top-1 within that band
  (real-weight runs had zero such cases; the only diffs anywhere are
  exact bf16 ties).
- Reload-free argmax on the vector side: `work = base + base`
  element-wise, then `cm = 0.5 * tree_max(work)` -- exact, and it dodges
  the MTE2-hoist trap entirely.

### Probe index (the load-bearing subset of 84)

| probe(s)                            | question                                              | answer                                                                                   |
| ----------------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| gemv.cpp .. gemv7.cpp               | single-core GEMV over the B-side NZ fractal layout    | gemv7 recipe: inner-N NZ pages, full-K per fragment, K-step accumulation across cal data |
| probe9                              | does BLOCK_MODE_VECTOR DataCopy (GM->L1) broadcast?   | NO on dav-2201                                                                           |
| probe11                             | A-side fractal broadcast in UB                        | the CopyRepeatParams recipe above, exact vs torch                                        |
| probe16/16a                         | legacy-launch arg passing                             | contributed to abandoning the legacy path for mix                                        |
| mx_0..mx_3 x PIPE_{FIX,MTE3,S} (12) | which FFTS cross-core mode/pipe combos work AIV->AIC? | NONE (hang or crash)                                                                     |
| mixv                                | mode-2 flag semantics with multiple setters           | wait passes after ONE setter (broadcast-increment)                                       |
| mixempty                            | AIC->AIV CrossCore flag with EMPTY fix pipe           | works (all 48 AIVs, 4-bit ids, reusable)                                                 |
| mix2addr                            | AIV->AIC GM flag polling, multi-round                 | works; per-round DISTINCT addresses mandatory                                            |
| mix2sync2                           | plane`SyncAll<false>(gm, ub, 48)`                   | works multi-round;`isAIVOnly` default hangs in mix                                     |
| mixev                               | the full recipe end-to-end, 12 sync rounds            | verified (runmev.py)                                                                     |
| p13norm                             | `Cast(f32->f32, CAST_NONE)`                         | corrupts (garbage out)                                                                   |
| p14aicw                             | AIC direct store + dcci CACHELINE_OUT publish         | works -- the seq-row pattern                                                             |
| p15vis                              | fixpipe L2 landing rates                              | bare poll 1318/1440 stale; poll+SyncAll 0/1440; AIC cache flush useless                  |

The remaining ~70 probes covered fractal layout math (fract/layout/lay2),
phase timing (phasemark), host-launch experiments (host2/host3,
mix5_host/mix7_host), and the many failed sync schemes (mix1-mix8,
mixab/mixup*, mixs3*, mixp*/mixpoll*) whose collective failure is itself
the finding: on this chip, the only reliable AIV->AIC edge is a GM poll,
and the only reliable all-48 aggregation is the plane `SyncAll<false>`.

### What it cost

Measured from the opencode session log (full analysis in
session.md): 13.5 h active work over a 22.7 h window, 1542
messages, 1580 tool calls (1377 bash / 94 edit / 63 write / 45 read),
16 kernel builds. About 10 of those hours went to platform discovery --
i.e., the content of this section is the compressed output of ~75% of the
project's active time. The single most expensive mistake: designing the
v1 sync scheme on an unverified assumption about flag semantics (~1.5 h
lost to the redesign). The single most effective practice: one-probe-one-
question with exact torch references, which turned an undocumented
platform into a checklist.
