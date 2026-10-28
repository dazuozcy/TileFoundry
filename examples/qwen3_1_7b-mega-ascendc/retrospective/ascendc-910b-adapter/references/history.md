# 生成与优化全过程：从平台探索到 R30

数据主源为 [OPTLOG.md](../../../OPTLOG.md)、[README.md](../../../README.md)、[session.md](../../../session.md)；各阶段数值是原记录的对应口径。跨日期/设备/输入/同步修复的数值仅展示演变，不作严格加速比。早期“当前/最终”均只指当轮。

## 初次生成：主要时间花在平台语义

会话统计：13.5小时活跃、22.7小时墙钟、1542消息、1580工具调用；约10小时用于平台发现。顺序是工具链侦察 → GEMV/fractal 探针 → mix/launch/flag 探针 → v1 → 推翻同步假设 → v2 → 中间量对拍 → 全层真实权重 → 补 HIR/twin/check。

约84个最小探针揭露：legacy mix launch 不可靠、mode-2 不是多 producer 汇聚屏障、Fixpipe event 与 L2 可见性不同、UB reload 提前、Cast/stride 等问题。v1 在未认证 flag/launch 语义前就开始设计，造成约1.5小时同步重做。临时调试脚本自身也有 barrier 人数、dump 越界和行号错误。

最早正确版本以 poll+双 SyncAll 掩盖 Fixpipe 落地延迟，ctx1约8.65ms、40k约15.04ms。历史 HIR check 四输出 PASS 后补齐原任务的 twin 门禁；这一历史结果不能替代后来布局变更后的重新验证。

## 哪些提前信息能改变路线，哪些属于执行责任

原任务已明确硬件、工具链、API来源、单launch与HIR/check要求。先设计未经probe认证的flag/launch方案、后补HIR，以及没有尽早建立严格重复验证，属于agent执行顺序问题；不能把这些返工都归因于用户缺信息。

| 阶段 | 若提前可得，最有价值的信息 | 能明确的方向 | 仍须由agent完成 |
|---|---|---|---|
| 初次构建/拼kernel | 可运行MIX最小包、参数回传、双平面交接代码、版本与认证范围 | 复用stub与已证实协议，避免v1建立在错误flag语义上 | 当前环境冒烟、布局/边界probe、全模型门禁 |
| R1–R6 | Fixpipe目的可见性证据、ready/free生命周期、stage等待timeline | 用数据落地认证替换时间延迟，再做有界预取与grain交接 | 证明marker覆盖真实consumer，验证reader重用与多轮 |
| R12–R16 | 实际HBM/L2事务、广播膨胀、同pattern持续带宽 | 优先compact probs等可消除流量，预估收益再实施 | 核对计数器口径，做A/B与字节消融 |
| R25/R29 | host与device分开计时，RMS输入等待和argmax未隐藏尾部 | 处理host往返与输入准备，避免高估独立argmax收益 | 测关键路径与组合消融，保留必要launch前同步 |
| R28–R30 | 完整失败fixture、输入hash、baseline-vs-baseline和跨进程重复结果 | 先查RoPE输入、确定性、局部首错和数值累计，暂停更激进同步优化 | 定位Rsqrt、GU live marker、event/history/队列等实际根因 |

这些是“如果已存在就应优先复用”的材料，不是假定用户在发现前就知道根因。R30的根因属于本轮调查产出，后续应作为带版本和覆盖范围的交接证据，而不是跨模型/跨硬件的无条件结论。

通用交接优先级见 [生成框架](../../ascendc-mega-workflow/references/framework.md)，调试现场见 [故障诊断](../../ascendc-mega-workflow/references/debugging.md)，优化决策输入见 [调优方法](../../ascendc-mega-workflow/references/tuning.md)。

## R1–R7：先消除大颗粒等待，再改数据流

| 轮次 | 改动与依据 | 当轮测量 / 结论 |
|---|---|---|
| R1 | sentinel 实测数据落地后 seq，去除全局 SyncAll | ctx1 178.4、40k 83.5 tok/s |
| R2 | GEMV/GEMM/LM MTE2 ping-pong | 197.1 / 93.5 tok/s |
| R3a | gate poll sweeps 重叠 | 200.7 / 93.5 |
| R3d | 权重按消费 chunk-major repack | 230.0 / 98.9 |
| R3e | SiLU 从24倍重复改分片 | 244.3 / 102.0 |
| R3f | V append 批量 RMW | 251.7 / 101.8；初始3轮 ctx1约8.34→3.97ms |
| R4 | 权重4槽 L1 ring，等输入时预取 B；收窄 QKV 等待 | 277.0 / 106.1，短上下文收益大 |
| R5 | 宽块 softmax、减 Fixpipe 写行、连续 K cache | 真实权重40k 106.1→127.8 tok/s |
| R6 | score/probs progress 分块，PV 与 softmax 尾部重叠，pass1扩宽 | 真实40k 127.8→134.9 |
| R7 | AIV 在特定 O 等待窗口 warm GU 权重到 L2 | 随机 ctx1 299.5→305.0，40k基本不变 |

R4 首次 ring hang 的关键是 A-load 与 B ring 通道/顺序冲突；不是“ring 设计无效”。R4 的6槽尝试当时反而回退，后来 R10 重新安排 alias/event 才保留，说明容量数字离开调度上下文没有意义。

R5 L0C 双缓冲、R6 AIC warmer、R7 Down warmer未留下：局部可重叠不代表端到端有净收益。Down warm 让短 ctx 305→286，属于带宽/流水竞争。

## R9–R17：用测量推翻错误瓶颈，再消除主存膨胀

| 轮次 | 发现 / 决策 | 证据意义 |
|---|---|---|
| R9 | K/V 路径若干“明显优化”均无收益 | 不凭等待表象推断缺预取 |
| R10 | GEMV ring4→6，用384KB L1 alias | 随机305.0→313.7 /139.8→142.1 tok/s |
| R11 | attention K ring6 实验失败，后续出现设备异常 | 初判预存故障后被推翻；候选可能触发异常，撤回并停止重跑 |
| R12 | 每 tile 时间与 Roofline 说明 attention 已接近 MTE2 限制；pack_out 合并 DMA | 314.3→318.3 /141.8→143.1；很多 Fixpipe/双缓冲想法无端到端收益 |
| R13 | score 按4096-position group 发布，max pass 与 QK 重叠 | 收益小但保留；随后审计发现 probs 非“都在L2” |
| R14/R15 | AIV 直接轮询 Fixpipe sentinel 代替 AIC 认证转发 | polling/arming 路径增加开销，回退；少一跳不一定更快 |
| R16 | probs 从16行广播 GM 改2行 compact，MTE2 Nd2Nz 展开 | 40k约568→72MB 往返，省约496MB；6.930→6.535ms，+6.0% tok/s |
| R17 | ready word 免 flag 版本错误；grain1024→2048无净收益 | ready/data 可见性不可混淆，粗 grain 也未必更省时 |

R16 是最清晰的“字节账本→性能”例子：496MB / 1.25TB/s≈397us，与实测395us接近。相比之下，Fixpipe batching 或 L0C 双缓冲省掉的等待被 K copy 隐藏，因此局部指令变少但总时间不变。

## R18–R27：减少重复工作、揭露可见性隐患、回到数值契约

- R18/R19：减少 AIV 重复阶段工作，按上下文限制 GU warm；L2 竞争会随上下文改变，不能无条件 warm。
- R20：删除无消费者的旧 scores-done publish，前提是新 progress DAG 已覆盖依赖。
- R21：删 softmax max pass 看似成功，但 sl=300失败。R22 恢复 pass 的延迟也不足以可靠修复。
- R23：定位 combine 过早读取 `part_l`，把读取移到 partAcc 落地认证之后。**被删除的计算曾意外充当同步延迟**。
- R24：在修好的 combine 上删 pass1，保留单遍计算。不能把此前失败都归咎于 softmax 数学。
- R25：kernel约3.11ms，而实际 decode约3.33ms；host约0.2ms/token 值得优化。减少读回/清理路径后，历史真实短 decode约300→315.2 tok/s。最终 R30 必要的 raw launch 前同步仍保留。
- R26：部分 streaming 想法搁置，先考虑收益上限及新交接的可见性成本。
- R27：shift=0 对 Qwen score 范围可行但泛化不安全，改同 head 三 slice 共享 carried shift，首层显式 bootstrap。score+200 测试证明优于固定0，代价约0.3%；跨层任意幅值突变仍需新模型认证。R30 再给 history 加双 bank，修复覆写风险。

## R28：小幅分块交接，有收益但没有精度闭环

Down 原本等全部24个 SiLU producer，改每8个 K chunk 就绪即加载一组，让后续组与前面计算重叠。6个实验仅保留 group8；group4 单层精度失败，cache Q、L0A驻留、早 refill、32bit ring cursor 无稳定额外收益。

独立 profiler：ctx1 3081.773→3068.889us，40k 6419.225→6394.586us；0.418%/0.384%。随机压力和短真实生成通过，但固定长 KV 仍有 baseline drift。此阶段没有证明长上下文真实序列等价。

## R29：输入准备/交接比 cube 算术更值得改

timeline 显示每层 RMS1/2 输入等待可缩短，LM约406us，但独立 argmax 尾部只有6–7us。于是保留：

1. compact GEMV A：GM 从16行广播变2行，在L1展开，减 pack/store/read。
2. paired GU：同一 AIC 产生配对 gate/up slab，SiLU 依赖更局部。
3. streamed LM argmax：每2048词表块认证落地后立即消费，不等完整 span。

消融 A/B wall time：ctx1 R28 3.058632→2.892764ms；40k 6.389681→6.204671ms。独立设备测量：3060.134→2904.298us /6398.972→6211.561us。40k 主存读8602.81→8384.06MiB，UB→GM132.48→51.54MiB。

LM tile128 虽减少循环且过门禁，ctx8192仍比96慢，回退。提前下一 stage 预取约退化12–15us/token，也回退。**少循环、更多重叠都不是收益证明**。

R29 仍在4096真实前缀的 baseline-vs-baseline 就分歧；日志明确不计为通过。scratch 初始化排除一项干扰但没有解决根因。

## R30：先封闭正确性，再重测最终性能

六类修复是本项目最值得复用的诊断案例：

1. 相同 CPU frequency tensor 可生成不同 cos 表，甚至同次表的两半不同。scalar FP64 libm 后落 FP32，8个新进程整表 hash 一致。
2. RMS Rsqrt 近似误差在28层累计，一步 Newton 最大长度仍2.192%误差；两步才过原2%阈值。
3. GU dead-row 不保证 live-row 及时；改逐128列 live marker，空 Down 核补 reader ack。
4. FIX_S / S_MTE2 明确目的完成与 arm 顺序；score/PV distinct event ID；GEMV drain 后复用。
5. UB reload/vector reader、scalar/vector scratch、A-group reader 都补生命周期；denominator读回，history双 bank。
6. torch_npu host queue 与 ctypes raw launch 顺序不能靠 raw stream handle 推断，launch前同步。

最终 control 是**同一 R30 生产源码、相同数值和同步**，仅关闭 compact-A/GU-interleave/LM-stream。control/repeat/release 的所有 FP32 vocabulary logits 与完整 KV bitwise 一致；独立 CPU reference 仍按数值容差检查。

| 场景 | terminal logits relL2 | cosine | 结论 |
|---|---:|---:|---|
| prefix4096，检查pos4223 | 0.01368172 | 0.99994087 | 通过 |
| prefix16384，检查pos16511 | 0.01039816 | 0.99997061 | 通过 |
| prefix40000，检查pos40127 | 0.00631288 | 0.99998081 | 通过 |
| live40960，检查pos40959 | 0.00943002 | 0.99995565 | 通过 |

各场景308项 operator-local checks；另有最大边界4进程833步、4096次轮换token/40k KV全词表严格比较、短prefill重复与70随机门禁等记录。

最终 device：ctx1 **2936.361us**，40k **6265.543us**；对当轮同精度 control 改善5.44%/2.72%。wall median为2.97551/6.29530ms，不能与device混用。40k AIC MTE2占用86.29%、ctx1 69.66%，说明权重/KV streaming仍主导。

R30 比某些早期不完整正确版本稍慢不应回退：最终目标是同精度约束下的速度。没有证据证明已达所有模型、所有输入和所有调度的全局最优。
