# 精度与卡死：按证据缩小范围

## 精度诊断的三轴

分开判断：**输入是否相同、同输入执行是否确定、确定性结果是否符合独立 reference**。R30 的经验是三者都可能出错，不能把所有漂移都归因于浮点误差。

1. 保存 token、pos、原始权重标识、完整 KV、RoPE 表、scratch 初始状态、配置和 binary hash。先对输入 hash，再对输出 hash。固定 teacher forcing，避免第一个 token 分歧之后整段 trajectory 都换了输入。
2. 同输入同库重复；同库独立进程重复；同精度 control 与 candidate 独立进程比较。每个进程尽量只加载一份设备库。检查完整 logits 和完整 KV，不能只看 argmax。
3. 找到首个错误 layer/stage/tile，用独立原始权重做 operator-local 对照，再从原始输入独立传播全轨迹。前者定位局部计算，后者揭露多层误差积累。

## 调试交接：把报错变成可重复的现场

最有帮助的输入是一个完整失败fixture和一条确定的运行命令。仅提供prompt、最终token或maxdiff，通常不足以恢复当时的KV、RoPE、scratch与设备状态。优先保存原失败现场，再尝试缩小；seed不能替代完整状态，输出hash也不能替代可供重放的输入数据。

```yaml
case_id: long-kv-first-divergence
status: measured_failure
source_binary_hashes_and_build_flags: <实际值或记录路径>
device_toolchain_and_health: <型号、设备、CANN/编译器版本、实验前后健康>
reproduce_command_and_working_directory: <命令与cwd>
inputs:
  checkpoint_hash_and_logical_layout: <权重身份和方向>
  token_position_and_lstop: <当前token、零基位置、层数>
  kv_rope_scratch: <完整tensor文件、shape/dtype/layout、hash及初值规则>
  stream_and_launch_epoch: <队列提交顺序和tag；重放时按协议分配新tag>
reference: <独立reference版本、输出/中间值、预先约定的容差>
first_failure: <首个layer/stage/tile/元素，期望值、观测值、误差指标>
repeatability: <同进程/跨进程、同库/控制组，失败次数与总次数>
logs_and_trace: <原始路径；hang时最后已知wait与want/observed>
unknowns_and_next_discriminating_test: <尚未排除的原因与下一实验>
```

若fixture很大，先记录可定位原文件的manifest和hash，并保存首错附近dump；只有证明最小fixture仍复现同一问题后，才用它替换完整现场。消费时刻的数据和kernel结束后的最终dump要分别标明。

诊断方向由区分实验决定：输入hash不同先查输入生成；输入相同但执行漂移先查可见性/生命周期；结果稳定而逐层误差增长再查数值语义。将这些已测现象提前交接，比直接指定“多加屏障”或“放宽容差”更有价值。已有现场由用户提供即可；首错定位、重复性矩阵和根因验证由agent继续完成。

## 最短诊断阶梯

| 层次 | 覆盖 | 排除的问题 |
|---|---|---|
| packing/primitive | 小矩阵、逐元素精确比较、rounding 边界 | offset/stride、转置、NZ、原语语义 |
| 单层单步 | pos=0 和带历史，dump 每个 stage | 顺序、mask、norm、局部交接 |
| 全层短上下文 | 每层中间结果与 logits/KV | scratch 复用、跨层同步、累计误差 |
| 固定长 KV 重复 | 完全相同输入、轮换 token、严格全词表 | sentinel 重臂、缓存可见性、状态污染 |
| 自然前缀自由生成 | 中英文、4k/16k/40k，独立进程 | 实际状态分布、跨进程 RoPE、轨迹差异 |
| 最大上下文 | append 到最后合法位置，检查尾部 | 容量、空 slice、tile 尾部、off-by-one |

本案例 R30 的门槛与次数见硬件层证据；换模型预先确定精度阈值，不在失败之后放宽阈值。诊断 drift guard 与独立 reference 的精度阈值是两回事。

## “症状 → 区分实验 → 修复方向”

| 症状 | 下一步实验 | 典型原因/处理 |
|---|---|---|
| 跨进程不一致，单进程可能一致 | 比较全部输入 hash，特别是 RoPE | CPU 三角函数并行路径不确定；确定性建表，不改 kernel 容差 |
| 最终 GU dump 一致，但 SiLU 曾读错 | dump 消费时刻输入，按 tile 记录 marker/epoch | 最终值正确不代表消费时已落地；认证 live 行和 reader 生命周期 |
| 越多层误差越大，重复稳定 | 独立传播 vs operator-local；检查 norm 近似误差 | Rsqrt 误差在 bf16 舍入前积累；提高该处精度而非整体放宽容差 |
| 日志出现 -1e30/NaN marker | 比较 device 消费与 host 读回，检查 mask/尾部 | 未写区、陈旧数据或读回不一致；不能默认是 top-1 near tie |
| 相邻迭代串值 | alternating token/值模式，关闭 buffer alias | MTE2 提前覆盖 vector reader；显式 reader-done 和 ping-pong |
| 第一核正确，其余核错 | 递增矩阵 dump 每核所有行 | padded width、全局 dstStride、slab 偏移错误 |
| 局部相差统一倍数 | 对照各 slice 的 shift、分母和 numerator | softmax 分片尺度不可比，或 denominator 提前读 |
| 只在最大长度/某个非整 tile 失败 | 测 tile±1，head/slice 边界，空分片 | mask、补零、索引和空任务 ack |
| 加屏障后暂时好转 | 对原始数据做 readback/landing probe | timing race 被延迟遮住；仍需证明依赖，不能靠 sleep 交付 |

R30 的两步 Newton `r=r*(1.5-0.5*x*r*r)` 是当前硬件 Rsqrt 的精度修复。硬件无关的是“近似误差须在模型舍入点之前达标”，不是每种硬件都固定做两次。

near tie 只能在完整 logits 已符合既定容差、数据有限且确定时解释个别 argmax 差异。随机模型的近似 top-1 规则不能替代真实模型严格 greedy 契约。

## 卡死的定位路径

可疑实验放在有时限的独立进程，timeout 后终止该实验进程并确认退出；先查设备状态与 runtime 日志，再跑已知良好小探针。不要因出现 ECC 就直接认定为原有硬件问题，也不要自动重置其他人的设备。

把等待写成 wait-for 图：`consumer -> channel/epoch/grain -> producer -> producer 当前等待`。逐项核对：

- producer 是否启动、分支是否进入、空任务核是否遗漏 publish/ack；
- 所有 barrier participant 是否到达，核型/人数是否符合 adapter；
- tag 是否撞初值、上一轮或溢出，`==` 和 `>=` 是否符合协议；
- 每个 outstanding event 是否有匹配 Set/Wait，channel 是否被另一流水阶段占用；
- buffer alias 是否让 producer 等 reader、reader 又等下一次 producer，形成环；
- scalar/MTE/vector/cube/FIX 的源和目的完成是否混淆；
- host 是否在 DMA 完成前重用指针、队列是否实际提交到 stream。

诊断 trace 记录 `{launch, plane, core, layer, stage, before_wait, after_wait, want, observed}`。每核占独立 cache line，编译开关默认关闭。host 杀进程后不保证在途 trace 都可读取，优先用单层/单交接最小探针；不要将可能停在 kernel 后的 Python dump 当成一定能拿到的证据。

二分缩小：单原语 → 单核 → 两核 → 多 producer → 多轮同地址/异地址 → L1/UB 双缓冲 → 全层。一次编译尽量布置足以区分所有当前假设的 dump，避免 40–60 秒编译只回答一个可预见的小问题。

R11 的日志先判“预存硬件故障”，随后证明候选运行会使后续结果异常，故撤销初判。教训是候选前后都跑健康对照；L1/event 越界类失败出现设备异常后，停止重跑该候选，先静态审计和最小化。
