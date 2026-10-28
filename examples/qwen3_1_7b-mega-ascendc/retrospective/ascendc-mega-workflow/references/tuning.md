# 从正确基线到性能平台的方法

## 0. 先定义能比较的目标

同卡、同权重、同精度、同同步保证、同 context/KV 状态下比较。维护三个计时口径：设备 kernel；launcher wall time（可能含清理、同步、读回）；完整生成（tokenize/prefill/采样/输出是否包含须声明）。固定位置零 KV 可以测吞吐，但不能认证自然长前缀正确性。

上下文集合至少含短、中、长、最大长度与 tile/slice 边界；性能目标是整张目标分布，不是只挑 ctx=1。验收精度先过，再比较性能。

### 优化前哪些信息能给出明确方向

| 信息 | 必须说明的口径 | 如何转成下一步修改 |
|---|---|---|
| 业务负载与目标 | context分布/权重、生成长度、关注平均还是尾延迟、kernel还是端到端 | 短ctx优先检查交接/host，长ctx优先审计KV/主存；由测量确认 |
| 可改与不可改项 | 精度/greedy、单launch、量化/batch、重排、启动预处理、scratch预算 | 先排除不满足契约的方案，避免优化完成后被验收否决 |
| 同精度正确baseline | 源码/库hash、卡与时段、输入KV状态、热身和统计方式 | 建立能做A/B的control；旧错误版本不能充当性能目标 |
| 分阶段timeline和依赖 | 多核分布、wait/compute/transfer、可重叠区间、插桩开销 | 找未隐藏的关键路径，不把stage耗时简单相加 |
| 字节账本与可持续带宽 | 逻辑/padded/实际事务、HBM/L2、GB/GiB、读或读写pattern | 估计可消除字节与收益上限，决定布局还是流水优先 |
| 既有实验台账 | 变体/配置、精度、各context性能、噪声、保留/撤回原因 | 排除已证伪方向；调度或硬件改变后有依据地重新评估 |

例如，本项目R29的RMS输入等待与约6–7us的独立argmax尾部，使“压缩输入准备”比“只优化argmax”更有收益空间；R16识别probs广播膨胀后，约496MB的可消除流量可直接换算为约397us的预计收益。这些数字仅解释当轮选择，新模型需重新测量。

用户主要提供目标分布和可接受的取舍；timeline、字节审计、带宽测量和消融由agent获取。没有profiler记录时先建立正确baseline并测量，不能让用户先猜瓶颈，也不能将缺少完整交接包视为停止工作的理由。

## 1. 先算字节，不凭 FLOPs 选方向

对未量化 BF16 decoder，逻辑权重读取下限近似：

```text
B_weights = 2 * [ L*(H*(QH*D+2*KVH*D) + H*H + 3*H*FFN) + Vocab*H ]
B_KV(C)   = 2(K,V) * L * KVH * C * D * sizeof(cache_dtype)
B_total   = 实际 padded 权重 + KV + GM stage 往返 + 额外复制/重复读
```

GQA、投影维度或结构变化时按实际参数矩阵重算。区分逻辑最低字节、按实现推导的字节、profiler 主存事务和 L2 访问；它们不是一个数。GB 与 GiB、单向读与读写带宽要统一。

`T_lower >= max(B/BW_sustainable, FLOPs/compute_sustainable)` 是参考下界；测得的 BW 必须说明访问模式。本案例早期 d2d 的 1248 GB/s 不是新卡或所有读取模式的永久上限。当前性能超过旧字节模型推导的旧下界，应审计模型/带宽口径，不应宣称超越物理极限。

## 2. 找关键路径，而非最高占用率

先看 msprof 的 device duration、主存/L2/UB 事务与所有核分布，再用编译期开关的少核 timeline 定位每层等待：RMS input、QK、softmax、PV、combine、GU、Down、LM/argmax。插桩本身有扰动，不当作最终性能。

- MTE2 高、cube 低：先减少 padded/重复/广播字节，改善连续性。
- 短 ctx 慢而长 ctx 差异小：检查交接延迟、pack、host round trip。
- AIV scalar 高：可能在 poll，不能直接推导成“标量计算瓶颈”。
- 等待长但传输已饱和：增加 ring 深度未必有收益，可能只是移动等待。
- 局部阶段占比小：先算 Amdahl 上限。6 us 的 argmax 即使消除，对 3 ms step 也不到 0.2%。

## 3. 优化顺序：优先可证明的大收益

| 顺序 | 操作 | 进入前提与验收 |
|---|---|---|
| 1 | 从保守全屏障改为准确的数据依赖交接 | 先认证 landing、publish、reader 完成；不能只删 barrier |
| 2 | GM 权重按消费次序打包，连续化 KV | pack/unpack 精确一致；比较字节与 MTE2 |
| 3 | 双缓冲和有限深度预取 | 每 slot ready/free 完整；buffer 容量和 event 不冲突 |
| 4 | 消除重复 vector 工作，批量 pack/store | 检查唯一 writer、UB 生命周期和舍入不变 |
| 5 | 从广播 GM 改 compact GM，在本地展开 | 只压缩无效行，不丢有效数据；按节省字节预测收益 |
| 6 | score→softmax→PV、SiLU→Down、LM→argmax 分块交接 | grain 边界与空核 ack 正确，逐项测量 overlap |
| 7 | 减少数值 pass、复用历史统计 | 稳定范围有证明或 fallback；新模型重新评估 |
| 8 | host 清理/同步/结果读回 | 维持队列与 raw launch 的真实顺序 |

这是收益优先级，不是固定必须全部应用的菜单。L2 warming、增大 tile、增加 ring 只在计数器和关键路径支持时试；保留收益不足噪声的优化会增加同步维护成本。

## 4. 每个候选的一轮闭环

1. 保存 control 源码、库、编译参数、layout 和 hash。precision/sync 修复进入 control 后，再比较纯性能变体。
2. 写出假设、受影响阶段、字节/时间预算、风险与可证伪实验。例如“减少 496 MB 主存事务，1.25 TB/s 时约省 397 us”。
3. 单变量实现；布局变化同时更新真实权重 packer、随机 packer、独立 reference 的逻辑还原和 metadata。
4. 先最小算子与边界门禁，再重复确定性、全层真实长 KV 与最大长度。任何失败保存日志，暂不作为合法性能候选。
5. 多轮 A/B 与 B/A 交替，热身，固定测量次数；汇总 median/mean/离散程度，不只选 best。跨进程验证；设备忙或健康异常的样本剔除并说明原因。
6. profiler 独立验证 device duration 和假设对应的字节/占用率。变化若与假设不符，解释或回滚。
7. 对组合优化做消融。保留整个 context 目标集上的净收益，记录被拒候选。

实验记录模板：

```yaml
candidate: compact-a
source_hash: ...
control_hash: ...
device_and_toolchain: ...
layout_and_flags: ...
hypothesis: ...
hypothesis_status_and_supporting_evidence: ...
expected_bytes_or_us_saved: ...
correctness_reference_and_thresholds: ...
contexts_and_input_state: ...
warmup_rounds_iterations: ...
wall_ms_and_device_us: ...
raw_log_and_profiler_paths: ...
result: keep_or_revert_with_reason
coverage_unknowns_and_revisit_conditions: ...
```

失败台账区分精度失败、hang、设备异常、无收益、上下文局部退化与尚无结论，保留原始记录和适用版本。重跑旧失败方案前写出哪个条件已经变化，例如event预算、buffer alias或访问pattern；不能把失败记录解读为永久禁用，也不能无变化地重复同一实验。

## 5. 何时停止，如何说明“接近极限”

报告最终字节预算、实测 BW、每阶段未隐藏时间和相对于同访问模式下界的差距。剩余候选的理论上限低于噪声、或已受可持续内存带宽限制，可称为这组实验下的平台。不要把 cube 利用率低当作浪费：单 token GEMV 可能天然受带宽限制。

换模型后重新测 HBM/L2、score 范围、各阶段占比和 host 开销；只沿用这套决策流程，不沿用“GU warm 有用”“LM tile 96 最好”等答案。交付必须让后一个 agent 看得出什么是事实、什么是推断、什么仍未覆盖。
