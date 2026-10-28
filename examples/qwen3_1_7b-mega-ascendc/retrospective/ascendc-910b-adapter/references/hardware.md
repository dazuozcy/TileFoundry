# 910B / dav-2201 / CANN 8.5 实现契约

## 适用范围与源码映射

以下描述当前 R30 源码及已有实机记录，不是跨型号 API 保证。API 真值优先查当前 CANN `kernel_operator.h`、`basic_api/interface/`、`highlevel_api/lib/` 和对应实现/例子；不要凭名称猜参数单位。

| 抽象能力 | 当前实现（路径相对项目根） |
|---|---|
| 设备入口 / host ABI | `kernel/mega.cpp` 的 `megak`、`run_megak`、`megak_layout` |
| 资源 / 预取 | `CubeCtx`、`GemvCtx`、`ring_push/fill`、`gemv_begin/ring` |
| compact 本地展开 | `load_gemv_a`、`nd2nz_probs` |
| scalar 发布 / 等待 | `aic_seq`、`poll_seq`、`aic_wait_prog` |
| 数据落地确认 | `aic_arm`、`aic_wait_land`，GU live tile marker |
| vector 发布 / reload | `seq_write`、各 `MTE*_V/S` 与 `V_S/S_V` 事件 |
| 数值原语 | `rms_norm`、`tree_sum/max`、`rope_apply` |
| host 管理 | `model_run.py:Runtime`、`test_mega.py` 的布局/launch 逻辑 |

## 提前交接哪些硬件信息最能减少探索

优先提供可运行的最小构建/launch包、已正确的跨平面交接代码和认证记录，而不只给芯片名或API名称。最小包应含源码、构建命令、ELF检查、host ABI、初始化/stream顺序、运行命令、预期输出、成功与失败日志。当前可复制起点为 [实体模板](../assets/megakernel-template/README.md)，已有踩坑与代码对应见 [PITFALLS.md](../assets/megakernel-template/PITFALLS.md)。

单条原语结论应能回答：谁等待谁、等几个producer、哪种pipe/核型、保证源可复用还是目的可见、buffer何时可覆盖、经过哪些压力和边界测试。记录形式例如：

```yaml
claim: <待认证的具体语义；避免笼统写“可同步”>
status: measured_or_inferred_or_unknown
scope: <芯片型号/修订、CANN与编译器版本、编译参数、设备>
probe_source_and_binary_hash: <可重建路径与hash>
participants_and_pipes: <producer/consumer核型、数量、方向、event ID>
ordering_guarantee: <源释放/目的完成/可见性/reader完成，明确区分>
coverage: <单轮/多轮、同地址/异地址、ring wrap、tail/空核、大权重L2压力>
results: <通过/总次数、错误值或超时、原始日志>
counterexamples_and_limits: <已知失败配置、未覆盖项、失效条件>
```

只有单producer、单轮或低缓存压力的成功，不能推出多producer、多轮和真实权重条件下可靠。版本改变后重新认证受影响项；缺少probe时由agent从头文件与最小实验补齐，不要求用户预知Fixpipe或编译器缺陷。

设备健康信息包括实验前后状态、占用、已知异常与健康小探针对照。设备编号是现场选择，历史“坏卡”记录和当前健康快照都不能单独证明某次错误的根因。将芯片/CANN相关认证留在本层；负载目标、模型语义和证据状态规则留在硬件无关层。

## 编译与拓扑

- `bisheng --npu-arch=dav-2201 -std=c++17 -xasc`，由 `KERNEL_TYPE_MIX_AIC_1_2` 自动识别 mix。
- 当前构建不加 `-cce-enable-mix`；旧实测加上会产生双 `_mix_aic_mix_aic` 符号并触发注册错误 107000。
- 通过编译进 `.so` 的 `run_megak` 执行 `megak<<<grid,nullptr,stream>>>`；旧手写 legacy launch 在 mix 上 hang。检查 ELF 的 `.aicore_binary` 和导出 stub。
- grid=24 对应 24 AIC + 48 AIV。组 `cid` 包含 AIC cid 与 AIV 2cid/2cid+1，sub=bid&1。用 `GetBlockIdx()`，不照搬旧底层同名函数。
- 若使用 FFTS，先有 device context 再取 `rtGetC2cCtrlAddr()`；上下文前可能得到零。不同硬件重新验证。

## placement 与参数：模型维度和硬件维度的交界

| 项目 | 当前值 / 推导 |
|---|---|
| 模型 | L=28，H=2048，II=6144，VV=151936，KVH=8，D=128 |
| 权重 GEMV | N 轴分给 24 核，K tile=256，bf16 输入、FP32 累加 |
| QKV | N=4096，pad=4224，slab=176，N tile=96+80 |
| O/Down | pad=2304，slab/tile=96；最后两个核可能无有效输出 |
| GU | N=12288，slab=512，4×128；每核 paired gate/up |
| LM | slab=6336，tile=96，packed span=8192，argmax grain=2048 |
| Attention | cid%8=KV head，cid/8=sequence slice，3 slice；2 Q heads/KV |
| 上下文容量 | SLT=256，STEP=13824，SPADG=41472，覆盖最大 live=40960 |

这些 padding、tail、slice 与核数联动，换模型/核数必须重算，尤其小词表、小 FFN 时不能让空任务核跳过必要 ack。

## 布局与资源

- B 权重为 chunk-major：`[layer][cid][nt][kc][16 kfrac][maxncc nfrac][256]`，使消费 tile 的 DMA 连续；使用 padded 宽度，尾 tile 按 maxncc 补齐。
- K cache：`[layer*KVH+kv][d_block][s_block][16s][16d]`；V cache：`[layer*KVH+kv][s_block][d_block][16d][16s]`。两者分别迎合 QK/PV 的读取方向。
- GEMV A：每 K=256 chunk 在 GM 存两条 bf16 ND 行（512 元素），MTE2 Nd2Nz 展开到 L1，row0 有效、row1 用于 landing。旧版存 16 行。
- probs GM 为 `[tile][head0 256][head1 256]`，每 tile 1KB。`Nd2NzParams` 将两行展开到 L1；`dstNzC0Stride=16` 的单位是 16 bf16 元素，不是字节。保留未填行的假设需审计，不能让它们参与有效输出。
- UB：`VECTIN/VECTOUT/VECCALC` 等 vector buffer；L1：A1/B1 类 buffer；L0：A2/B2/CO1。按 `InitBuffer` 的真实大小和 live interval 计算占用，不能把 GM 分配量当本地容量。
- GEMV B ring 深度 RN=6，每 slot 64KB，共 384KB L1。后两 slot alias attention 的 K/V L1 buffer；它们只在互斥 stage 复用，不是免费增加容量。
- `Fixpipe.dstStride` 是完整目标行 stride；LM 的跨核布局不能误填单个 span 的 stride。mSize 从早期16行缩到权重2行、score/PV3行，需保留计算行与认证行。

## 当前同步协议：ready 之前先证明数据可见

核心次序：`arm(marker) -> arm/cache操作有序 -> 实际 DMA/compute/Fixpipe -> 目的完成 -> marker 落地确认 -> seq publish -> consumer wait -> consumer load -> reader done -> reuse`。

- `aic_wait_land` 先 `FIX_S` 再以 dcci/volatile 轮询 marker。`FIX_M` 只保证 L0C 源可复用，不能作为目的 GM 对 scalar/另一平面的可见性证明。
- `S_MTE2` 让 sentinel store/cache clean 先于后续会覆盖它的工作。
- marker 必须认证消费的数据。R30 GU 对 **live row0 每个128列 tile** 放 marker；旧 dead-row marker 曾在实际 SiLU 消费仍旧值时通过。
- 活跃 marker 的重臂也会写坏旧 reader。R30 无 Down 输出的 AIC22/23 仍等待 hP chunks8..23 被确认后才能进入下一层重臂。
- AIC `aic_seq` 用 scalar store + `dcci(CACHELINE_OUT)`；AIV `seq_write` 走专门 UB + MTE3，并管理 UB source 释放。不能只写 ready word 后就假定另一个 buffer 已落地。
- workspace seq：offset4096，48 row，每 row256 round，每 slot16 int32；0..23 为 AIV、24..47 为 AIC。每层 R=8*li，final=224，PR_PROBS=232、PR_SCORES=233、PR_LM=234。
- tag 从512开始，每 host launch 加512。进度值为 tag+li*16+count，当前 grain 预算使它不跨越下一 launch 区间；更大模型/更细 grain 必须重新证明。长寿命服务须设计 int32 tag 溢出前的 quiescent reset/更宽 epoch，当前生成长度不代表无限调用安全。
- 完成 seq 用精确匹配；stream progress 用 `>=`。按使用点核对地址/值/生命周期，不能互换判断。
- carried maxima 用交替 history bank，防止覆盖前一层尚未读完的数据；分母 publication 做 GM readback。

`mega.cpp` 顶部“plane SyncAll”段落是旧注释。当前实际流程已由 sentinel/seq 替代。历史保守 barrier 配方 `SyncAll<false>(gmWs,ubWs,48)` 只用于理解早期 bring-up；不应重新塞回当前生产路径当作默认协议。

## 事件预算与 buffer 生命周期

当前 GEMV ring 的 MTE2_MTE1 / MTE1_MTE2 使用通道1,2,3,4,6,7，A-group 用5；score/PV ping-pong 使用0和1，必须先 drain ring 再复用通道。

`FetchEventID` 在该实现中是 peek，不会自动分配不同 ID。连续调用不能证明两个在途 DMA 的通道不同。

需保护的边包括：MTE2→MTE1→Mmad→Fixpipe；MTE3→UB复用；MTE2→scalar GetValue；vector→scalar 与 scalar→vector；MTE1→下次覆盖 A-group；reload→前次 vector reader 完成。`PipeBarrier` 不能代替跨引擎依赖。

早期实测 S_MTE3 通道重复使用可能 hang，mode-2 多 setter flag 在第一 setter 后即可放行，不能充当 all-producer barrier。旧 AIV→AIC cross-core mode/pipe 矩阵未得到可靠配方，当前采用 GM seq。这些是特定版本实测结果，升级后重新 probe，不能宣称所有 Ascend 都如此。

## 编译器、数值与 host 特例

- reload 循环曾被软件流水提前发射 MTE2，覆盖仍在 vector 读取的 UB。当前需要 ping-pong 和显式 reader-complete 事件。
- `Cast(f32->f32, CAST_NONE)` 旧 probe 在 dav-2201 出错；copy 用已验证路径。`tree_sum/max` 原地破坏输入，先保存原值再归约。
- RMSNorm 的 `Rsqrt` 后两步 scalar Newton，再按 Qwen 的 bf16 落点乘 gamma。一步在最大长度仍失败；不能删作“等价简化”。
- 原 CPU 96线程 cos 路径跨进程输入不一致；`model_run.make_rope_tables` 用 scalar FP64 libm sin/cos → FP32 → 复制两半，属于启动工作。
- ctypes raw launch 绕过 torch_npu host 队列；当前 production/test 在清理之后、stub 之前同步 current stream。拿到 handle 并不代表 host queue 已提交。
- Fixpipe 直接 host DMA 读回曾看到旧值。以已验证 AIV/MTE3 结果通路及直接 ACL D2H 检查定位，不把 `.cpu()` 读数天然当成设备消费真值。

## 换 NPU / CANN 的认证清单

| 必测 probe | 最低覆盖 | 产生的 adapter 配置 |
|---|---|---|
| 编译/注册/launch | 空 kernel、mix 双分支、参数逐项回传、多轮 | arch、kernel type、stub、grid |
| ND/NZ/Mmad/Fixpipe | 递增值与随机值、tail、跨核 dstStride | tile/布局/单位/舍入 |
| 跨平面可见性 | 多 producer、大权重 L2 压力、live marker | publish/wait/cache 策略 |
| events | 双缓冲、ring wrap、drain、跨 stage 复用 | 通道预算、ready/free 配对 |
| UB reload/归约 | 重复与 alternating pattern | ping-pong、barrier、Cast/reduce 规则 |
| 全核参与/空任务 | grid 变化、空 slab/slice、多层 | owner 与 ack 映射 |
| host 顺序/读回 | 排队 clear→raw launch、完整 logits/KV | 队列提交与可见性方案 |
| 近似数学 | Rsqrt/Exp/舍入边界、多层累计 | 精度修正及误差门槛 |
| 持续带宽 | HBM/L2、单向读/读写、相同访问 pattern | 流量模型与可持续 roofline |

每条保存硬件/软件版本、源码 hash、输入/输出、次数、是否超时和结论。旧 `/tmp/opencode/smoke3` 探针路径属于临时产物，不能依赖其仍存在；按问题重建 probe，原 README 的84探针记录只是历史证据。
