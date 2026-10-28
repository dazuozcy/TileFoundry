# 快速生成正确 kernel：先固定契约，再填计算

## 先用可编译模板，再读抽象框架

实际模板在 [megakernel-template](../../ascendc-910b-adapter/assets/megakernel-template/README.md)，接入步骤见 [PORTING.md](../../ascendc-910b-adapter/assets/megakernel-template/PORTING.md)。默认stage包含完整R30计算，不是TODO空壳。复制后先构建并跑通，再逐stage替换新模型逻辑；同步、buffer/event生命周期与host launch已经有实际实现。

下面的ModelOps/HardwareAdapter伪代码用于说明职责；编译入口是模板的`kernel/mega.cpp`。代码分为`model/`计算与`backend/ascend910b/`物理支持，但当前各stage仍有910B placement耦合，换卡需重做适配。

## 0. 开工前最有价值的信息与交接包

优先收集能改变实现决策的材料：可运行基线、精确数值契约、可复现状态、已认证的硬件行为。已有仓库、checkpoint和日志能回答的问题由agent先查；不要要求用户重新解释源码里已经明确的内容。

| 信息 | 推荐提供形式 | 直接影响的决策 / 可避免的返工 |
|---|---|---|
| 执行目标与硬约束 | batch/token分支、最大上下文、每步launch数、精度门槛、是否允许预处理/重排/额外scratch | 决定DAG与可行方案，避免实现完成后才发现不符合验收 |
| 模型数值契约 | 原始checkpoint/config、reference、逐stage dtype/rounding；见下一节ModelSpec | 确定norm、RoPE、QK norm和MLP落点，避免数学式相同但数值不等价 |
| 可运行工具链基线 | 最小源码、构建命令、host ABI、成功日志及版本/hash | 直接复用编译/注册/launch路径，避免先在不可靠启动链上拼全模型 |
| 已正确的交接与布局 | 实际producer/consumer代码、ready/free条件、pack/unpack、stride单位、padding | 保留经过认证的同步壳，避免重写flag、可见性与buffer复用协议 |
| reference与固定fixture | 短/中/长/最大边界的token、pos、完整KV/RoPE/scratch状态，以及中间结果 | 从首个错误stage定位，避免只有最终文本时无法区分输入、计算和状态问题 |
| 工具能力边界 | 能运行的check/analyze命令、输入依赖、明确不支持的功能 | 避免在CLI参数或不可用投影上盲试 |
| 设备与原语认证 | 设备健康、空闲状态、适用版本、最小probe及覆盖；见硬件层 | 决定哪些底层语义可以复用，哪些必须先验证 |

完整交接包可以按优先级组织为：①可构建运行的正确基线；②模型契约与独立reference；③固定测试状态；④硬件认证记录；⑤同口径性能基线、timeline与字节账本；⑥成功/失败/未解决实验台账。后两项用于从有到优，详见 [tuning.md](tuning.md)；fixture要求见 [debugging.md](debugging.md)。这些是收集优先级，不是要求用户一次提交全部材料的前置门槛。

### 责任边界与证据状态

业务目标、不可变约束、验收偏好和外部权重/参考的访问条件主要由用户提供；agent负责从现有材料提取契约，执行工具链冒烟、原语probe、确定性诊断、关键路径测量和实验记录。未知的硬件缺陷或race根因不能反过来要求用户提前知道。仅在无法从环境推导且会改变方案的目标或约束缺失时澄清，其余独立工作继续推进。

每条关键信息标记状态，并附适用范围与证据路径：

```text
【约束】每token一次launch；来源：任务验收要求。
【实测】某交接协议在指定设备/CANN/编译器、源码hash与覆盖集合下通过。
【推断】MTE2可能是关键路径；来源：计数器；下一步：流量消融验证。
【未知】新模型跨层score变化范围；下一步：采样/边界构造与稳定fallback。
```

实测结论不能越过其输入、次数和软硬件版本的范围；推断不能写成adapter保证。交接时同时记录失败、未覆盖项及失效条件，避免下一位agent将历史经验无条件套用。

## 1. 本次生成链路与应采用的顺序

实际初次 bring-up 先做 GEMV/mix 探针和裸 C++，后补 HIR/twin；`session.md` 明确承认这偏离了原任务要求。后续复用应把语义程序和验证接口提前，避免实现做完却无法接入门禁。

原任务已经提供硬件、工具链、API来源和验收要求。先采用未验证的同步假设、后补HIR与未尽早严格重复测试，是agent的执行顺序问题；更完整的交接包可以加速工作，但不能替代这些基本验证。

推荐链路：checkpoint/config → ModelSpec + 独立 reference → 单个完整 decode 的 HIR/阶段 DAG → 硬件 placement/layout/资源计划 → AscendC C++ + host ABI → 构建与原语探针 → 局部/全模型验证 → 性能调优。这里的“生成”包含 agent 编写和专用 packer，不是本目录已有一个通用模型到 C++ 编译器。

`model.py` 注释称由 `gen_model.py` 生成，但当前目录没有该生成器。不能把它当作现成可运行工具。其 HIR 是语义依据；其中 placement 和部分算法描述也有历史性差异，需与当前实现重新核对。

## 2. ModelSpec：换模型先补这张表

| 项目 | 要明确的契约 | 本案例 |
|---|---|---|
| 形状 | L/H/FFN/vocab、Q/KV heads、head dim、最大 context | 28/2048/6144/151936，16/8，128，40960 |
| 运算顺序 | pre/post norm、bias、残差位置、激活函数、QK norm | pre RMSNorm，QK RMSNorm，RoPE，SiLU(gate)*up |
| 精度落点 | 每次 cast、round mode、累加精度、norm scale 前后舍入 | bf16 点积、f32 累加；RMS 归一化先落 bf16 再乘 gamma |
| 位置编码 | theta/scaling、rotate-half 或相邻配对、生成 dtype | theta=1e6，rotate-half，确定性 FP32 表 |
| KV 状态 | 旧前缀、新 token 位置、尾部、生命周期、GQA 映射 | append 到 pos，只读有效前缀，padding 为零 |
| 输出 | logits 的词表顺序、padding mask、argmax tie、KV 更新 | 排除 padding vocab，返回下一 token 和更新 KV |
| 执行 | 每 token launch 数、允许的 host 工作、最大边界 | 单 MIX launch；host 仍有清理、同步、读回 |

不能只替换 H/L。比如 QK norm、RMS 舍入顺序、RoPE pairing、FFN 激活变了，原有“看起来合理”的计算也可能完全不符合新模型。

## 3. 抽象框架：ModelOps / Schedule / HardwareAdapter

以下是**设计伪代码，不是可编译 AscendC API**。硬件层负责给抽象方法实现；已有实现在硬件文档中逐个映射。这样 agent 添加计算时不必重写已经认证的交接协议。

```cpp
struct ModelOps {
  // 新模型填写：运算、dtype/舍入、参数、逻辑 shape。
  norm_input(); project_qkv(); qk_norm_rope_append();
  attention_scores(); softmax_partial(); attention_pv(); merge_slices();
  project_o_residual(); norm_mlp(); gate_up(); activate_mul();
  project_down_residual(); final_norm(); lm_head(); argmax();
};
struct HardwareAdapter {
  allocate_local_buffers();        // UB/L1/L0 的空间、对齐与 alias 生命周期
  load_weight_tile(); cube_dot(); vector_op(); store_result();
  wait_input(channel, epoch, grain);
  publish_visible_output(channel, epoch, grain); // 先证明数据落地，再发布
  wait_buffer_readers_done(buffer);             // 才可覆盖/重臂
  drain_events_and_dma();
};
// 单个 __global__ __aicore__ 入口内，两类核遵循同一 DAG：
mega_step(args, pos, launch_epoch) {
  hardware.init();
  if (is_cube_core()) {
    for (layer : layers) {
      // QKV -> QK -> PV -> O -> GU -> Down
      // 每项：预取独立权重；等所需输入 grain；计算；落地确认；发布。
      run_cube_schedule(layer);
    }
    run_lm_schedule();
  } else {
    for (layer : layers) {
      // RMS1 -> QKnorm/RoPE/KV -> softmax -> combine/residual
      // -> RMS2 -> SiLU -> residual；每个 buffer 有唯一写 owner。
      run_vector_schedule(layer);
    }
    final_norm_and_argmax();
  }
  hardware.drain();
}
```

stage 边界是依赖交接，不是 return 或额外 kernel launch。数学 DAG 可以保持稳定，物理全局屏障可逐步细化为分块 producer/consumer 交接。

## 4. DAG 上每条边都填“交接契约”

```text
edge: GU -> SiLU
logical tensor / dtype: [gate, up], FP32（消费时按模型舍入）
physical layout: adapter 定义，与 host packer 同一版本
writer / reader: 哪个核写哪个 tile，哪些核读取
ready condition: 所有将被读取的 live 行已落地
epoch: launch + layer + stage + grain，不与旧 token 混淆
reuse condition: 所有旧 reader 完成后，writer 才能 arm/覆盖
tail: 空 slab / padding 核仍须执行哪些 ack
```

每个 buffer 同时记下：逻辑 shape、物理 shape、dtype、stride 单位、padding、GM/UB/L1/L0、对齐、owner、alias 对象与 live interval、ready/free event。用小矩阵递增值对 pack/unpack、NZ 索引、stride 做精确对拍，再换随机数。

## 5. 先拼保守骨架

先验证一次 launch ABI（ctypes 指针数/整数顺序/stream），再验证一核 GEMV、ND/NZ、Mmad+Fixpipe、双平面交接；不能把未验证原语直接拼成 28 层。保守实现保留中间 GM、lstop、单层/分阶段 dump，确认一次完整 step 后再减少往返。

保留三个互相独立的验证对象：原始 checkpoint 的数学 reference；保守同精度实现；优化实现。只比较两个共享 packer 的实现会漏掉同源布局错误。

host 模板：创建 device context → 加载库并读取 layout metadata → 按匹配布局打包 → 预分配 KV/scratch → 定义初始状态 → 排队清理 → 确认 host 队列提交且 stream 有序 → 分配新 epoch → stub launch → 按验证过的路径读回。

模型参数变化后重新推导 padding、head/slice placement、词表分片、tag 预算和空工作核路径。运行时最好导出布局版本及编译参数；本案例 `megak_layout()` 已比环境变量猜测可靠，但还不是通用的完整 ABI 版本协议。

## 6. 可复用算法的边界

稳定分片 softmax 令 `m_s=max(score_s)`、`l_s=sum(exp(score_s-m_s))`、`a_s=sum(exp(score_s-m_s)*V_s)`，合并时用 `m=max(m_s)`，输出 `sum(exp(m_s-m)*a_s)/sum(exp(m_s-m)*l_s)`。

若多分片共用一个 shift，才可直接合并 numerator/denominator。当前 R27+ 用前一层最大值作为共同 shift，不能据此保证任意新模型的下一层 score 都在安全范围。统一加常数的 shift-invariance 测试也不能证明跨层任意幅值突变安全。换模型先用稳定 max/LSE 或有界回退，再根据实际 score 范围评估单遍版本。

先实现 shape 边界、mask 和稳定数值，再接流水。不要把当前模型“小分数范围”当成硬件无关的不变量。
