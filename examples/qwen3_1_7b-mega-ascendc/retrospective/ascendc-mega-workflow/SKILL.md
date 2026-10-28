---
name: ascendc-mega-workflow
description: 为 AscendC 单次 launch 的模型 decode megakernel 建立语义契约、代码框架、精度和死锁诊断及证据驱动的调优流程。适用于新模型接入和现有 megakernel 复盘。
---

# AscendC mega 工作流：硬件无关层

先确认范围：模型、batch/token 形状、精度、KV 契约、最大上下文、每步 launch 数和性能口径。不要把本案例的单 token 假设扩大到 prefill、batch 或其他模型分支。

先按生成框架的交接包清单核对已有基线、模型契约、fixture与测量。关键信息标记约束/实测/推断/未知，保留适用范围和证据。用户主要提供目标、硬约束和访问条件；agent负责从现有材料查明语义、补probe、定位首错和测瓶颈，不能要求用户预知根因，也不因清单未填满停止独立工作。

按任务加载：

- 需要实际代码时，复制 [完整可编译模板](../ascendc-910b-adapter/assets/megakernel-template/README.md)，按其 [PORTING.md](../ascendc-910b-adapter/assets/megakernel-template/PORTING.md)逐stage接入；不要只返回伪代码。
- 新模型生成与框架抽取：[references/framework.md](references/framework.md)。
- 精度错误、随机漂移、hang：[references/debugging.md](references/debugging.md)。
- 性能优化与验收：[references/tuning.md](references/tuning.md)。

执行顺序：固定 reference 和输入 → 验证工具链及原语 → 建立保守全模型基线 → 定位关键路径 → 单变量优化与消融 → 全范围精度/性能回归。若有 HIR/runtime twin，布局和语义修订必须同步，历史 check 结果不能覆盖新二进制。

本层定义依赖、所有权、可见性和数值要求，不定义核数、tile、event ID、内存容量或 cache 指令。读取当前硬件的 adapter 后才生成这些代码。当前案例使用 [910B adapter](../ascendc-910b-adapter/SKILL.md)；换型号时建立新的 adapter，不把旧平台探针结论当成通用 Ascend 事实。

交付保留：语义契约、可重建源码/配置/hash、buffer 与 event 生命周期、独立 reference、通过及失败记录、同精度 control、不同上下文性能和瓶颈证据。精度失败的候选不进入性能最优集合。
