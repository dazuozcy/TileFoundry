# AscendC megakernel 生成与调优复盘

本知识库以当前目录的 **Qwen3-1.7B、batch=1、单 token decode、单次 MIX kernel launch** 为案例，供 agent 更换模型或 Ascend NPU 时复用。复盘日期：2026-10-28。复盘中的历史性能来自已有记录；后续补充的可编译模板已实际构建并运行NPU门禁，范围见模板validation记录。原生产代码未修改。

## 可直接复制的代码模板

第1部分现已补为实体代码：[megakernel-template/README.md](ascendc-910b-adapter/assets/megakernel-template/README.md)。默认完整运行R30 Qwen3，拆出6个Cube stage、7个Vector stage、backend/资源/同步、host打包和reference；无需原项目源码即可编译。

- [新模型逐步接入与改动示例](ascendc-910b-adapter/assets/megakernel-template/PORTING.md)
- [每条跨stage交接契约](ascendc-910b-adapter/assets/megakernel-template/EDGE_CONTRACTS.md)
- [README踩坑到模板代码的逐项映射](ascendc-910b-adapter/assets/megakernel-template/PITFALLS.md)
- [本次实际编译、设备验证结果](ascendc-910b-adapter/assets/megakernel-template/validation/results.json)

## 两层入口

| 层 | 技能入口 | 内容 | 更换时的处理 |
|---|---|---|---|
| 硬件无关 | [ascendc-mega-workflow/SKILL.md](ascendc-mega-workflow/SKILL.md) | 模型语义契约、生成框架、精度与死锁诊断、优化决策方法 | 换 Ascend 型号继续使用；换模型填写新语义契约 |
| 硬件相关 | [ascendc-910b-adapter/SKILL.md](ascendc-910b-adapter/SKILL.md) | 910B/dav-2201/CANN 8.5 的布局、事件、缓存可见性、构建与测量 | 换芯片、CANN 或编译器后按探针重新认证 |

建议阅读顺序：

1. [生成框架与新模型接入](ascendc-mega-workflow/references/framework.md)。
2. [精度与卡死诊断](ascendc-mega-workflow/references/debugging.md)。
3. [可复用调优流程](ascendc-mega-workflow/references/tuning.md)。
4. [910B 实现契约及换卡清单](ascendc-910b-adapter/references/hardware.md)。
5. [从初版到 R30 的实验复盘](ascendc-910b-adapter/references/history.md)。
6. [证据索引与复现](ascendc-910b-adapter/references/reproduction.md)。

## 提前准备什么能少走弯路

信息按工作阶段补在对应章节：

| 场景 | 对应章节 | 补充内容 |
|---|---|---|
| 从无到有 | [生成框架](ascendc-mega-workflow/references/framework.md) | 高价值输入、六项交接包优先级、用户与agent责任、证据状态 |
| 精度/卡死定位 | [故障诊断](ascendc-mega-workflow/references/debugging.md) | 可重放失败fixture、首错/重复性矩阵、消费时刻与最终dump的区别 |
| 从有到优 | [调优方法](ascendc-mega-workflow/references/tuning.md) | 目标分布/约束、同口径baseline、关键路径、字节/带宽与失败台账 |
| 换硬件或工具链 | [硬件契约](ascendc-910b-adapter/references/hardware.md) | 最小构建/launch包、原语认证格式、适用范围与反例 |
| 复盘原过程 | [实验历史](ascendc-910b-adapter/references/history.md) | 各阶段哪些提前信息能改变路线，哪些返工属于执行顺序问题 |
| 复制模板接新模型 | [移植文档](ascendc-910b-adapter/assets/megakernel-template/PORTING.md) | 可直接填写的接入交接清单 |

重点是可运行代码、可复现状态和带覆盖范围的证据；用户不必提前知道平台缺陷或race根因。未验证结论标为推断/未知，由agent通过最小实验补齐。

## 当前已证实的终点

源码 `kernel/mega.cpp` 与 `kernel/mega_r30src.cpp` 的 SHA256 均为
`a6bc5c2b884b7a2b15feef96e36a82cb54e1b2c2a0d4d88aeba9c343934b3737`。
R30 最终记录在相同精度和同步方式下比较 control 与 release：

| 测量口径 | ctx=1 | ctx=40000 |
|---|---:|---:|
| release 设备 kernel 均值 | 2936.361 us | 6265.543 us |
| 由设备时间换算 tok/s（不含 host） | 340.6 | 159.6 |
| release launcher wall time 中位数 | 2.97551 ms | 6.29530 ms |
| 对同轮 control 的设备延迟降低 | 5.44% | 2.72% |

证据：[最终 Roofline 汇总](../profiling/r30/final-roofline-summary.json)、[最终精度汇总](../profiling/r30/final-accuracy-summary.json)、[完整优化日志](../OPTLOG.md)。40k 自然前缀后生成、最大 live context=40960、全 logits/KV 跨进程一致性与独立 CPU 精度均有记录；bitwise 一致指 control/repeat/release 之间，不指与 CPU 浮点计算 bitwise 相同。

“极致性能”在这里应理解为有字节模型和实验支持的局部最优，不能承诺换模型后机械套用就达到全局最优。最终 R30 没有重新提供所有整数上下文长度的性能表；不能用早期 README 的全表拼成 R30 的全范围测量。

## 阅读历史记录时的四个纠偏

- README 的 8.65/15.04 ms 是早期双屏障版本，不是当前性能。
- 初版“所有交接都用双 SyncAll”的经验已被 R1 后的 sentinel/seq 协议取代。`mega.cpp` 顶部部分注释也滞后，以实际函数和 R30 修复为准。
- R28/R29 的随机门禁通过不等于长上下文精度闭环；R30 才补齐输入确定性、数值精度和同步问题。
- 初次复盘时 `runtime_model.py` 仍使用早期布局，历史check不能认证新库。2026-10-28已为当前R31修复chunk-major/paired GU、256-position K tile、workspace及launch同步，并恢复CLI check；当前入口为项目根的 `bash check.sh`，报告在 `profiling/check-r31/`。本知识库的实体模板仍是R30基线，不能把R31结果直接移作模板的新验收。
