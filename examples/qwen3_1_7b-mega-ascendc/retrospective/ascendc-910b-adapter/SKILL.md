---
name: ascendc-910b-adapter
description: 将单 token AscendC megakernel 适配到本案例的 Atlas 910B、dav-2201、CANN 8.5，复用经过验证的布局和同步实现，并在换芯片或工具链时重新认证。
---

# 910B adapter：硬件相关层

使用前先读 [硬件无关工作流](../ascendc-mega-workflow/SKILL.md)，仅把本技能用于硬件映射和该目录实验事实。

- 直接复制代码时用 [assets/megakernel-template](assets/megakernel-template/README.md)，内含实际R30 stage、构建/host/reference及踩坑防线。先跑默认门禁，再按PORTING逐stage替换，不能只改config。
- 生成/修改 kernel 时读 [references/hardware.md](references/hardware.md)，核对 buffer、event、landing 和 epoch。
- 判断为什么当前版本如此安排时读 [references/history.md](references/history.md)。历史失败不是推荐实现。
- 构建、验证和测量时读 [references/reproduction.md](references/reproduction.md)。调试脚本移动后的路径需处理。

参数适用范围是本目录的 910B/dav-2201/CANN 8.5 及对应编译器，不是所有 Ascend。换 NPU 或工具链先建立最小原语 probe 结果表，通过后再跑模型；核数、容量、地址对齐、event 通道、Fixpipe 可见性和读回路径都需要重新认证。

交接硬件经验时提供可运行构建/launch包和原语认证记录，明确producer/consumer、源释放/目的可见性、版本、压力/多轮覆盖及反例。单轮低压力成功不推广为通用保证；缺失认证由agent补测。记录格式见hardware.md。

保留当前 R30 的精度/同步修复作为 control，仅用 compact-A/GU-interleave/LM-stream 等开关做性能消融。不要直接拿 R28/R29 或 README 的早期版本作同精度正确 control。设备编号按当时健康与占用情况选择，文档中的 0/4/6/7 都不是固定要求。
