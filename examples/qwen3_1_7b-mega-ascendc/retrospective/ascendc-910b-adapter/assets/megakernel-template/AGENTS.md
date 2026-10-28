# 使用此模板接入新模型

本目录是可独立复制的R30 AscendC生产模板。先读README、PORTING、EDGE_CONTRACTS和PITFALLS，再修改相应文件。

- 先构建默认版本并通过随机门禁，留存hash；在有通过基线的基础上逐stage替换，不清空全部实现。
- 新模型计算入口在kernel/model/stages及operators；纯修改计算时保留已认证的读前wait、DMA完成、landing、publish和reader完成顺序。
- 形状变化同时改model_config、物理layout/resources、host allocation/packer、随机reference。当前还有明确的Qwen专用常量，不能只改L/H/D。
- 换硬件或改变流水/通道允许修改backend，但必须重新认证相应原语、buffer/event生命周期及跨核可见性。
- 保持一个MIX入口和<<<>>> host stub；不把stage变成多次host launch，不引入旧版双SyncAll方案。
- 新增ABI参数同步修改入口、launch.inc、ctypes argtypes/ptrs并验证；权重和scratch必须在异步执行期间保持存活。
- token/position/epoch检查、确定性RoPE、两步Rsqrt、live-GU marker和空核ack都是已有正确性修复，调整时提供替代证明和回归结果。
- 保存失败实验；随机gate、同输入bitwise重复、独立CPU精度和自然长前缀是不同证据，不用其中一项替代另一项。
- validation/results.json只描述初始Qwen基线，移植后添加新结果并清楚标记模型/硬件/配置，不冒用原结果。
