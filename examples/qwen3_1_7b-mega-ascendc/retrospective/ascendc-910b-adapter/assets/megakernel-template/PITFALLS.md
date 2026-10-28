# 第一版踩坑怎样固化到模板

下表覆盖原项目README的bring-up记录，并补入R30修复。模板保留的是最后有效的实现，不重新引入早期仅靠额外barrier延迟“压住”的方案。

| 坑/历史误判 | 模板中的具体防线 | 后续agent改代码时的规则 / 验证 |
|---|---|---|
| `-cce-enable-mix`造成双mix后缀、107000 | `kernel/build.sh`自动识别MIX，检查ELF/双后缀 | 保留编译路径；换编译器重跑注册/launch probe |
| 手写legacy `rtKernelLaunch`使mix hang | `kernel/launch.inc`内 `megak<<<...>>>` | host只调用导出stub，不重写注册/launch链 |
| context前FFTS地址为0 | host先建NPU tensor，再取rtGetC2cCtrlAddr | 设备初始化顺序保留，检查返回码 |
| 参数错位/host原型错误 | 入口、launch.inc、ctypes都保留同一36指针+4int+stream ABI | 添加权重或scratch时三处一起改；restype明确None |
| `constexpr`设备辅助计算曾错误 | `protocol.hpp:SEQ_WANT`用 `__aicore__ inline` | offset带模型边界测试，不能随意改成host-only helper |
| 底层`get_block_idx()`值错误 | 双resources使用`GetBlockIdx()` | 核型/组/sub映射先probe |
| `ld_dev`误当GM load导致异常 | `cube.hpp`等用已认证volatile GM+dcci | 不按名字猜低层指令语义 |
| mode2多setter被当all-producer gate | 当前全部跨平面交接走seq/progress | wait集合明确列出，不能换成一次CrossCoreWaitFlag |
| AIV→AIC cross-core矩阵全部失败 | `vector_io.hpp:seq_write`，`cube.hpp:poll_seq` | 沿用独立round地址及递增tag；升级后单独认证 |
| poll重用旧地址/旧值挂死 | `protocol.hpp`专门round，host每launch+512 | 不将所有stage压到单个反复改写地址 |
| S_MTE3重复ID hang | QK/RoPE/KV stage中已验证的append事件使用方式 | 改循环次数或ID池必须重建多轮probe |
| SyncAll默认isAIVOnly在mix hang | 当前生产模板无全局SyncAll调用 | 若新建保守probe，按平台验证participant与`<false>`；不要复制旧顶部注释 |
| FIX_M后数据未落地L2 | `cube.hpp:aic_wait_land`先FIX_S再查marker | 目的完成与源L0C释放分开；不能只等FIX_M就seq |
| 整cache flush不能解决FIX写落地 | marker验证真实数据，而非依赖flush耗时 | cache操作只按具体方向/访问者使用 |
| 双SyncAll在大权重下才能稳定 | sentinel+seq替代时间延迟 | 不把多加屏障当永久可见性证明 |
| dead-row marker没有认证消费live行 | GU每128列live marker | 改mSize/tile时重新核对每个被读tile |
| 新层arm覆盖旧reader | Down空核仍等SiLU后续组ack | 空工作核也有控制义务，不能early return |
| ccec提前MTE2覆盖UB | 各vector阶段ping-pong+V_MTE2等reader完成 | 不仅要双buffer，还要准确free边；PipeBarrier不足 |
| `Cast(f32,f32,CAST_NONE)`损坏 | QK stage直接DataCopy各f32区；没有该同型Cast | 同型复制用验证过的copy，非必要不Cast |
| tree_sum/max原地破坏数据 | `vector_core.hpp`明确in-place；softmax先存probs再归约 | 新reference/dump也不能复用被reduce破坏的原值 |
| Fixpipe dstStride误用单slab宽度 | GEMV用outStride；LM传NB*SPANP | 检查row1之后是否覆盖其他核，不能只验row0 |
| NZ索引用逻辑宽度丢padding | pack_w_chunks与device的chunk-major布局配对 | 递增矩阵测试所有核/tail，不能只比较首核 |
| A-side广播API猜错、stride单位错 | bcast_pack/strided_rows/Nd2Nz保留认证参数 | Copy stride单位32B块，Nd2Nz参数各自核对头文件 |
| GFLOPs高却优化错方向 | ring在等待中预取B；compact A/probs减少GM字节 | 增加ring深度前先看带宽、容量、event生命周期 |
| host DMA读Fixpipe旧DDR | argmax由AIV写part_max/part_idx供host | 检查设备实际读取与host读回，不能把一次.cpu当唯一真值 |
| host队列未提交就ctypes launch | step/launch在清理后同步current stream | raw handle不能保证torch_npu host队列顺序 |
| Rsqrt近似误差28层累计 | `operators.hpp:rms_norm`两步Newton | 不删修正；换模型同时测local与independent轨迹 |
| CPU RoPE cos跨进程不确定 | make_rope_tables用FP64标量libm落FP32 | 检查全表hash/两半一致/单位圆，而非只看frequency |
| carried max被下一层覆盖 | softmax用双history bank | 历史“整层slack所以不用同步”不能替代reader生命周期 |
| FetchEventID只是peek，双缓冲同ID | cube_resources显式0/1，ring独立通道并drain | outstanding同方向必须不同ID |
| denominator早于真实落地被combine读 | softmax发布readback；combine先Acc再读part_l | 改grain/顺序时检查所有伴随元数据 |
| 删除softmax pass暴露隐藏race | 不依赖计算耗时维持同步；交接表显式依赖 | 对每条被缩短的窗口做固定输入压力测试 |
| 固定shift0仅对当前模型碰巧安全 | 当前共同carried shift；PORTING要求新模型稳定方案 | 数学上shift可约去不代表有限精度任意shift安全 |
| checkpoint和随机packer共同写错 | 保留逻辑ref/pack-unpack对照，要求独立原始权重CPU reference | 共享同一bug的两个实现相同不等于正确 |
| 比较attention用错尺度/partial定义 | test_mega区分scale、未归一化LSE partial、逻辑head | 改softmax算法时更新所有中间量重建，不只logits |
| argmax忘记SPAN-global index | stage与host按cid*SPAN+ci；mask padding vocab | 不再给ci重复加CH偏移；测最后词表span |
| KV append行号/最后一层ring偏移错 | stage中sl//16、sl%16与li*KVH区间保留 | 检查新行、旧前缀不变、边界tile及全部KV |
| near-tie掩盖错误 | 随机gate原规则保留，另有strict repeat | 真实模型契约独立检查；-1e30或NaN不能解释成near-tie |
| probe自身越界/participant错误 | 不额外写未验证probe；用当前工作实现作基线 | 每个新probe检查分配、offset和参与者，仍须超时 |
| 设备故障被误判为候选之外问题 | host保留旧故障卡guard；测试记录实际设备 | 实验前后健康对照；不自动reset其他人的设备 |
| tag溢出/pos越界不报错而hang | host在launch前显式检查范围和int32 epoch预算 | 长寿命服务quiesce后重建，不能让tag回绕 |

这些规则按当前版本成立。换NPU/CANN时，把对应行转换成最小probe验收项，而不是把910B经验推广成所有芯片的规定。
