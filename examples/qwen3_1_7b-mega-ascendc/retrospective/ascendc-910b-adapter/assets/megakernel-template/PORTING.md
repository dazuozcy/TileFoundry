# Agent 如何把模板移植为另一个模型

## 接入前填写交接清单

按以下表记录已有信息及其证据路径；未知项明确填“未知”和下一项验证，不要求用户提前完成整套平台调查。agent先查当前代码、checkpoint和日志，只对无法推导且会影响方案的业务目标/约束澄清。

| 项目 | 接入时填写的值/路径 | 主要获取方式 |
|---|---|---|
| 目标与硬约束 | batch/token、context分布与最大值、launch数、精度/greedy、计时口径、可改布局/预处理/scratch预算 | 用户目标与任务要求，agent核对 |
| 模型契约 | 原始checkpoint/config/hash、dtype/舍入、norm、RoPE、激活、KV和权重方向 | 已有reference与配置；agent提取并验证 |
| 可运行基线 | 源码/库hash、构建/运行命令、ABI、成功日志、依赖与cwd | 先运行此模板或既有正确实现 |
| 固定fixture与reference | token/pos/lstop、完整KV/RoPE/scratch、shape/layout/hash、参考中间值与阈值 | 复用已有状态；缺失时agent生成与独立校验 |
| 硬件认证 | 设备/CANN/编译器版本、健康/占用、交接代码、原语probe和覆盖/失败记录 | 复用匹配版本记录，agent补测未知项 |
| 性能证据 | 同精度baseline、关键路径timeline、HBM/L2字节账本、同访问模式带宽 | agent测量；用户指定实际负载优先级 |
| 实验台账 | 已尝试变体、配置/版本、精度、各context性能、撤回原因、未解决项 | 已有日志及后续每轮实验维护 |

每项附 `【约束】/【实测】/【推断】/【未知】` 状态。实测附版本、输入、次数与证据；推断写可证伪实验；未知不默认为可用。不要将旧版成功记录当作新模型或新硬件验收。

失败时先保存完整现场、首个错误stage和同进程/跨进程重复结果；仅有最终文本或seed不够重放。优化前先分开kernel与host时间、识别未隐藏关键路径，再依据可消除字节或等待的收益上限选择修改。缺少完整交接包时从能独立验证的部分继续工作。

## 先复制一个能运行的基线

复制整个模板，运行README的audit、build、0/1、0/28门禁并保存源码/库hash。不要先清空所有stage再开始填模型；逐项替换，可以让每次失败只对应一个变化。

默认两个执行骨架已经写好：

```cpp
// kernel/mega.cpp：同一次launch的两个核型分支
if ASCEND_IS_AIC {
#include "model/cube_schedule.inc"
} else {
#include "model/vector_schedule.inc"
}
```

```cpp
// cube_schedule.inc：resources之后，for层内的真正计算文件
#include "model/stages/cube/qkv.inc"
#include "model/stages/cube/attention_qk_pv.inc"
#include "model/stages/cube/output_projection.inc"
#include "model/stages/cube/gate_up.inc"
#include "model/stages/cube/down.inc"
// for层结束之后：lm_head.inc
```

Vector对应顺序为input_norm → qk_norm_rope_kv → softmax_combine → residual_mlp_norm → activation → residual，最后final_norm_argmax。保留一个入口，不把每个include变为host算子launch。

## 第一步：填写新模型的逻辑契约

记录L/H/FFN/vocab、QH/KVH/D、max_position、权重名称和转置方向、QK norm有无、norm epsilon/落点、RoPE pairing/scaling/base、bias、有无tied embedding、激活与残差。用原始checkpoint实现独立reference。

本模板逻辑常量入口是 `kernel/model/model_config.hpp`。host和随机reference目前也分别在 `model_run.py`、`test_mega.py` 顶部定义常量；它们尚未自动由单一schema生成，须同步修改。`tools/check_contract.py` 在编译前检查18个重复字面量的一致性；表达式生成或schema变化后要同步更新checker，不能把检查通过当作所有stride正确的证明。`EPS` 仍在layout.hpp，因提取时保持原token顺序而未迁移，数值语义变化时也要修改它。

## 第二步：按形状重新设计物理布局

这一步属于硬件相关层，不能省略。改 `backend/ascend910b/layout.hpp`、resources和host buffer分配：

| 耦合 | 默认代码中的实例 | 新模型要完成的动作 |
|---|---|---|
| GQA比率 | 两Q头/KV；`q0/q1`、head0/head1、多处128/256 | 比率改变须扩展head映射、softmax、PV与combine |
| head维 | `rope_apply`固定128/64，KV 8个d-block | D变更重算RoPE、cache pack、LoadData、UB与L0容量 |
| QKV切片 | H=2048、NQKV/2、3*NQKV/4、slab176 | 依据QH*D/KVH*D推导边界和producer覆盖 |
| H和FFN | A-group固定8个K=256，Down3组，SiLU每核256 | 更改K-group数量、尾组mask、publish与空核ack |
| 输出分片 | O/Down只等0..21，22/23空任务 | 重算活跃producer集合，保留空核reader完成路径 |
| attention切片 | NB=24=KVH8×NSL3；slice跨度256对齐 | 新KVH/核数重做head×slice分配与leadership |
| 上下文 | STEP13824、SPADG41472、NTM54 | 最大长度、padding、KV容量和tile尾界一起重算 |
| 词表 | SPAN6336、SPANP8192、NCH4、LM tile96 | 重算packing/padding/argmax grain及stream进度预算 |
| epoch空间 | 8*28+final224、grain round232..234、step512 | L或grain变更后分配不重叠channel/round与tag跨度 |
| warm窗口 | sl<4096、GU slab硬编码32个chunk | 初始可以禁用warm并drain已有事件，再测量选择 |

现有 `gemv_ring` 假定K按256分块，A按8块成组，NC为16倍数且不超过ring slot容量。新维度若有尾部，先明确pad/有效长度，不让整数除法悄悄截断。

建议先保留现有24核与基础搬运协议，只改必要placement。无法复用的head映射/算法要重写相应stage，不勉强套默认值。

## 第三步：逐stage替换计算，保留交接框架

### 例1：修改pointwise计算，不改变DAG

假设新模型MLP定义为 `h=0.5*silu(gate)*up`。修改 `stages/vector/activation.inc`，在已有两个Mul之后、bf16 Cast之前增加：

```cpp
Sigmoid(ex0, ggu, sig, 256);
Mul(ggu, ggu, ex0, 256);
Mul(ggu, ggu, uup, 256);
Muls(ggu, ggu, 0.5f, 256);     // 新模型的具体逻辑：在模型指定的落点前缩放
Cast(hb[0], ggu, RoundMode::CAST_RINT, 256);
```

保留上面的 `aic_gate_poll1`、MTE2_V和下面的bcast_pack、V_MTE3、DataCopy、MTE3_V、R+7 seq发布。它们是计算前后已认证的交接壳，不是可删的样板。

同步改 `test_mega.py:ref_run`：

```python
h = (0.5 * g.float() * torch.sigmoid(g.float()) * u.float()).to(torch.bfloat16)
```

本模板附有可实际应用的 [examples/scaled_mlp.patch](examples/scaled_mlp.patch)，可在独立副本根目录用 `patch -p1 < examples/scaled_mlp.patch` 应用；不要在未保存的基线上直接套用。若reference在gate里额外独立重建h/Down/MLP，也必须同改，不能只改ref_run一个输出。重新build并跑0/1、300/2、0/28，再跑真实独立reference。这个例子用于说明替换点，不声称模板默认已采用缩放。

### 例2：新增投影的bias

在消费者取得投影的live row、完成MTE2_V后，才把bias加载到**独立UB**并加到计算输入。增加权重指针/ABI、host分配、GM view、UB容量与reference；保留原producer落地和reader重用保护。不能借用仍被异步DMA使用的pack/sig scratch。

bias放在AIC累加后还是AIV消费前必须符合模型舍入契约。只向kernel加一个Add，而不更新host指针表和reference，会让模板“能编译”却不能正确运行。

### 例3：完全替换attention或增加跨平面stage

先画新DAG，再在 `EDGE_CONTRACTS.md` 分配producer/consumer、round/grain、ready条件、buffer和reader完成条件。修改两条schedule、stage、resources、packing、launch ABI及reference。不要直接套一个看起来空闲的event ID；ring和attention的同向通道会跨stage复用。

新增结果必须按 `compute -> destination complete -> actual live data landing -> ready publication -> consumer wait/load -> reader done` 路径闭环。把大stage拆成grains时，每grain都需要这个证明，不能靠全stage末尾的flag来认证前面的早读。

## 第四步：host/packer/reference同步修改

| 文件/位置 | 接入责任 |
|---|---|
| `model_run.py:_load_raw/Runtime.__init__` | checkpoint命名、转置、QKV/GU逻辑拼接、tied/untied LM |
| `pack_w_chunks` | kernel消费次序、tail pad、每核slab、N/K tile |
| `make_rope_tables` | 新位置编码数学和确定性建表 |
| `Runtime` KV/scratch分配 | 容量、补零、不重叠、初值、lifetimes |
| `Runtime.ptrs/pargs` | 与launch.inc和入口完全同顺序；保持tensor存活 |
| `Runtime.step` | pos范围、padding mask、epoch、队列同步、读回 |
| `test_mega.py` | 随机原始权重、逻辑reference、pack/unpack、中间量对应 |
| `kernel/launch.inc` | ctypes ABI与metadata；任何新增参数两侧同时修改 |

`megak_layout()`目前仅编码tile/compact/GU/stream四项。若布局增加维度、参数或版本，扩展metadata或新增ABI version，并让host在分配/launch前拒绝不匹配。

## 第五步：优先稳定正确，再启用特化

1. norm/激活/RoPE与每个投影，先单层验证。
2. `lstop`从1到2到L，pos从0到tile±1/非整slice/长KV/最大位置。
3. strict同输入重复、alternating token、独立进程，对完整logits/KV做比较。
4. 独立原始权重CPU reference，同时测operator-local和全轨迹误差。
5. 完整自然前缀与多语言真实生成，再做同精度A/B。

默认softmax沿用R27/R30的carried共同shift。换模型先认证跨层score范围；不能证明时改稳定max/LSE并加回退，随后修改所有partial/combine reference。当前shared-shift的plain-sum combine不能直接消费不同slice各自max归一化的partial。

改完后保留修改后的逻辑契约、交接表和新validation结果；不要把本模板的Qwen PASS记录改名冒充新模型认证。

## 保存真实token对照（独立进程）

```bash
ASCEND_RT_VISIBLE_DEVICES=6 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
  python tools/real_probe.py --checkpoint /path/to/checkpoint --output candidate.json
# 可选：用另一个有model_run.py的control目录，单独进程记录
ASCEND_RT_VISIBLE_DEVICES=6 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
  python tools/real_probe.py --checkpoint /path/to/checkpoint \
  --module-root /path/to/control-project --output control.json
```

两份JSON的tokens应按既定greedy契约比较；这项短生成对照不能代替独立CPU精度或自然长前缀测试。
