# 可直接复制、编译、验证的 AscendC megakernel 模板

这是从 Qwen3-1.7B 最终 R30 **按源代码原样拆分**的工作模板，不是伪代码，也不是空 stage。默认包含完整28层计算、真实权重运行器和随机权重逐算子验证。代码不引用原项目目录；复制整个文件夹后可独立构建。

开始前读 [PORTING.md](PORTING.md)；每条数据交接见 [EDGE_CONTRACTS.md](EDGE_CONTRACTS.md)，第一版踩坑对应代码见 [PITFALLS.md](PITFALLS.md)。[validation/results.json](validation/results.json) 记录本次实际验证范围。

## 复制与冒烟

在原项目根目录执行，目标目录必须是新路径：

```bash
cp -a retrospective/ascendc-910b-adapter/assets/megakernel-template /tmp/my-model-mega
cd /tmp/my-model-mega
python tools/check_contract.py
python tools/audit.py --baseline
bash kernel/build.sh
# 下例设备6仅为本次验证设备；使用前确认健康、空闲。
ASCEND_RT_VISIBLE_DEVICES=6 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
  MEGA_REPEAT_CHECK=8 timeout 300s python test_mega.py 0 1
ASCEND_RT_VISIBLE_DEVICES=6 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
  MEGA_REPEAT_CHECK=8 timeout 300s python test_mega.py 0 28
```

继续验证已有历史KV和最大位置：`python test_mega.py 300 2`、`python test_mega.py 39999 28`、`python test_mega.py 40959 28`。`sl` 是当前token位置（live context=sl+1），`lstop` 是执行层数。长门禁的reference/打包可能很慢，按实际机器调整进程超时；超时不直接等价于kernel死锁。

使用已有 Qwen3 checkpoint 验证真实生成：

```bash
MEGA_CHECKPOINT=/path/to/qwen3_1_7b ASCEND_RT_VISIBLE_DEVICES=6 \
  OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
  python run.py --prompt 'The capital of France is' --max-new-tokens 32
```

随机门禁不需要checkpoint。软件依赖沿用已有CANN8.5/bisheng/torch_npu环境，不安装或升级包。host保留历史设备0–3禁止使用的guard；换机器先根据设备诊断调整这条局部策略，不把该编号规则当作Ascend特性。

实际验证：默认库与原R30发布库SHA256相同；0/1、0/28、300/2、39999/28、40959/28门禁及每项8次严格重复通过。独立副本中的MLP缩放改动也编译并通过0/28+8次重复；仓库外副本真实权重生成32个token与原版逐ID一致。详见validation记录与日志，本次不额外声明CPU长前缀或性能重新认证。

## 文件布局

```text
kernel/mega.cpp                         单设备入口，组织双平面单launch
kernel/launch.inc                       <<<>>> stub + layout metadata
kernel/model/model_config.hpp           模型逻辑维度
kernel/model/operators.hpp              RMS/RoPE数值语义
kernel/model/{cube,vector}_schedule.inc  层循环与计算stage装配
kernel/model/stages/cube/*.inc           QKV/QK+PV/O/GU/Down/LM
kernel/model/stages/vector/*.inc         norm/RoPE/KV/softmax/MLP/argmax
kernel/backend/ascend910b/layout.hpp      物理布局、tile、padding、编译开关
kernel/backend/ascend910b/protocol.hpp    epoch/seq/sentinel/trace
kernel/backend/ascend910b/cube.hpp        GEMV ring、Mmad/Fixpipe、落地确认
kernel/backend/ascend910b/vector_core.hpp vector/scalar fence及破坏性reduce
kernel/backend/ascend910b/vector_io.hpp   pack、MTE3、seq、poll、warm
kernel/backend/ascend910b/*_resources.inc  buffer/alias/event ID/GM views
kernel/build.sh                         可从任意cwd调用的bisheng构建
model_run.py                            checkpoint打包、状态、同步、ctypes、解码
run.py                                  CLI
test_mega.py                           随机权重reference、门禁、严格重复
tools/audit.py                         展开include、校验提取无遗漏
tools/check_contract.py                编译前检查18个host/device重复维度
tools/real_probe.py                    独立进程保存真实生成token IDs
validation/provenance.json              上游hash、分片行号、token摘要
```

`.inc` 在入口函数的词法作用域中展开，不产生device函数调用、额外launch或新的同步边界。编译时 `-I.` 以kernel目录为根解析include。stage可以访问resources定义的tensor/event变量；换stage先查交接表，不把它们误当完全独立的纯函数。

本模板的基线展开后与上游R30可执行token摘要相同，本次默认构建的库hash也完全相同。开始移植后 `audit.py --baseline` 失败是预期；不加该选项仍可检查include闭包并输出新摘要。

## 模板边界

这是一个**可编辑的、带正确性基线的生产实现模板**，不是任意模型只改config就能运行的通用生成器。D=128、两Q头/KV、24核/3slice、H与FFN分组等还有专用代码。PORTING明确列出这些耦合，agent必须同步修改stage/packer/reference。

硬件无关层沿用上层skill的方法与逻辑契约；本文件夹的实体AscendC实现属于910B适配层。换NPU需重新认证backend和物理stage映射，不能只改 `--npu-arch`。

本模板没有附带`runtime_model.py`或HIR，不声称模板副本自身已接入CLI check。原项目后来已修复R31 twin并通过CLI门禁，该结果单独记录，未移作本R30模板的新验收；也不能把现有随机近似reference当作新模型的独立CPU精度验收。新模型须补原始权重reference和真实长前缀验证。
