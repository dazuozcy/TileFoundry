# 证据索引、复现步骤与当前目录陷阱

所有命令在项目根执行；选择空闲健康设备并已激活原环境/source CANN。这里提供复现入口，本次文档整理未运行耗时 NPU 验证，不把历史 PASS 写成本次新测。

## 证据优先级与关键路径

| 事实 | 原始材料（相对项目根） | 如何解读 |
|---|---|---|
| 初次生成耗时、走弯路、补 HIR | `session.md` | 修订后的13.5h统计优先于早期估算 |
| API/平台初次发现 | `README.md` 后半 | 特定软件硬件实测；双SyncAll等已成为历史 |
| R1–R30选择与失败 | `OPTLOG.md` | 每轮“当前”仅当轮有效 |
| 进一步会话上下文 | `session-mega-opt-asc.md` | 细节查询用，不优先于最终源码/汇总 |
| 最终源码和构建 | `kernel/mega.cpp`、`kernel/build.sh` | 当前源码与R30 snapshot hash相同 |
| R30精度封闭 | `profiling/r30/final-accuracy-summary.json` | 含308项局部检查、独立轨迹、最大边界 |
| 长自然前缀证据 | `profiling/r30/rsqrt2-accuracy-{4096,16384,40000}/` | 同输入、独立进程、control/repeat/release |
| 最大长度 | `profiling/r30/rsqrt2-cached-max40960/summary.json`、`profiling/r30/final-cpu-max40960.json` | 833 steps与独立CPU检查，非仅随机填充KV |
| 最终随机/重复 | `profiling/r30/final-random-gates.log`、`final-alternating.log`、`final-short.log`（均同目录） | 随机精度与严格重复是不同检验 |
| 最终wall time | `profiling/r30/final-bench.log` | 6轮×30，交替顺序，中位数 |
| 最终device时间 | `profiling/r30/final-roofline-summary.json`、`profiling/r30/roofline-rsqrt2-*` | 15次独立tag launch，弃前5取10 |
| R29布局消融 | `profiling/r29/roofline-summary.json`、`OPTLOG.md` R29 | 解释性能来源，不替代R30精度门禁 |

最终发布记录的 binary SHA256：`355ee25d56368e96c61b58db2be9c8fe19b8a79fbf0f5ab1e1ea02b6bc6b77be`。重建 hash 可能因工具链等变化，须记录新值并复验，不能只凭文件名。

## 1. 先检查实际环境与历史路径

```bash
pwd
npu-smi info
command -v bisheng
command -v tilefoundry
sha256sum kernel/mega.cpp kernel/mega_r30src.cpp kernel/libmegak.so
```

设备0、4、6等均出现在不同时期记录中，部分曾故障或被占用。先查健康和占用再设 `ASCEND_RT_VISIBLE_DEVICES`，不要照抄“永远用0/4”。不安装、升级或替换环境包来掩盖失败。

当前 `debug/*.py` 多处子进程仍调用裸文件名，如 `python isolated_decode_suite.py`；从根目录直接执行 `python debug/long_accuracy_suite.py` 会找不到子脚本。只加 PYTHONPATH 不会修复子进程脚本路径。

本知识库附 [run_debug.py](../scripts/run_debug.py)，以临时平铺 symlink 工作目录兼容原脚本及其子进程；kernel/profiling 指回项目目录，测试结果仍写入原 profiling。它不改原脚本，不复制权重。不指定设备时原脚本可能默认4，运行前显式选择设备。使用新 label 避免覆盖历史结果。

```bash
python retrospective/ascendc-910b-adapter/scripts/run_debug.py --dry-run long_accuracy_suite.py
```

## 2. 构建同精度 control 与候选

以下构建会覆盖指定库，若需保留现有二进制先用不同输出名保存。`profile_r30_suite.py` 固定读取 `libmegak_r30.so`，所以示例为它显式构建，避免默认库重建了但 snapshot 库仍旧。

```bash
bash kernel/build.sh mega.cpp libmegak_retrospective_release.so
MEGA_COMPACT_A=0 MEGA_GU_INTERLEAVE=0 MEGA_LM_STREAM=0 \
  bash kernel/build.sh mega.cpp libmegak_r30_control.so
bash kernel/build.sh mega.cpp libmegak_r30.so
readelf -SW kernel/libmegak_retrospective_release.so
readelf -sW kernel/libmegak_retrospective_release.so
```

release默认：LM tile96、compactA=1、GU interleave=1、LM stream=1；control保留两步Rsqrt与所有R30同步修复，仅关三个优化。不同LM tile/GU布局必须配对应打包；`model_run.py` / `test_mega.py` 会读 `megak_layout()` 并拒绝矛盾环境设置。

## 3. 快门禁 → 长参考 → 严格A/B

先设置设备（下面仅示意编号，必须换成实际选中的空闲健康卡）：

```bash
export ASCEND_RT_VISIBLE_DEVICES=6
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export MEGA_LIB=libmegak_retrospective_release.so
timeout 180s python test_mega.py 0 1
timeout 300s python test_mega.py 300 2
timeout 600s python test_mega.py 39999 28
python retrospective/ascendc-910b-adapter/scripts/run_debug.py long_accuracy_suite.py \
  --contexts 4096 16384 40000 --candidate libmegak_retrospective_release.so \
  --label retrospective-accuracy-new
python retrospective/ascendc-910b-adapter/scripts/run_debug.py bench_variants.py \
  --mode bench --strict --rounds 6 --iterations 30 \
  --variants libmegak_r30_control.so:96:0:0 libmegak_retrospective_release.so:96:1:1
```

上面的 timeout 是待调整的预算示例，超时本身不证明死锁；长reference/prefill可能很久，应按环境给独立进程更长时限。`long_accuracy_suite` 的默认三context不等于最大live40960验证；最大边界还需查看并按参数运行 `debug/cached_accuracy_suite.py`、`debug/cached_decode_check.py` 与 CPU validator。历史最大边界通过结果已经单独索引。

`bench_variants.py` 用合成随机模型作受控性能比较，不等价于完整自然语言端到端 benchmark。`single_decode_check.py` 等真实权重路径独立覆盖自然前缀。不要用严格A/B一致替代原始checkpoint CPU reference。

## 4. profiler：必须 application replay

```bash
python retrospective/ascendc-910b-adapter/scripts/run_debug.py profile_r30_suite.py \
  --label retrospective-roofline-new
```

套件读取刚才构建的 `libmegak_r30.so` 与 `libmegak_r30_control.so`。MIX完整符号匹配、Roofline、application replay、15次launch、CLI warmup=0，随后从CSV弃前5次取10次。每次应用launch有新tag；同tag kernel replay会破坏状态协议，不能用作正确测量。

只从 `OpBasicInfo` 的 `Task Duration(us)` 取设备时间，不用instrumented Python wall time。统计核群平均而非只取block0；对照Memory/L2/PipeUtilization证实优化假设。固定context/初始零KV是性能负载，不是40k自然前缀正确性证明。

## 5. 应用计时与 HIR/twin 的边界

```bash
MEGA_LIB=libmegak_retrospective_release.so python run.py \
  --prompt '详细解释一下昇腾 NPU 上一次矩阵乘法是如何执行的' \
  --max-new-tokens 2048
```

观察真实生成总量和prefill/decode统计；不要把输出文本耗时与kernel-only混用。

初次审计发现twin仍采用早期打包与launch。2026-10-28已对齐当前R31：读取`megak_layout()`，使用chunk-major/paired GU，修复K cache的256-position tile读写，扩充workspace并定义scratch初值，补上logits padding清理与raw launch前同步。恢复的`acts/`、`prepared/`相对链接只借用邻例的输入与HIR权重，不借用其设备kernel。

在项目根执行（checkpoint和设备按实际环境设置，容差保持原值）：

```bash
QWEN3_CKPT=/path/to/qwen3_1_7b ASCEND_RT_VISIBLE_DEVICES=6 \
  bash check.sh --json profiling/check-latest.json
```

`MEGA_CHECK_ACTS`/`MEGA_CHECK_WEIGHTS`可指定独立fixture和HIR权重目录，`MEGA_CHECK_CTX`指定fixture的cache长度。当前报告在项目`profiling/check-r31/`。CLI门禁是全28层、四输出的单step对照；不要将几个边界fixture的PASS扩展为完整自然长前缀生成验证。

本次原始pos255及额外pos0/256/257通过；重复KV历史的人工pos511用例在V行allclose上失败，R30基线有同样失败且全部报告指标一致。该失败保留在`summary.json`及对应CLI JSON中，未放宽容差，也未声称任意合法KV状态均满足原门槛。

`tilefoundry analyze --compute-cost` 历史可用；当时 Ascend 的 memory/roofline/performance 投影不可用，不能声称已有“analyze性能推到底”的完整自动路径。先查当前CLI帮助与支持，再把可得HIR分析和真实字节/带宽模型并列报告。
