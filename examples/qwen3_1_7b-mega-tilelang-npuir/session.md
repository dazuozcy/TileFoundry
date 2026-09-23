# Session 首个 prompt 执行复盘：tilefoundry CLI 交互与耗时分析（修订版）

> **修订说明**：初版报告只覆盖了第三个执行块（~66 min），且用 tilelang 内部日志
> 时间戳外推，严重低估。本版数据源改为 **opencode 自身的会话数据库**
> （`~/.local/share/opencode/opencode.db` 的 `message`/`part` 表），每条工具调用
> 带毫秒级 `time_created`/`time_updated`，所有数字为实测非推断。

**范围界定**：本 session 第一个 prompt 是 05:03:17 的原始任务（"在 TileFoundry
上跑通 Qwen3-1.7B 的真实解码，并把它做快"）。其执行过程横跨**三个连续块**
（期间两次上下文压缩 + 两次用户确认续跑），至 09:48:36 交付全部完成、
09:48:36 输出总结为止。用户下一个独立提问（"已经调优过了？"）出现在 10:00:51。

---

## 0. tilefoundry CLI 交互

**有，共 38 次真实调用**（18 次侦察 + 15 次 check + 5 次 analyze），
按绝对路径 `/home/.../envs/tilefoundry/bin/tilefoundry` 或裸命令调用。

### 按阶段分布

| 阶段 | 时间窗 | 次数 | 形态 |
|---|---|---:|---|
| **A：前期侦察** | 05:03:37–05:11:47 | 18 | CLI-first：任务指引要求的环境勘察（见下表） |
| **B 尾：HIR 首过 check** | 08:37:06–08:40:35 | 7 | `check model.py:Qwen3Mega --expected 发布参考 --inputs random --weights random --dim ctx_len=1,257`，全部结构性失败（4–15s 早退） |
| **C：数值门 + 分析** | 08:57:39–09:45:23 | 13 | 8 次 check（twin 调试→PASS→回归）+ 5 次 analyze |

### A 阶段侦察明细（05:03–05:11，每次 5–10s）

| 命令 | 问的问题 | 得到的答案 |
|---|---|---|
| `--help` | CLI 全貌 | 子命令目录 |
| `models` | 仓库里有什么模型 | 列出本仓 models（含 `tests/models/qwen3_1_7b`） |
| `tutorial` ×4（migrate/showcase/optimize） | 工作流怎么组织 | 迁移四步法、单 kernel 组装、analyze 驱动优化 |
| `target list` / `target show huawei.ascend910b2c` | 目标硬件画像 | dav2201、24 AICore、HBM 带宽 |
| `spec` ×8（hir 1.2.1、tir 1.5、runtime 1.3 …） | DSL/运行时语义 | MeshRegion、Sync、runtime_func/twin 契约 |

侦察后转向框架源码精读（check.py、decorator.py、rope.py 等）补深语义——
**CLI 先行、源码补充**，初版报告"直接读源码"的说法有误。

### B/C 阶段 check/analyze 明细（修正初版的 19 次计数）

初版只数了 C 块且含误报。实测：B 尾 7 次（`--expected` 语义三连败、递归爆栈、
selector 要选到函数）；C 块 8 次 check（twin 打包 bug → rope 越界 → tile 窗口
越界 → FAIL 268/22/249 → 相干输入 0.079 → f32 对齐 0.041 → **PASS** → cta 改名
回归 PASS）+ 5 次 analyze（`aic` 拓扑名非法 → MemoryHierarchyFacts 无投影 →
`--compute-cost` 成功 5.53 GFLOP/step）。**单次 check 仅 4–49 s**（初版
"2–4 min/次"的估计错误，880 s 超时从未触发）。

---

## 1. 耗时与细分（实测）

**总墙钟：05:03:17 → 09:48:36 = 285.3 min（4 小时 45 分）**；
扣除两次等待用户确认续跑（10.1 + 1.7 min）后**纯执行 273.5 min（4 小时 34 分）**
——与"至少 4、5 个小时"的记忆吻合。

### 三块结构

| 块 | 时间窗 | 墙钟 | 工具执行 | 工具调用数 | 主题 |
|---|---|---:|---:|---:|---|
| A kernel bring-up | 05:03:17–06:00:35 | 57.3 min | 21.2 min | 176 | 侦察、kernel 改写、L-sweep、真实权重 7/7 MATCH、首批混沌异常 |
| （用户确认） | 06:00:35–06:10:41 | 10.1 min | — | — | |
| **B 深度调试 + leak 猎杀** | 06:10:41–08:41:05 | **150.4 min** | 75.0 min | 138 | slice-2 尾巴、1-ULP 混沌底噪、SEQ 门创建、**pos-8464 冻结→launcher 1 MB/launch 泄漏**、40960 全程验证、gen_model 适配 + 首过 check |
| （用户确认） | 08:41:05–08:42:47 | 1.7 min | — | — | |
| C 交付物 + 验收门 | 08:42:47–09:48:36 | 65.8 min | 23.0 min | 105 | HIR 修复链、twin、check PASS、HF 64/64、bench、analyze、文档 |

### 耗时归因（全窗口，419 次工具调用）

| 归因 | 耗时 | 占墙钟 | 说明 |
|---|---:|---:|---|
| **模型生成/思考** | **154.3 min** | **54%** | ~230 个助手回合 × 平均 ~40 s（推理 + 代码生成），**最大单项** |
| kernel 测试/编译/探针 | 46.5 min | 16% | 88 次（编译 60–120 s/次、L-sweep、sl 扫描） |
| kernel SEQ 门 | 38.0 min | 13% | 11 次；top3：507 s 首次 SEQ（pos 1304 已失败仍跑满）、437 s + 398 s 两次 40960 全程验证 |
| CLI check/analyze | ~8.4 min | 3% | 20 次运行，4–49 s/次——**初版"check 占 45%"的结论错误** |
| 文件写作（write/edit） | 9.0 min | 3% | 37 次（kernel 改写、生成器、twin、交付物） |
| python 探针/脚本 | 5.5 min | 2% | 29 次（pos-8464 取证、探针） |
| CLI 侦察 | 1.4 min | <1% | 15 条命令 18 次调用 |
| 检视（read/grep）+ 杂项 | 8.8 min | 3% | 213 次 |
| 用户确认间隙 | 11.8 min | 4% | |

### Block B 内部（最大块 150 min 的去向）

1. **pos-8464 冻结猎杀（≈06:52–07:56，~64 min 工具时间）**：首跑 507 s SEQ 发现
   pos 13823 后输出冻结→二分到 8464→poison 元素定位（x[1646]=-172）→
   探针走了三段弯路（探针 block 映射写错、fresh-buffer 无操作、模块数上限
   4–5 个/进程）→"28 层写出相同 k 行 = 层循环重放"突破→**8464 × 1 MB ≈ 8.6 GB
   指向 rtMalloc 泄漏**→wrapper 渲染取证→`syncBlockLock` 每 launch 泄漏 1 MB 确认。
2. **泄漏修复 + 全程验证（≈07:56–08:26，~17 min 工具时间）**：改 jit_npu 模板
   （f-string 转义返工一次）→10240 SEQ PASS→40960 全程 15 检查点 PASS（437 s）。
3. 前段（06:10–06:52）：slice-2 尾巴修复、pass_configs、1-ULP 混沌底噪论证、
   SEQ 模式编写——这段产出 NOTES 陷阱 12/13 的核心认知。

---

## 2. 减少耗时的方案（按真实成本重排，初版结论已修正）

1. **压缩 agent 回合数（针对 54% 的生成/思考时间）**。这不是工具慢，是回合多：
   419 次工具调用意味着 ~230 轮"思考+生成+等待"。抓手：(a) 每回合批量并行
   调用再推一步；(b) **从第一块起维护 NOTES.md 陷阱清单**——两次上下文压缩
   丢了 A/B 块细节，C 块重新读了大量已读过的文件；(c) 大文件一次写成而非
   多轮小补丁。
2. **泄漏猎杀预案化（Block B 的 ~1 小时弯路）**。三个已知信号现在都有档案：
   探针映射 `cid = sid*8 + kv`（06:41 走错）、"各层 k 行相同 = 循环重放"、
   "magic 位置 × 固定步长 ≈ 内存耗尽指纹"。同样的 hunt 今天约 30–45 min。
3. **SEQ 门加 early-abort**。06:52 那次在 pos 1304 就失败了却跑满 507 s；
   首个不匹配即中止可省 ~6 min/次失败跑（成功跑不受影响）。
4. **check 单次 4–49 s，"加速 check"是伪命题**（初版方案 1 作废）。相反，
   可以更放心地高频使用 check 做增量回归。
5. **kernel 编译走热缓存迭代**（46.5 min 的大头）：编译后只改 harness 参数
   重跑、避免触发重新编译的源码小改动；L-sweep 用一次性多 L 脚本代替逐个跑。
6. **侦察流程保留**（A 阶段 18 次 CLI + 源码精读共 ~20 min 换来了整个 session
   零方向性返工，性价比最高，不动）。

综合估计：若 1–3 落实，同规模任务（真实 bug + 三次全量验证的诚实流程）可从
**4 h 45 m 压到 ~3 h**；其中工具时间几乎不可再压（SEQ 全程验证、编译属必要
正确性成本），主要收益来自回合数与弯路消除。

---

*数据来源：opencode 会话数据库 `~/.local/share/opencode/opencode.db`
（session `ses_f3880ca95ffenlT5lPNgU6EIAh`，message/part 表毫秒级时间戳，
419 次工具调用全量统计）。生成于 2026-09-22。*
