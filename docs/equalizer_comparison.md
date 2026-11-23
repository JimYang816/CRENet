# NOF1 / NCS1 七算法统一评测

> 本文记录旧 P-D-D-P / 参考辅助补偿条件下的设计或结果，仅供历史参考。当前实验使用 D-D-D-D 帧和未经补偿的 CSI；范围及运行方式见 [四数据符号 LMMSE 小实验](four_data_lmmse.md)。

主比较包含 Single-tap LMMSE、Full-matrix LMMSE、MMSE-DFE、DD-DPSS、UDNet、OAMP-Net2 和 CRENet。统一入口为 `scripts.evaluate.compare`，配置为 `configs/evaluation/ber_comparison.json`。CRENet 默认采用8层联合 CSI 修正与 MMSE 展开。各方法的实现见 [CRENet 算法设计](crenet_algorithm_design.md)、[MMSE-DFE](mmse_dfe.md)、[软判决 DPSS-LMMSE 重估](dd_ce_lmmse.md) 和 [OAMP-Net2](additional_equalizers.md)。

## 数据与指标

所有算法使用相同接收观测、已验收的 compensated CSI、既定物理记录划分和 QPSK 映射。NOF1 数据目录为 `data/processed/nof1_compact_20261026_final`，NCS1 为 `data/processed/ncs1_compact_20261026_nmse`。SNR 0→30 dB 时，输入平均 NMSE 分别为 −12.70→−18.30 dB、−6.20→−10.67 dB。

compensated CSI 按已保存的 alpha 构造 `H_init=H_ref+alpha*(H_raw-H_ref)`，属于既定的参考辅助数据构造。检测器只接收构造后的 CSI、含噪观测、已知导频及接收机信息，不单独接收参考 CSI、alpha 或数据标签。raw 视图仅用于显式诊断，检查点和缓存不得跨视图复用。

正式测试每算法、每 SNR 使用全部600帧、D1/D2和512个数据槽，共1,228,800比特。模型训练与超参数选择仅使用训练集和验证集；统一汇总再次核对数据哈希、输入视图、SNR与比特总数。

当前报告位于 `outputs/experiment/equalizer_comparison_dd_ce_20261102/{nof1,ncs1}/`，包含 `metrics.json`、`report.html`、七算法的 `ber_snr.png` 和平均 NMSE 图 `nmse_snr.png`。历史报告中的旧绝对路径保留为生成时的来源记录，实际文件以迁移后的目录为准。

NMSE 对每块完整矩阵的线性相对误差先平均，再转换为 dB。固定 CSI 方法共用输入曲线，DD-DPSS 和 CRENet 各自报告修正后的 CSI。零误码在图中以 0.5/总比特数处的空心倒三角显示，数值 BER 为零。

## 配置与运行

| 算法 | 当前配置 | 实现 |
|---|---|---|
| Single-tap LMMSE | 无额外参数 | `src/algorithms/single_tap_mmse/equalizer.py` |
| Full-matrix LMMSE | 完整矩阵、完整热噪声协方差 | `src/algorithms/full_matrix_lmmse/equalizer.py` |
| MMSE-DFE | 完整矩阵，Cholesky，固定逆数据列顺序，硬QPSK反馈 | `src/algorithms/mmse_dfe/detector.py` |
| DD-DPSS | QPSK 后验软均值作虚拟导频，在 32 个 DPSS 先验模式内重估一次 CSI，再做完整矩阵 LMMSE 检测 | `src/algorithms/dd_ce_lmmse/detector.py` |
| UDNet | 22层，窗口32，隐藏维度48 | `src/algorithms/udnet/` |
| OAMP-Net2 | 10层，方差下限5e-13 | `src/algorithms/oamp_net2/` |
| CRENet | 默认8层，B=16，逐层CSI修正，窗口32/16 | `src/algorithms/crenet/` |

配置中的 `checkpoints` 指定每信道三个学习模型的正式检查点。以下路径中的 `<channel>` 为 `nof1` 或 `ncs1`：

| 模型 | 当前正式检查点 |
|---|---|
| CRENet | `outputs/experiment/icr_full_20261031_095549/joint/<channel>/icr_mmse.pt` |
| UDNet | `outputs/experiment/icr_full_20261031_095549/udnet/<channel>/best.pt` |
| OAMP-Net2 | `outputs/experiment/oamp_net2_compensated/<channel>/best.pt` |

开始计算前校验文件及身份；缺失或不匹配会报错。`icr_budgeted_unified_20261031` 和 `icr_budgeted_20261031` 的检查点只作为历史预算复核来源，不是当前主比较配置引用的权重。

```bash
# 默认七算法统一评测
python -m scripts.evaluate.compare --device cuda --threads 4
python -m scripts.evaluate.compare --channel NOF1 --device cuda --force dd_ce_lmmse
python -m scripts.evaluate.compare --channel NCS1 --device cuda --force crenet

# 只运行四种传统方法，不依赖学习检查点
python -m scripts.evaluate.compare --device cuda --baselines-only

# 从保存数值重新绘图
python -m scripts.evaluate.compare --plot-only
```

`--force` 接受 `single_tap_mmse`、`full_matrix_lmmse`、`mmse_dfe`、`dd_ce_lmmse`、`crenet`，可同时指定多项。`--force all` 刷新已配置的传统算法、OAMP-Net2 和 CRENet 结果。UDNet 按自身数据、权重与源码身份复用缓存。`--channel` 限定信道，`--plot-only` 从已保存的指标重绘。日常用法见 [独立训练与增量评测](incremental_evaluation.md)。

从头训练当前 CRENet 并自动衔接完整评测时，使用独立实验目录：

```bash
python -m scripts.run.crenet_experiment --output outputs/experiment/crenet_full_<run_id> --device cuda --threads 4
# 同一代码、配置与目录中断恢复
python -m scripts.run.crenet_experiment --output outputs/experiment/crenet_full_<run_id> --device cuda --threads 4 --resume
```

该入口依次完成两信道共享预训练、固定与联合分支、全参数学习审计、七算法完整测试及固定分支测试。
比较报告自动绑定本次联合权重；OAMP-Net2 和已有有效 UDNet 检查点复用，缺失的 UDNet 按正式配置训练。
运行前核对完整数据划分、compensated CSI 及基线身份。`manifest.json` 保存代码和配置，
`progress.json` 标记当前步骤、成功或失败，`logs/` 保存逐步日志；最终 `comparison/{nof1,ncs1}/`
包含两类指标图和 HTML 报告，`results_digest.json` 汇总两信道实际误码率。旧实验目录不改写。

论文正式消融 A–D 与七算法主比较分开汇总。C 组使用学习型迭代 CSI 修正，DD-DPSS 使用软符号和接收机的 DPSS 先验。控制条件见 [CRENet 说明第5节](crenet.md#5-比较消融与输出)。

## 缓存与效率

传统基线只校验并读取 `test.h5` 和 `model.h5`。完整矩阵输入保留所有接收行和数据列，按当前 CSI 消除已知导频。未提供算法配置时只运行两种 LMMSE，主比较配置另外启用 MMSE-DFE 和 DD-DPSS。

缓存位于 `outputs/experiment/_cache/`，逐SNR、帧和D块记录完成状态、校验和及来源。计算身份绑定输入视图、算法参数、所需代码、数据内容和数值环境；更换batch、线程数、输出路径或只修改文档不使有效数值缓存失效。目录迁移改变了源码模块标识，因此旧计算缓存须重新通过身份校验；历史报告仍可离线重画。新增计算辅助函数时，在 `src/evaluation/registry.py` 和 `src/evaluation/fingerprints.py` 登记实际依赖。

损坏块重新计算，中断后从有效块续跑，文件锁阻止同键并发写入。数据在计算中变化会中止运行；报告在所有所需曲线和两类指标图完成后发布。离线重画会剔除退役算法的曲线和参数，保留其余算法的原始计数与来源。

效率入口 `scripts.evaluate.crenet_efficiency` 默认比较七种算法，要求三个学习模型检查点。每算法使用独立进程，计入 CSI 重建、预处理和全部迭代，报告初始化耗时、每块耗时、参数量和峰值内存。完整命令见 [CRENet 运行说明](crenet.md#6-验证与运行)。

## 论文材料

论文主稿为 `outputs/paper/main.tex`，论文使用的图片直接放在主稿同目录。现有算法图为 `算法架构图.pdf` 及其可编辑 PowerPoint 源文件，图注对应(a)八层展开网络、(b)单个联合层。此前图件设置为 PNG 5520×2580，SVG 印刷宽度 182 mm 并保留可编辑文字与图形；原独立生成脚本已从当前工作区移除。历史稿保存在 `outputs/paper/archive/`，实验检查点和数值结果继续留在 `outputs/experiment/`。
