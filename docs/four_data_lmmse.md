# 四数据符号 LMMSE 小实验

2026-11-26 起默认 CP 为 **80 ms / 480 点**，当前入口与补偿主结果见
[CP80 实验](four_data_cp80.md)。本页下方为 CP=256 的历史实验及已删除数据的来源记录。
旧训练、验证、测试 HDF5 与配套共享模型已删除；正式 CP80 补偿数据已重新生成并全量验收。

正式四符号 CRENet 独立训练及全量三算法/完美 CSI 测试入口见
[四符号双信道训练与测试](four_data_crenet.md)。旧两符号入口和检查点不用于该实验。

## 历史 CP256 配置与保留结果

本页历史实验使用 D-D-D-D、FFT=1024、CP=256、512 数据与 256 梳状导频、12 Hz / Q=24，
两条信道均估计 192 个物理时延抽头。原始 CSI 和参考辅助结果分别保留在：

| 用途 | 目录 |
| --- | --- |
| 双信道 192 点原始 CSI、基线指标与完整物理数据 | `outputs/smoke/four_data_delay192_20261124/` |
| 当前指定端点的参考辅助指标、图表与核验记录 | `outputs/smoke/four_data_delay192_target_nmse_specified_20261124/` |

参考辅助目标为 NOF1 -9.69 → -16.05 dB、NCS1 -5.24 → -9.13 dB，使用尺度 10 dB 的先陡后缓指数曲线。
它包含测试参考矩阵的完整支撑，属于理想辅助敏感性实验；原始 CSI 基线保持不变。
直接复现依赖为：

| 配置 | 用途 |
| --- | --- |
| `configs/data/{nof1,ncs1}/dddd.json` | 两条信道的基础物理帧和数据划分 |
| `configs/data/ncs1/delay_support192.json` | 冻结的 NCS1 物理选点与独立先验来源 |
| `configs/evaluation/four_data_delay192.json` | 双信道 192 点原始 CSI 实验 |
| `configs/evaluation/four_data_delay192_compensation.json` | 当前指定端点与补偿曲线 |
| `configs/evaluation/ncs1_delay_support.json` | 共用接收机及原始实验代码身份的依赖 |

训练、统一评测、历史 P-D-D-P/L=176 入口仍引用其独立配置；`configs/backups/` 保存正式模型的参数来源。
清理时依赖审计确认 13 份独立配置与 20 份备份均有保留用途，没有孤立的端点备选配置。
本轮清理旧结果目录内的运行配置快照及缓存，所有保留配置的 SHA-256 均未变化。

本地清理 3 个目录，服务器清理 13 个目录，服务器释放约 438 MiB。
旧端点、L=176、固定 5 dB 以及单 NCS1 重复小实验已清理，测试源码与正式模型来源继续保留。
单 NCS1 的先验与当前原始数据一致，独立选点来源已迁入
`outputs/smoke/four_data_delay192_20261124/ncs1/support_selection_provenance.json`。
逐文件删除清单、哈希和保留项校验见指定端点结果目录的
`cleanup_configuration_local.json` 与 `cleanup_configuration_remote.json`。
清理后本地 15 项与服务器 40 项现有检查通过，原始物理数据身份、当前目标曲线和独立 PDP 来源均复核通过；
验证临时目录已随测随清。

## 正式 D-D-D-D 训练、验证与测试数据

正式入口为 `data_generation/scripts/four_data.py`，协议选项为 `configs/data/four_data_dataset.json`。
沿用上述物理帧、独立 25 窗口先验及两信道冻结的 192 个物理抽头，直接复用已验收原始小实验的先验。
原始 CSI 的联合估计不读取本窗口参考矩阵；参考矩阵用于独立误差评分和用户指定的参考辅助补偿。
默认输入视图为 `reference_assisted`，并保留显式 `raw` 视图。

每信道按现有 `learning_dataset` 划分 35/5/10 条记录，窗口起点为 2、4、…、30 s。
训练、验证、测试记录互斥，且与先验记录 51–55 和保留的 CSI 验收记录 56–60 互斥。
所有测试重复均有独立的发射数据和噪声种子；同一帧在各 SNR 下使用配对的标准噪声。

| 划分 | 物理窗口 | 每窗口重复 | 帧数 / SNR | 含 7 个 SNR 的帧样本数 |
| --- | ---: | ---: | ---: | ---: |
| train | 525 | 1 | 525 | 3,675 |
| validation | 75 | 1 | 75 | 525 |
| test | 150 | 4 | 600 | 4,200 |

每个样本包含 4 个数据符号、4,096 bit。三套数据都使用当前目标配置的 7 点曲线：
NOF1 -9.69 → -16.05 dB，NCS1 -5.24 → -9.13 dB，尺度为 10 dB。
默认 `compensation_calibration=per_split`，每个划分按其完整线性 NMSE 均值计算公共 SNR 误差系数，
同一 SNR 的所有帧和符号共用系数。各划分独立完成参考辅助校准，测试参考信息不参与训练划分的系数计算。
此视图属于明确标记的理想参考辅助数据，不能解释为真实接收机的无参考 CSI 估计性能。

正式文件写入 `data/processed/{nof1,ncs1}_dddd/`，每个目录包含 `model.h5`、`train.h5`、
`validation.h5`、`test.h5` 和验收 `manifest.json`。完整 1024 点物理参考抽头以 complex64 每窗口存一次，
原始估计以 24×192 complex64 系数存储；补偿矩阵按参考、原始系数和冻结误差系数重建，保留全部延迟位置。
物理构造和估计使用 complex128，所有验收与补偿使用落盘精度定义的参考和原始系数，避免精度口径不一致。
两信道的保守未压缩存储上界合计约 49.2 GiB。

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/fay/anaconda3/envs/jimyang/bin/python -m data_generation.scripts.four_data all --device cuda --threads 4 --workers 6 --worker-threads 2
```

每个物理窗口与帧在写入并刷新后提交完成标记，重复运行相同命令会续跑未完成部分。
运行身份固定生成代码哈希、当前配置、原始信道归档和独立先验来源；参数变化须使用新的运行身份与输出目录。
验收与运行记录写入 `outputs/experiment/dddd_delay192_assisted_20261124/{nof1,ncs1}/`，
总验收清单为同一运行目录的 `dataset_manifest.json`。

验收逐窗口检查完整参考哈希，逐帧检查数据、噪声、随机种子、SNR、完整矩阵 NMSE 和补偿均值，
并通过显式完整矩阵检查四个符号的载波/标签/导频消除。BER-SNR 对每个划分均匀抽取最多 16 帧做诊断，
报告明确记录样本身份和比特数；NMSE-SNR 汇总该划分的全部帧。

四符号读取接口与历史 P-D-D-P 接口分离，历史两数据符号训练器须适配后再用于本数据：

```python
from src.data import FourDataDataset
with FourDataDataset("data/processed/nof1_dddd/train.h5") as dataset:
    inputs, labels = dataset[0]  # 默认参考辅助 CSI，包含四个数据符号
# 原始接收机估计视图：FourDataDataset(path, input_view="raw")
```

本次正式生成代码为 `ad3a007`，本地 24 项检查、服务器 49 项检查及双信道真实 CPU 冒烟通过。
两信道共 8 个 HDF5 文件已完成生成、全量逐帧验收及 SHA-256 复核，两个信道的 DPSS 时间基哈希相同。
NOF1 占用约 12.40 GiB，NCS1 约 5.67 GiB，合计约 18.07 GiB。
训练、验证、测试各自的 7 个 SNR 平均 NMSE 均达到上述曲线。

服务器首次准备模型时发生 SciPy/OpenBLAS 外部崩溃，未写入样本；统一库线程数后恢复并全部验收通过。
实际成功运行使用 `OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`，
主进程 4 线程、6 个物理工作进程、每工作进程 2 线程。恢复记录见 `runtime.json`，数值库版本与环境见
`numerical_runtime.json`；总验收为运行目录的 `verification.json`。

## 历史 L=176 链路基准

本节先记录 L=176 双信道历史链路基准；当前双信道 192 点小实验见下文“NOF1 与 NCS1 的离散 192 点小实验”。该历史小实验使用 `configs/data/{nof1,ncs1}/dddd.json` 与
`configs/evaluation/four_data_lmmse.json`。两条信道使用相同的 D-D-D-D 帧、
1024 点 FFT、256 点循环前缀、512 个数据子载波、256 个梳状导频、128 个带内保护空载波和两侧各 64 个边缘空载波。
时间域 DPSS 基跨完整帧和循环前缀，半带宽 12 Hz，固定 Q=24、延迟支撑 L=176。
其余链路参数和模型误差比率 0.03 沿用原配置。先验使用独立记录 51–55；
测试使用已定义测试划分中的记录 11、15、19、23、27，各取 2 s 处一帧。

四个数据符号的梳状导频与保护空载波共同用于无反馈联合 CSI 估计。
均衡直接使用该估计，不按参考 CSI 或目标 NMSE 缩放。先验从独立记录的物理信道构造；
测试帧的参考 CSI 只用于理想参考均衡和 NMSE 评分。两种 CSI 使用相同发射数据和噪声。
单抽头算法沿用数据载波原始观测；完整矩阵算法消除已知导频，使用全部接收载波和有色噪声协方差。
SNR 为四个数据符号接收有用窗平均功率与滤波后预期热噪声功率之比。

```bash
python -m scripts.evaluate.four_data_lmmse --device cuda --threads 4
```

默认写入 `outputs/smoke/four_data_lmmse/`，包括实验与代码身份、各帧物理数据、
逐帧误码及模型诊断、`summary.csv`、`ber_snr.png/pdf` 和 `nmse_snr.png/pdf`。
每条信道每个 SNR 为 20,480 个比特；结果是小样本诊断，不是正式性能结论。
NMSE 为每帧每数据符号完整信道矩阵的线性 NMSE 先平均、再取 dB。参考 CSI 相对于自身
的 NMSE 为 0，dB 值为负无穷，因此图中以文字注明。BER 图中的空心点表示零实测误码，
绘制在半个误码的计数下限；表格仍记录真实零值。

历史 P-D-D-P 压缩数据及基于该帧训练的检查点并不适用于本实验；该入口不会读取它们。
若需后续正式训练，须先为新帧生成并验证对应的数据集和划分。

当前 CP=256、Q=24 时，12 Hz DPSS 累计尾能量比约为 `5.60e-5`，高于配置中的 `1e-5` 自动选阶阈值；
固定 Q=24 优先，阈值只作为诊断，不改变指定阶数。

该 L=176 小实验及其物理波形、旧固定 5 dB 补偿结果已清理，当前保留项见文首。
历史对比脚本仍可作为研究工具，重现时须重新生成对应输入并通过物理参数及来源校验。

## NOF1 与 NCS1 的离散 192 点小实验

入口为 `scripts/evaluate/four_data_delay192.py`，配置为
`configs/evaluation/four_data_delay192.json`。两条信道均使用 192 个离散物理时延位置，
保留 `dddd.json` 的 D-D-D-D、CP=256、512 数据、256 梳状导频以及 12 Hz / Q=24；
运行时将估计系数的时延列数设为 192，每帧待估系数为 24×192=4608。
`dddd.json` 中的 L=176 仅供历史入口使用，不能据此判断本入口的实际抽头数。

NOF1 从记录 51–55 在 2、8、14、20、26 s 的 25 个独立先验窗口计算完整物理 PDP，
按平均功率选择最大的 192 个位置，先冻结再估计测试 CSI。NCS1 复用
`configs/data/ncs1/delay_support192.json`，完整运行时核对这 25 个先验窗口能否重新选出相同位置。
两条信道分别选点，不将选中的物理位置重新排列成 0–191。
测试采用记录 11、15、19、23、27 的 2 s 帧，扫描 0、5、10、15、20、25、30 dB，
每条信道每个 SNR 为 20,480 bit。CSI 为接收机原始联合估计，不使用参考 CSI 补偿。

```bash
python -m scripts.evaluate.four_data_delay192 --device cpu --threads 4 --smoke --output outputs/smoke/four_data_delay192_local
python -m scripts.evaluate.four_data_delay192 --device cuda --threads 4 --output outputs/smoke/four_data_delay192_20261124
```

CPU 冒烟模式仅用两个先验窗口、一帧测试及 0、15、30 dB；其中 NOF1 选点只用于验证流程，
不作为完整小实验的固定支撑。完整运行在服务器进行，物理波形不传入 Git。
输出包含双信道 `ber_snr.png/pdf`、`nmse_snr.png/pdf`、`summary.csv` 和逐帧误码；
各信道子目录另外保存 `frozen_support192.json`、PDP、支撑覆盖、先验/测试划分及代码和数据哈希。
平均 NMSE 对每帧每符号的完整矩阵线性误差比先平均再取 dB，计入所有未选时延的误差。
理想参考检测使用同一发射数据和噪声，仅用于对照；BER 空心点表示零实测误码，显示在半误码计数下限。

本次完整结果保留在 `outputs/smoke/four_data_delay192_20261124/`，物理数据留在服务器同名目录，
图、指标表、逐帧结果和来源记录已带回本地。计算代码为 `139b975`；本地 7 项检查与双信道 CPU 冒烟通过，
服务器 26 项接收机/检测器检查通过。逐帧误码与线性 NMSE 汇总核验通过，NCS1 的 NMSE、BER 和能量覆盖
复现此前固定 192 点结果，详见结果目录的 `verification.json`。

NOF1 的 192 个选点分布在物理索引 45–501 内，包含非连续位置；NCS1 为索引 32–223。
对应测试平均能量覆盖率为 97.4982% 和 99.9737%。30 dB 原始 CSI 指标为：

| 信道 | Single-tap BER | Full-matrix BER | 平均完整矩阵 NMSE |
| --- | ---: | ---: | ---: |
| NOF1 | 0.0294922 | 0.0269531 | -10.3935 dB |
| NCS1 | 0.168213 | 0.136328 | -4.4692 dB |

完整 0–30 dB 指标见 `summary.csv`；本次每信道仅 5 帧，属于小样本诊断。

## 192 点原始 CSI 的目标 NMSE 参考辅助实验

入口为 `scripts/evaluate/four_data_delay192_compensation.py`，目标配置为
`configs/evaluation/four_data_delay192_compensation.json`。使用上一节保存的 192 点原始估计链路、
25 个独立先验窗口与 5 个测试帧，固定相同的数据和噪声，对参考矩阵辅助补偿后的两种 LMMSE 重新评测。
不修改原始实验；当前指定端点版本输出到 `outputs/smoke/four_data_delay192_target_nmse_specified_20261124/`。

目标曲线采用尺度 10 dB 的归一化指数下降：
`t(s) = t(0) + [t(30)-t(0)] * [1-exp(-s/10)] / [1-exp(-3)]`。
因此 SNR 每增加 5 dB，目标 NMSE 的改善幅度逐步减小，且起止值严格匹配要求：

| SNR / dB | NOF1 目标平均 NMSE / dB | NCS1 目标平均 NMSE / dB |
| ---: | ---: | ---: |
| 0 | -9.6900 | -5.2400 |
| 5 | -12.3236 | -6.8508 |
| 10 | -13.9209 | -7.8278 |
| 15 | -14.8898 | -8.4204 |
| 20 | -15.4774 | -8.7798 |
| 25 | -15.8338 | -8.9978 |
| 30 | -16.0500 | -9.1300 |

每条信道、每个 SNR 使用一个公共误差系数
`alpha(s) = sqrt(10**[t(s)/10] / mean_raw_linear_NMSE(s))`，
构造 `H_assisted = H_ref + alpha(s) * (H_raw - H_ref)`。
`H_raw` 先按实际时延位置补零到参考矩阵的完整 1024 点支撑，包含所有原始遗漏抽头的误差。
同一 SNR 的所有帧、所有符号共用系数，保留帧间相对误差；不将每个符号单独归一到相同 NMSE。
平均值仍按完整矩阵的逐帧/逐符号线性 NMSE 先平均，再取 dB。

该构造明确使用测试参考 CSI，补偿后的矩阵可包含原 192 个位置以外的参考分量，
属于理想参考辅助敏感性实验，不能作为实际接收机的 CSI 估计性能。
运行时校验源物理数据身份、冻结支撑和计算代码，重估原始 CSI 并复核原始/参考误码，
同时核验误差缩放恒等式及各 SNR 实际平均 NMSE 与目标的偏差。

```bash
python -m scripts.evaluate.four_data_delay192_compensation --device cuda --threads 4
```

输出 BER-SNR 与 NMSE-SNR 的 PNG/PDF、`summary.csv`、逐帧误码与来源记录。
图中包含原始、参考辅助与理想参考对照；CSV 额外记录目标 NMSE 和公共误差系数。

当前指定端点版本的计算代码为 `7367a1a`，本地 15 项检查与服务器 40 项回归检查通过。
7 个 SNR 的实际平均 NMSE 均达到目标，最大绝对偏差约 `3.6e-15 dB`；原始/理想参考误码
与源实验完全一致，原始物理数据哈希未变。结果目录的 `verification.json` 保存完整核验记录。
每信道为 5 帧、每 SNR 为 20,480 bit，端点及 30 dB 误码指标为：

| 信道 | 0 dB 平均 NMSE | 30 dB 平均 NMSE | 30 dB Single-tap BER | 30 dB Full-matrix BER |
| --- | ---: | ---: | ---: | ---: |
| NOF1 | -9.69 dB | -16.05 dB | 0.0163086 | 0.00688477 |
| NCS1 | -5.24 dB | -9.13 dB | 0.144531 | 0.0269043 |

图、CSV 和逐帧指标已带回本地，原始完整物理数据保留在服务器。

前一版自动选择的非整数端点已由当前指定端点取代，旧结果及其运行配置快照已清理。
历史参数可由计算代码 `b09d43c` 复核；本节目标表与默认入口均对应当前指定端点。

## NCS1 的固定 192 点时延支撑

当前 NCS1 支撑实验仅使用 `configs/evaluation/ncs1_delay_support.json` 与
`configs/data/ncs1/delay_support192.json`。192 个位置由记录 51–55 在 2、8、14、20、26 s
处共 25 个独立先验窗口的平均物理 PDP 冻结，测试 CSI 不参与选点。本次选择恰好覆盖时延索引 32–223，
系数列数为 192，重建跨度为 224；需保留物理位置，不能把这 192 列移到 0–191。

帧保持 CP=256、512 数据与 256 梳状导频，DPSS 为 12 Hz / Q=24，待估系数为 4608。
`dddd.json` 提供基础物理帧，当前入口显式将模型抽头数设为 192。接收机使用原始联合 CSI 估计，
不使用参考 CSI 补偿误差。测试固定为记录 11、15、19、23、27 的 2 s 帧，扫描 0–30 dB。

```bash
python -m scripts.evaluate.ncs1_delay_support --device cpu --smoke --output outputs/smoke/ncs1_delay192_local
python -m scripts.evaluate.ncs1_delay_support --device cuda --threads 4 --output outputs/smoke/ncs1_delay192
```

CPU 冒烟模式仅用两个先验窗口、一帧测试及 0、15、30 dB，选点仍读取完整先验冻结的 192 点配置。
输出仅包含 192 点原始 CSI 及理想参考检测结果，包括平均完整矩阵 NMSE-SNR、两种 LMMSE 的 BER-SNR、
PDP、抽头索引与能量覆盖、代码和数据身份。NMSE 包含全部遗漏时延分量，按帧/符号的线性误差比先平均再取 dB。

该单信道结果已并入当前双信道 192 点原始实验，重复目录与物理数据已清理。
独立先验和 PDP 与保留的原始数据核对一致；原始归档、源代码、选点和评分来源记录见
`outputs/smoke/four_data_delay192_20261124/ncs1/support_selection_provenance.json`。

固定 192 点独立运行的计算代码为 `5c15dd4`。本地相关测试通过，服务器 19 项接收机及检测器测试通过；
25 个先验窗口、5 个测试帧和 7 个 SNR 的 NMSE、BER 及理想参考误码计数均复现选定方案，PDP 哈希也一致。
每个 SNR 为 20,480 bit；当前保留指标为：

| 指标 | 固定 192 点结果 |
| --- | ---: |
| 测试平均能量覆盖 | 99.9737% |
| 最差符号能量覆盖 | 99.9356% |
| 30 dB 完整矩阵平均 NMSE | -4.4692 dB |
| 30 dB Single-tap BER | 0.16821 |
| 30 dB Full-matrix BER | 0.13633 |

当前单信道原始指标和完整物理数据保留在双信道原始目录的 `ncs1/` 下，
本次统一清理明细见文首所列的两份配置清理记录。

## 历史参考 CSI 辅助的 5 dB 误差补偿实验

固定 5 dB 入口 `scripts/evaluate/four_data_csi_compensation.py` 保留作历史研究工具。
旧输入和结果已清理，重现时须先生成 L=176 原始实验并显式传入新的 `--source` 和 `--output`；
当前指定端点实验使用前述 `four_data_delay192_compensation` 入口。

该实验读取上述 CP=256、Q=24 物理数据，重新计算原始 CSI 并核对原始及参考 CSI 的误码。
对每帧、每个 SNR 和每个数据符号，将原始 CSI 补零到参考矩阵的完整延迟支撑后构造
`H_assisted = H_ref + 10^(-5/20) * (H_raw - H_ref)`，使每个完整信道矩阵的 NMSE
相对原始值降低 5 dB。完整支撑包含原 L=176 之外的参考分量。两种 LMMSE 共用原接收观测、
导频和噪声；结果与原始和参考 CSI 一并绘制，输出独立存放，不覆盖原始实验。
该构造使用测试样本的参考 CSI，属于理想辅助的敏感性实验，不能作为实际接收机估计性能。
