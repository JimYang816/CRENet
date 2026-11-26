# CP=80 ms 的正式数据与历史实验记录

## 当前载波分布

NOF1 和 NCS1 的 D-D-D-D 帧每个符号共 1024 个子载波，其中 512 个数据子载波、224 个梳状导频、
224 个带内保护空载波和两端各 32 个边缘空载波。导频相隔 4 或 5 个 FFT 索引，
每个导频的高频侧紧邻一个保护空载波。接收机使用每个符号的 448 个导频与带内空载波进行四符号联合 CSI 估计。

正式数据入口为 `python -m data_generation.scripts.four_data`，默认写入
`data/processed/dddd_pilot224/{nof1,ncs1}_dddd/`。物理先验使用原 CP80 独立记录的 DPSS 基和信道系数。
正式数据验收汇总各划分的平均 NMSE，并抽样计算两种 LMMSE 的 BER。

2026-11-27 的当前正式数据已全量生成和验收：NOF1 约 14.09 GiB，NCS1 约 5.70 GiB，
每信道训练/验证/测试分别为 3,675 / 525 / 4,200 个帧样本，记录位于
`outputs/experiment/dddd_cp80_delay192_pilot224_assisted_20261127/`。

当前均衡算法为 B=16 带状 LMMSE，评测入口为 `python -m scripts.evaluate.compare`，
详见 `docs/equalizer_comparison.md`。
当前参考辅助目标改为 NOF1 -11.41 → -15.77 dB、NCS1 -5.37 → -9.84 dB，
每个 SNR 目标保留两位小数；重校准与完整 B16 测试记录位于
`outputs/experiment/dddd_assisted_b16_20261127/`。下文旧端点为各历史生成版本当时的设置。
下文的 Single-tap / Full-matrix LMMSE 结果及运行命令均为历史记录，旧入口源码见提交 `a08592b`。

## 256 导频的 CP80 实验记录

2026-11-26 起 **CP=80 ms / 480 点为项目默认值**。当时 NOF1、NCS1 的 `dddd.json`、
配置校验、正式数据入口、训练预检及后续指标图均使用该值。
旧 CP256 训练/验证/测试数据和配套共享模型已删除，服务器释放约 18.07 GiB；正式 CP80 补偿数据已生成并全量验收。
默认参考辅助视图及其目标曲线保留，Q=24、192 个冻结物理位置和划分计划保持既定设置。

正式生成的默认先验来源改为本次已验收的 CP80 原始物理实验。
生成运行 ID 为 `dddd_cp80_delay192_assisted_20261126`。当时旧版 CRENet 的双信道训练与全量测试
保存在 `outputs/experiment/four_data_crenet_cp80_20261126/`，实验说明见提交 `38c1b00` 的 `docs/four_data_crenet.md`。
两信道原始 CP256 配置快照保存在 `configs/backups/dddd_cp256_20261125/`，
仅用于核对冻结时延选择及历史实验来源，不作为新的运行默认。

默认切换通过 14 项本地检查及双信道 CPU 小样本生成/完整验收；两信道实际 DPSS 基均为
6016×24，三个划分的补偿曲线均达标，验证用临时 HDF5 随后清理。
数据删除记录、旧数据验收来源和本地验证记录保存在
`outputs/smoke/default_cp80_reset_20261126/`。

## 正式 CP80 补偿数据

生成代码为 `5ee9af5`，运行记录为 `outputs/experiment/dddd_cp80_delay192_assisted_20261126/`。
当时复用同配置完整文件并再次执行全量验收。下表所列旧数据已于 2026-11-27 删除，验收记录保留：

| 信道 | 数据目录 | 训练帧样本 | 验证帧样本 | 测试帧样本 | HDF5 大小 |
| --- | --- | ---: | ---: | ---: | ---: |
| NOF1 | `data/processed/nof1_dddd/` | 3,675 | 525 | 4,200 | 14.06 GiB |
| NCS1 | `data/processed/ncs1_dddd/` | 3,675 | 525 | 4,200 | 5.66 GiB |

样本数计入 0、5、…、30 dB 的 7 个 SNR；每样本含 4 个数据符号、4,096 bit。
每个 SNR 的训练/验证/测试分别为 525/75/600 帧，测试在 150 个物理窗口各重复 4 次。
每个数据目录含 `model.h5`、`train.h5`、`validation.h5`、`test.h5` 和 `manifest.json`，
共 8 个 HDF5、约 19.72 GiB。

默认输入视图为 `reference_assisted`，保留显式 `raw`；完整参考抽头与原始估计系数以
complex64 存储，读取时按完整参考和公共 SNR 误差系数重建补偿 CSI。
训练、验证、测试各自按全量线性 NMSE 均值校准，同一划分、同一 SNR 的全部帧和符号共享系数。
NOF1 的 7 点平均 NMSE 为 -9.69 → -16.05 dB，NCS1 为 -5.24 → -9.13 dB，
均达到尺度 10 dB 的既定曲线，最大偏差小于 `1.8e-15 dB`。
该视图使用完整参考矩阵，是理想参考辅助数据。

两信道实际 6016×24 DPSS 基的 SHA-256 相同；全部物理参考、独立划分、重复数据/噪声种子、
逐帧完整矩阵 NMSE、补偿均值和四符号导频消除均经过验收。
各划分 BER-SNR 图均匀抽取 16 帧做诊断，平均 NMSE-SNR 汇总完整划分，
图表与逐项核验保存在运行目录的信道子目录。

```python
from src.data import FourDataDataset

with FourDataDataset("data/processed/nof1_dddd/train.h5") as dataset:
    inputs, labels = dataset[0]  # 默认补偿后的 reference_assisted CSI
```

本轮主对比使用**参考辅助补偿后的 CSI**，对应版本的入口为
`scripts/evaluate/four_data_cp80_compensation.py`，输出到
`outputs/smoke/four_data_cp80_assisted_20261125/`。下文原始 CSI 实验保留为补偿来源和审计对照。

原始 CSI 实验在记录版本中的入口为 `scripts/evaluate/four_data_cp80.py`，配置为
`configs/evaluation/four_data_cp80.json`。CP 按基带 6 kHz 换算为 **480 点**，
从发射波形、接收 FFT 窗口到全帧 DPSS 先验均按新 CP 重新构造。
不读取 CP=256 的物理数据、估计 CSI 或已校准的参考辅助输入。

保留 D-D-D-D、FFT=1024、每符号 512 数据/256 梳状导频、12 Hz / Q=24、
192 个离散真实时延位置。NOF1 与 NCS1 分别复用已验收独立先验所选出的固定位置，
来源为 `configs/data/{nof1,ncs1}/delay_support192.json`，不重排为连续索引。
NOF1 的配置完整保留原始选点记录、归档哈希和物理帧哈希；这些位置原先由 CP=256
下的独立记录冻结，本次保持位置不变，重新计算 CP=480 下的先验系数和协方差。

先验使用记录 51–55，在 2、8、14、20、26 s 共 25 个窗口重新构造。
测试使用原测试划分中的记录 11、15、19、23、27，各取 2 s 起始的一帧。
每条信道、每个 SNR 共 5 帧、20 个数据符号、20,480 bit；扫描 0、5、…、30 dB。
同一帧的各 SNR 共用标准噪声，各算法及 CSI 选择共用完全相同的发射数据、观测和热噪声。
不同 CP 实验的物理波形和噪声窗不同，不宣称与 CP=256 结果逐样本配对。

每条信道输出四条 BER 曲线：原始估计 CSI / 完美 CSI，各自采用 Single-tap LMMSE
和 Full-matrix LMMSE。单抽头使用原始数据载波观测；完整矩阵消除已知导频，
使用全部接收载波及实际滤波热噪声协方差。估计 CSI 不做参考辅助 NMSE 补偿。

完美 CSI 使用当前符号完整的物理矩阵，包含全部 1024 个时延位置，
不经过 Q=24 投影或 192 点裁剪。实际物理波形中的跨符号干扰仍保留，因此其含义是
“完美单符号 CSI”，不是整帧联合理想均衡。逐帧记录物理模型误差、跨符号泄漏，
并额外复算 30 dB 完美 CSI 的逐符号误码。

NMSE 为各帧、四个数据符号的完整矩阵线性误差比先平均再取 dB，包含未选中时延误差。
两种均衡算法使用同一份 CSI，所以共享其输入 NMSE。完美 CSI 对自身 NMSE 为 0，
即 −∞ dB，图中以文字标注。BER 零误码点显示在半个误码的计数下限，CSV 保留实际零值。

CP 加长使全帧 DPSS 时间长度变为 6016 点。两条信道共用相同实际 DPSS 基；
固定 Q=24 时尾能量比约为 0.0201846，作为模型截断诊断保留，不自动增加 Q。

```bash
# 本地仅做真实物理链路 CPU 冒烟：2 个先验窗口、每信道 1 帧、0/15/30 dB
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m scripts.evaluate.four_data_cp80 --device cpu --threads 4 --smoke --output outputs/smoke/four_data_cp80_local_20261125

# 服务器完整小实验；CUDA 不可用时省略 --device 使用自动选择
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/fay/anaconda3/envs/jimyang/bin/python -m scripts.evaluate.four_data_cp80 --device cuda --threads 4 --output outputs/smoke/four_data_cp80_20261125
```

输出包含 `comparison.png/pdf`、`ber_snr.png/pdf`、`nmse_snr.png/pdf`、`summary.csv`、
`metrics.json`、`manifest.json` 和 `verification.json`；各信道子目录保留实际物理帧、
冻结支撑、独立先验覆盖和逐帧误码/NMSE。完整物理数据保留服务器，图、表及来源记录带回本地。
这些生成产物不提交 Git。

## 本次结果

服务器以提交 `6414900` 在 RTX 4080 上完成双信道测试，用时约 176 s。
本地 4 项检查和双信道真实 CPU 冒烟通过，服务器 26 项相关检查通过。
结果保存在 `outputs/smoke/four_data_cp80_20261125/`，30 dB 汇总如下：

| 信道 | 估计 CSI 单抽头 BER | 估计 CSI 完整矩阵 BER | 完美 CSI 单抽头 BER | 完美 CSI 完整矩阵 BER | 估计 CSI 平均 NMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| NOF1 | 0.0261719 | 0.0243652 | 0.00566406 | 0.0000488281（1 bit） | -10.2808 dB |
| NCS1 | 0.161670 | 0.133545 | 0.125830 | 0（0 bit） | -4.4412 dB |

每个 BER 点分母均为 20,480 bit。NOF1 仍存在少量残余符号间耦合，
最大算子泄漏比例为 `1.3870e-3`、最大物理观测预测误差比为 `1.5558e-3`；
NCS1 对应泄漏比例为 0，最大预测误差比为 `4.2543e-9`。
完美 CSI 结果和估计 CSI 结果使用相同实际接收波形；未消除真实波形中的残余跨符号项。
该结果仅含每信道 5 帧，是小样本诊断。

`verification.json` 已核对逐帧误码分子、BER 分母、线性 NMSE 汇总、实际 CP 和公共 DPSS 基，
并从物理观测额外复算 30 dB 完美 CSI 的逐符号误码。
`transfer_manifest.json` 保存全部带回文件的 SHA-256，以及留在服务器的两份物理 HDF5 的路径、
大小、运行身份和 SHA-256；`runtime.json` 保存数值库、CUDA、硬件及运行命令。

## 参考辅助补偿后的主对比

补偿复用同一次 CP=480 的物理波形、发射数据和噪声，不重新生成物理链路。
首先核对来源物理 HDF5 的验收 SHA-256、运行身份、6016×24 的实际时间基、
25 个独立先验窗口、5 个测试窗口和两条信道的冻结 192 点支撑，
然后重建原始 CSI，并逐帧核对原始 BER/NMSE 与完美 CSI 的误码数。

目标继续采用 `configs/evaluation/four_data_delay192_compensation.json` 中的既定曲线：
NOF1 -9.69 → -16.05 dB，NCS1 -5.24 → -9.13 dB，尺度为 10 dB 的归一化指数下降。
每个信道、每个 SNR 按全部 5 帧/20 个数据符号的平均线性完整矩阵 NMSE 计算一个公共系数：

`alpha(s) = sqrt(10**(target(s)/10) / mean_raw_linear_NMSE(s))`

在完整 1024 点支撑上构造 `H_assisted = H_ref + alpha(s) * (H_raw - H_ref)`。
该构造使用完整参考矩阵，属于理想参考辅助补偿，不代表接收机无参考 CSI 估计性能。
同一 SNR 的全部帧和符号共享系数，保留相对误差差异；不逐帧或逐符号强行归一到目标。

`comparison.png/pdf` 为补偿后单抽头、补偿后完整矩阵、完美 CSI 单抽头、完美 CSI 完整矩阵
的四条 BER 曲线，以及补偿后平均 NMSE 与目标的对比。
`summary.csv` 和另外的 BER/NMSE 图保留原始结果作为来源审计对照。

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/fay/anaconda3/envs/jimyang/bin/python -m scripts.evaluate.four_data_cp80_compensation --device cuda --threads 4
```

补偿评测提交为 `02e5d33`，RTX 4080 计算及出图约 17.3 s。
本地 6 项 CPU 补偿/协议检查、服务器 19 项相关检查通过。
两条信道的全部 7 个 SNR 都达到既定平均 NMSE，最大偏差小于 `7.2e-15 dB`；
原始 CSI 的逐帧误码和 NMSE、完美 CSI 的全部 SNR 误码数均复现补偿来源。
30 dB 的**补偿后主结果**如下，每点仍为 20,480 bit：

| 信道 | 补偿后单抽头 BER | 补偿后完整矩阵 BER | 完美 CSI 单抽头 BER | 完美 CSI 完整矩阵 BER | 补偿后平均 NMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| NOF1 | 0.0112305（230 bit） | 0.00434570（89 bit） | 0.00566406（116 bit） | 0.0000488281（1 bit） | -16.05 dB |
| NCS1 | 0.138281（2,832 bit） | 0.0230957（473 bit） | 0.125830（2,577 bit） | 0（0 bit） | -9.13 dB |

结果目录的 `metrics.json`、`summary.csv`、`verification.json` 保存补偿尺度、逐帧误码、
完整矩阵 NMSE 和校验；`comparison.png/pdf` 为补偿后的主图。
原始数据和完美 CSI 对照保留实际物理波形中的残余跨符号项。
该补偿明确依赖完整完美 CSI，是理想参考辅助实验；每信道仅 5 帧，零误码仅指本次样本。
