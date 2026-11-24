# 四数据符号 LMMSE 小实验

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

当前保留的小实验结果位于 `outputs/smoke/four_data_cp256_q24_20261123/`；
完整物理波形保留在服务器的同名目录。本轮旧 CP/Q 实验的波形、结果和对比图已清理。
历史对比脚本仍可作为研究工具，其输入必须通过物理参数及来源校验。

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

当前保留结果为 `outputs/smoke/ncs1_delay192_20261124/`，完整物理数据留在服务器。
本轮其他抽头数的配置、对比结果和 CPU 缓存已清理；选点来源通过配置中的原始归档哈希、
源代码提交、独立先验窗口及 PDP 哈希复核，保留结果中的 `selection_provenance.json` 记录迁移验证来源。

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

本轮清理了本地 6 个旧支撑实验/CPU 缓存目录及服务器 3 个旧支撑实验目录；
删除明细分别记录在保留结果目录的 `cleanup_local.json` 与 `cleanup_remote.json`。

## 参考 CSI 辅助的 5 dB 误差补偿实验

为研究 CSI 误差降低时的均衡表现，可单独运行：

```bash
python -m scripts.evaluate.four_data_csi_compensation --device cuda --threads 4
```

该实验读取上述 CP=256、Q=24 物理数据，重新计算原始 CSI 并核对原始及参考 CSI 的误码。
对每帧、每个 SNR 和每个数据符号，将原始 CSI 补零到参考矩阵的完整延迟支撑后构造
`H_assisted = H_ref + 10^(-5/20) * (H_raw - H_ref)`，使每个完整信道矩阵的 NMSE
相对原始值降低 5 dB。完整支撑包含原 L=176 之外的参考分量。两种 LMMSE 共用原接收观测、
导频和噪声；结果与原始和参考 CSI 一并绘制，输出独立存放，不覆盖原始实验。
该构造使用测试样本的参考 CSI，属于理想辅助的敏感性实验，不能作为实际接收机估计性能。
