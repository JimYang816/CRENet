# 带状 LMMSE 评测

当前算法为 `src/algorithms/banded_lmmse/` 中的 Banded LMMSE，半带宽固定为 16。
先用完整输入 CSI 消除已知导频，再按实际 FFT 索引截断数据通道至 ±16，
最后使用全部接收行与完整热噪声协方差完成 LMMSE 求解。

恢复后的 CRENet 使用独立训练与测试入口：仅用 NOF1 训练验证，再用同一冻结模型分别测试
NOF1 和 NCS1，并与此处的 B16 对照比较，见 [CRENet 实验说明](four_data_crenet.md)。

令 D 为数据载波的实际 FFT 索引，使用 `r = y - H_est x_known` 消除导频，
构造 `A[i,j] = H_est[i,D[j]] * 1(abs(i-D[j]) <= 16)`。
在噪声白化之前截断通道，再求解
`x_hat = (A^H R^-1 A + variance I)^-1 A^H R^-1 r`。
其中 R 为单位 RF 噪声经过真实接收滤波后的完整协方差。
每个数据列最多保留 33 个接收载波，导频间隙也计入 FFT 索引距离。

正式数据由 `src.data.FourDataDataset` 读取，目录为
`data/processed/dddd_pilot224/{nof1,ncs1}_dddd/`。
默认数据与评测视图 `reference_assisted` 使用完整参考 CSI；可显式选择 `raw` 原始接收机 CSI。
每帧包含四个数据符号，正式测试划分每个 SNR 有 600 帧。

当前目标按 0、5、…、30 dB 排列，保留两位小数：

| 信道 | 各 SNR 的平均 NMSE 目标 / dB |
| --- | --- |
| NOF1 | -11.41, -13.22, -14.31, -14.97, -15.38, -15.62, -15.77 |
| NCS1 | -5.37, -6.84, -7.73, -8.27, -8.60, -8.80, -8.92 |

训练、验证和测试分别使用该划分的全量线性 NMSE 均值计算公共 SNR 系数，
构造 `H_assisted = H_ref + alpha_s (H_raw - H_ref)`。
完整参考包含未估计的物理时延，补偿完成后才构造频域带状数据通道。
原始系数、发射标签、观测和参考不改变；原始生成身份与重校准身份分别保留。

重新校准六份数据并进行完整测试：

```bash
python -m data_generation.scripts.recalibrate_four_data all --device cuda --threads 4 --batch-size 8
```

入口先核对旧验收文件哈希并保存旧补偿参数，重校准后逐帧验收全部划分，
然后对每信道全部 600 帧/SNR、每帧四符号进行 B16 测试。
默认输出为 `outputs/experiment/dddd_assisted_b16_20261127_v2/`；每信道每 SNR 为 2,457,600 bit。

本次仅更新 NCS1 至 -5.37 → -8.92 dB，保持 NOF1 正式数据与结果，运行：

```bash
python -m data_generation.scripts.recalibrate_four_data all --channels NCS1 --output outputs/experiment/dddd_assisted_b16_ncs1_892_20261127 --device cuda --threads 4 --batch-size 8
```

单信道模式仅重校准和验收选中信道的训练、验证、测试集，并对该信道进行完整 B16 测试。

已有正式测试数据的增量评测使用：

```bash
python -m scripts.evaluate.compare --device cuda --threads 4
python -m scripts.evaluate.compare --force banded_lmmse
python -m scripts.evaluate.compare --input-view raw
python -m scripts.evaluate.compare --plot-only
```

配置为 `configs/evaluation/ber_comparison.json`，算法标识为 `banded_lmmse`，
参数为 `half_bandwidth=16`。缓存按数据内容、CSI 视图、算法依赖和数值环境校验。
每份报告包含 BER-SNR 与平均 NMSE-SNR。NMSE 使用完整参考的线性误差均值转 dB。
零误码保留数值零，在图上按半个误码的计数下限显示。

Single-tap LMMSE、Full-matrix LMMSE 及其旧实验入口的来源版本为 `a08592b`。
历史报告保留原始误码和名称，不转换成新的带状 LMMSE 结果。

## 当前 NCS1 重测结果（终点 -8.92 dB）

2026-11-27 按 NCS1 -5.37 → -8.92 dB 完成训练、验证、测试三份数据的元数据重校准和全量验收，
计算提交为 `28d4710`。三个划分均达到两位小数目标，原始生成身份、模型、观测和标签核查通过。
NOF1 四份 HDF5 及验收清单的哈希和修改时间保持一致，原测试报告哈希保持一致，沿用其已有结果和缓存。

NCS1 每个 SNR 完整测试 600 帧、2,457,600 bit，全部 16,800 个符号块重新计算，半带宽为 16：

| SNR / dB | 平均 NMSE / dB | 当前 BER | 前次 BER |
| ---: | ---: | ---: | ---: |
| 0 | -5.37 | 0.23264038 | 0.23264038 |
| 5 | -6.84 | 0.13388835 | 0.12773966 |
| 10 | -7.73 | 0.07300496 | 0.06366496 |
| 15 | -8.27 | 0.04546387 | 0.03511190 |
| 20 | -8.60 | 0.03480713 | 0.02435181 |
| 25 | -8.80 | 0.03039063 | 0.02015951 |
| 30 | -8.92 | 0.02836060 | 0.01831991 |

30 dB 时为 69,699 / 2,457,600 个误码，BER 相比前次增加约 54.81%。
0 dB 的目标和误码计数与前次完全相同。两次均使用完整参考 CSI 进行理想辅助补偿。
校准与正式测试约 9.8 分钟，服务器回归检查 102 项通过、2 项跳过。
完整误码缓存、三个划分的校准备份与身份、数据哈希和读取器矩阵抽查全部通过独立核查。

本次结果保存在 `outputs/experiment/dddd_assisted_b16_ncs1_892_20261127/`：
`comparison.png` 为本次 BER/NMSE，`ncs1_target_comparison.png` 和同名 CSV 为前后对比，
`reports.zip` 包含验收报告、完整指标、校准前元数据与可复核缓存。
默认双信道报告路径中另有带原始来源记录的 NOF1 既有结果，未对 NOF1 重校准或重新计算。

## 前次全量测试结果（NCS1 终点 -9.84 dB）

2026-11-27 按 NOF1 -11.41 → -15.77 dB、NCS1 -5.37 → -9.84 dB 完成六份数据的重校准与全量验收，计算提交为 `f49d73b`。
平均 NMSE 相对目标的最大偏差约 `1.78e-15 dB`，原始生成模型哈希保持不变。
逐帧参考、观测、标签和独立随机种子核查通过；参考辅助输入的批量矩阵与读取器矩阵抽查完全一致。

每信道每 SNR 测试 600 帧、2,457,600 bit，半带宽固定为 16。30 dB 结果为：

| 信道 | 平均 NMSE / dB | 误码数 / 总比特数 | BER |
| --- | ---: | --- | ---: |
| NOF1 | -15.77 | 21,276 / 2,457,600 | 0.00865723 |
| NCS1 | -9.84 | 45,023 / 2,457,600 | 0.01831991 |

以上使用完整参考 CSI 进行补偿，属于明确的参考辅助结果。
完整 SNR 曲线、CSV、校准前元数据、文件哈希、测试缓存和验收报告保存在
`outputs/experiment/dddd_assisted_b16_20261127/`；完整数据仍位于服务器的原正式数据目录。
本轮校准与测试约 20.5 分钟，服务器回归检查 101 项通过、2 项跳过。
