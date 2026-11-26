# CRENet 恢复与 NOF1 训练、双信道测试

恢复提交 `38c1b00` 的八层 CRENet：共享局部 CSI 修正器、独立残差 MMSE 展开层、
半带宽 16 的非循环修正、局部 QPSK 校准；网络计算使用完整初始矩阵、complex64 和对角噪声权重。
本次保持原网络结构及三阶段训练方法，从随机初始化开始，不加载历史权重。

## 当前数据和划分

使用 `data/processed/dddd_pilot224/{nof1,ncs1}_dddd/` 的正式数据，
D-D-D-D、FFT=1024、CP=480（80 ms）、512 数据子载波、224 导频、Q=24、192 个冻结物理时延位置。
默认输入为 `reference_assisted`，依赖完整参考矩阵，属于理想辅助数据。
NOF1 输入平均 NMSE 为 −11.41→−15.77 dB；NCS1 为 −5.37→−8.92 dB。

只使用 NOF1 的 3,675 帧训练和 525 帧验证（每帧四符号），全部 7 个 SNR 混合训练。
NOF1 验证集用于早停与权重选择；NCS1 不参与训练、验证、选模或超参数调整。
选择完成后，冻结同一个 NOF1 模型，分别评测 NOF1、NCS1 的全部 4,200 帧测试数据。
每信道每 SNR 为 600 帧、2,400 个符号块、2,457,600 bit。

## 训练和对照

沿用 A 均衡器训练、B 冻结均衡器训练 CSI 修正器、C 联合微调。
每阶段最多 10 轮，连续 4 轮验证未改善早停，阶段初始权重也参与选择。
A 按验证 BER、B 按完整矩阵 NMSE、C 按各 SNR CSI 不退化约束下的 BER 选取权重。
保留原训练种子 20261125、batch=32 和学习率设置。

测试对照为当前 B=16 Banded LMMSE 和完美 CSI B=16 Banded LMMSE。
两种解析对照按实际 FFT 索引截断数据通道，使用完整 CSI 消除已知导频，
再使用完整有色热噪声协方差、complex128 求解。CRENet 的网络内部计算保持原版。
BER 共用相同观测和符号；NMSE 先平均逐符号完整矩阵的线性误差比，再换算为 dB，包含带外误差。

## 运行与复核

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/fay/anaconda3/envs/jimyang/bin/python -m scripts.run.four_data_experiment --device cuda --threads 4
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/fay/anaconda3/envs/jimyang/bin/python -m scripts.evaluate.four_data_crenet_verify --device cuda
```

CUDA 不可用时入口自动回退 CPU。输出默认为
`outputs/experiment/four_data_crenet_nof1_pilot224_20261127/`。
`training/nof1/` 保存唯一模型、三个阶段的最佳权重、恢复状态和逐轮训练日志；
`nof1/`、`ncs1/` 分别保存测试图表、逐帧计数、完整矩阵 NMSE 和独立核验。
根层保存双信道 BER-SNR、平均 NMSE-SNR 的 PNG/PDF、CSV、共同检查点哈希和完成状态。

入口核对正式数据文件 SHA-256、当前帧配置、划分记录互斥性和共用实际 DPSS 基。
恢复运行必须匹配数据校准身份、配置和源码；测试完成后检查模型文件未发生变化。
独立核验逐帧计数、全部符号覆盖、线性 NMSE 汇总，以及每个 SNR 四符号的时延域 B16 数值解。
小样本检查使用 `--frame-limit 1 --stage-epochs 1 1 1 --output outputs/smoke/four_data_crenet_nof1`，
这类输出明确标记为非正式结果。

## 2026-11-27 正式结果

计算提交为 `b478603`，算法目录与归档提交 `38c1b00` 无差异。
服务器使用 RTX 4080、PyTorch 2.5.1+cu121、CUDA 12.1，NOF1 从随机初始化训练，
A/B/C 实际运行 4/10/4 轮。A 选中初始权重，B 选中第 10 轮；
C 连续四轮未超过初始候选，因此最终权重等于 B 第 10 轮，保存为 C 的 epoch 0。
训练约 24.0 分钟，两个信道测试各约 5.7 分钟，含数据核对、缓存与独立核验的总时间为 37.1 分钟。

两信道共测试 33,600 个符号块、34,406,400 bit；每信道每 SNR 为 2,457,600 bit。
同一个 NOF1 模型分别测试两个信道，30 dB 结果如下：

| 信道 | B16 BER | CRENet BER | CRENet 误码数 | 相对 B16 的 BER 变化 | 输入 NMSE / dB | 修正 NMSE / dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NOF1 | 0.00865723 | 0.00368693 | 9,061 | 降低 57.41% | -15.77 | -20.2525 |
| NCS1 | 0.02836060 | 0.03676717 | 90,359 | 增加 29.64% | -8.92 | -8.4214 |

NOF1 全部七个 SNR 的 BER 降低 1.86%–57.41%，CSI NMSE 改善 1.59–4.48 dB。
NCS1 全部七个 SNR 的 BER 增加 1.19%–29.64%，CSI NMSE 退化 0.024–0.644 dB。
本轮结果显示该 NOF1 单信道训练模型在当前 NCS1 数据上出现跨信道泛化退化。
NCS1 测试指标未用于重新训练、调整超参数或更换检查点；完整结果按实测值保留。

本地 CPU 检查 121 项通过、1 项跳过，服务器 122 项通过；
当前真实 1024 点符号的本地 CPU A/B/C 前向和梯度检查、服务器单帧三阶段训练与双信道测试通过。
正式结果的完整误码计数、线性 NMSE 汇总和测试覆盖通过独立核验，
两信道全部 33,600 个 B16 符号误码计数与此前各自的正式缓存逐一相同。
每个 SNR 四个符号的独立时延域 B16 求解相对差异分别约为 `1.61e-15`、`2.19e-15`。

两份测试报告的共同模型 SHA-256 为
`8b1b05e3adecaec86a9fd51bb5cb2d397cfdeb3c7fbc7565e6ecc6b9be9e87b7`。
模型、图表、逐帧结果和来源记录已回传本地同名实验目录，46 个产物（含清单）共约 5.70 MiB 压缩包；
45 项文件 SHA-256 与服务器清单全部一致，本地另有 `local_transfer_verification.json`。
`baseline_regression.json` 保存与既有正式 B16 缓存的逐符号比对，
`training_selection_verification.json` 保存最终模型与 B 第 10 轮的逐张量核对。
