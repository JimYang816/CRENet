# 四符号双信道 CRENet 训练与完整测试

2026-11-26 起默认 CP 为 **80 ms / 480 点**。本页记录的是 CP=256 的历史训练与结果；
对应训练、验证、测试数据及共享模型已按用户要求删除，历史权重和结果仅供来源复核。
正式 CP80 补偿数据已生成并全量验收，新的训练输出为 `outputs/experiment/four_data_crenet_cp80_20261126/`，
训练入口会拒绝旧 CP=256 数据。当前设置见 [CP80 实验](four_data_cp80.md)。

本页历史运行使用已验收的 `data/processed/{nof1,ncs1}_dddd/`，NOF1、NCS1 分别从头训练，
不读取旧 P-D-D-P 权重。入口为 `scripts/run/four_data_experiment.py`，
配置为 `configs/training/four_data_crenet.json`。README 不作修改。

## 数据、训练和选择

沿用 D-D-D-D、FFT=1024、CP=256、512 数据与 256 梳状导频、12 Hz/Q=24，
保留各信道的 192 个真实物理时延位置。四个符号共享网络参数，各自处理；
每轮随机打乱全部 SNR/帧/符号组合，不丢弃最后不足一个 batch 的样本。
训练/验证分别为 3,675/525 帧，即每轮 14,700/2,100 个符号块。
运行前核对四个 HDF5 文件的验收 SHA-256 和训练、验证、测试记录互斥性。

输入视图为 `reference_assisted`：NOF1 平均输入 NMSE 为 -9.69 至 -16.05 dB，
NCS1 为 -5.24 至 -9.13 dB。该视图使用完整参考矩阵，属于理想辅助数据，
不能解释为无参考接收机估计性能。三种均衡算法使用相同的这一输入 CSI。

CRENet 沿用八层共享 CSI 修正与独立残差 MMSE 展开设计，使用完整基础矩阵、
半带宽 16 的非循环修正和局部 QPSK 校准，complex64、对角噪声权重。
权重随机初始化，A 阶段训练均衡器，B 阶段冻结均衡器训练 CSI 修正，C 阶段联合微调。
每阶段最多 10 轮，验证选择连续 4 轮未改善时提前结束。
A 按验证 BER 选择，B 按验证完整矩阵 NMSE 选择；C 优先选择所有 SNR 修正 NMSE
均不退化且平均 NMSE 改善的模型，再按 BER、NMSE 排序。阶段初始模型参与选择。
测试标签和参考不参与参数更新、超参数调整、早停或检查点选择。

参考矩阵仅作为已声明的辅助视图构造来源和独立监督/评分对象；模型 forward
只接收观测、初始 CSI、公开导频、载波位置与掩码、噪声参数。
每阶段选中权重、逐轮日志及可恢复的优化器与随机状态均保留。

## 测试与指标

两种 LMMSE 是解析算法，无须训练。测试全部 4,200 帧，每 SNR 为 600 帧、
2,400 个符号块、2,457,600 bit。四条 BER 曲线分别为 Single-tap LMMSE、
Full-matrix LMMSE、CRENet、完美 CSI 的 Full-matrix LMMSE。
单抽头使用原始数据载波观测；完整矩阵使用全部接收行、全部数据列、已知导频消除、
完整有色噪声协方差与热噪声正则项，以 complex128 求解。
完美 CSI 对照共用相同发射数据、接收观测和噪声，仅替换 CSI。
这里的完美 CSI 是当前 OFDM 符号的完整频域矩阵；原始波形中的跨符号干扰仍然保留，
该对照不等同于包含全部跨符号耦合的整帧理想接收机，具体复核见文末。

CSI 指标计算每帧每符号完整矩阵的线性 NMSE，先平均后取 dB。
CRENet 输出最后一层修正 CSI 的 NMSE，同时保留初始 NMSE 和逐 SNR 改善 dB。
所有未修正带外位置仍进入误差分子和参考分母。
零误码在表格中保留为零，BER 图在半个误码计数位置用空心点标记。

输出为 `outputs/experiment/four_data_crenet_20261125/`，包含双信道及单信道的
`ber_snr.png/pdf`、`nmse_snr.png/pdf`、`summary.csv`；信道目录另有 `model.pt`、
三阶段检查点、`training.json`、`history.json`、`metrics.json`、`per_frame.csv`、
`frame_metrics.npz` 和 `provenance.json`。大数据和实验结果不进入 Git。

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/fay/anaconda3/envs/jimyang/bin/python -m scripts.run.four_data_experiment --device cuda --threads 4
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/fay/anaconda3/envs/jimyang/bin/python -m scripts.evaluate.four_data_crenet_verify --device cuda
```

上述命令可按原目录恢复，身份不一致则拒绝复用。真实数据小样本检查使用
`--frame-limit 2 --stage-epochs 1 1 1 --output outputs/smoke/four_data_crenet_server`，
明确标记为非正式结果。独立核验入口检查全部误码计数、NMSE 聚合及测试符号完整性，
并在每个 SNR 的四个符号上与已有时域坐标 LMMSE 实现核对输出和误码数。

本地使用 CPU，运行新增流程、完整矩阵 NMSE 梯度、有色协方差解析解、
四符号数据排列、相对输出路径与既有数据协议检查，共 27 项通过。Windows Anaconda 数值库检查
使用 `MKL_THREADING_LAYER=SEQUENTIAL` 避免重复 OpenMP 运行库；服务器使用标准库线程设置。

## 2026-11-25 正式运行记录

计算代码为 `7b008cc`，服务器 RTX 4080、PyTorch 2.5.1+cu121，训练 batch=32。
本地 27 项独立检查及服务器 27 项检查通过；双信道真实数据小样本中，新的频域完整
LMMSE 与既有独立时域实现的相对符号差异约为 `1.9e-13`，四个符号和全部 SNR 的误码计数一致。

NOF1 实际完成 A/B/C 阶段 4/10/4 轮。A 阶段保留物理初始化；B 阶段选中第 10 轮，
C 阶段连续 4 轮未超过其初始候选，因此最终使用 B 第 10 轮权重（保存为 C 的 epoch 0）。
检查点选择不使用测试指标。全量测试每 SNR 为 2,457,600 bit，30 dB 结果为：

| 信道 | Single-tap BER | Full-matrix BER | CRENet BER | 完美 CSI Full-matrix BER | 输入 NMSE | CRENet 修正 NMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NOF1 | 0.01670410 | 0.00718140 | 0.00277995 | 0.00113403 | -16.0500 dB | -18.3227 dB |
| NCS1 | 0.13931437 | 0.02547323 | 0.00693604 | 0（0/2,457,600 bit） | -9.1300 dB | -12.7587 dB |

NOF1 的 7 个 SNR 测试 NMSE 均改善，改善幅度为 1.0391–2.3234 dB。
完整输入 NMSE 与正式验收目标最大偏差约 `6.8e-10 dB`。
全部逐帧误码和完整矩阵 NMSE 汇总复核通过；正式样本上的独立 LMMSE 数值及误码核对通过。
模型、图表、指标表和逐帧结果已带回本地同名实验目录，模型 SHA-256 与评分记录一致。

NCS1 实际完成 A/B/C 阶段 4/10/10 轮，A 阶段同样保留物理初始化，最终选中 C 第 10 轮。
其 7 个 SNR 测试 NMSE 全部改善，改善幅度为 1.2158–3.6287 dB；输入 NMSE 与正式验收目标
最大偏差约 `2.2e-10 dB`。完美 CSI 的 30 dB 点是有限样本的零实测误码，图中在半误码计数
位置以空心点显示，CSV 仍保存真实零值。

两信道均使用验证集选权重，测试数据未参与训练或检查点选择。所有 33,600 个测试符号块已评分，
共 34,406,400 bit。逐帧误码、线性 NMSE 聚合、完整测试覆盖和独立时域 LMMSE 核验全部通过；
独立 LMMSE 相对符号差异分别约 `1.9e-13`、`1.7e-13`。最终图表在实验目录根层及各信道目录
同时提供 PNG/PDF，全部 14 个信道/SNR 点的数值见根层 `summary.csv`。

NOF1 训练与评分分别耗时约 24.0、5.7 分钟；NCS1 分别约 33.7、5.7 分钟，
上述训练计时不含参考矩阵 RAM 缓存准备。主实验进程已正常结束，GPU 已释放。

## 完美 CSI 参考的跨符号干扰复核

对 30 dB 全部 600 帧、每信道 2,457,600 bit 的控制实验确认：NOF1 高 SNR 误码主要
来自当前单符号接收模型未包含的跨符号干扰。`BlockedReceiver.reference` 仅将当前符号的
CP/有用窗输入列折叠到 1024×1024 频域矩阵，窗外耦合记入 `intersymbol_leakage`，
并未进入该均衡矩阵。数据配置 `model_mismatch_policy=diagnose` 保留了这些物理残差。

CP=256、基带 6 kHz 时前缀为 42.67 ms；考虑 FFT 提前 32 点，有效覆盖约 37.33 ms。
冻结选点中的最晚物理时延按 `(index-32)/6000` 计算，NOF1 为 78.17 ms，NCS1 为 31.83 ms。
前者存在越过 CP 的拖尾；这里的 192 是选中位置数量，并不意味着最大物理时延为 192 点。

测试集平均干净波形残差功率比为 NOF1 -23.329 dB、NCS1 -86.094 dB。
NOF1 第一个符号残差约 -87.26 dB，后三个约 -22.13、-22.10、-22.01 dB；
30 dB 完美单符号 CSI 的逐符号误码为 13、1023、890、861。
第一符号之前为静默保护区，后三个符号存在前一符号拖尾，分布与跨符号干扰一致。

| 30 dB 控制条件 | NOF1 误码数 | NCS1 误码数 |
| --- | ---: | ---: |
| 正式原始接收波形、完美单符号 CSI | 2787 | 0 |
| 保留相同噪声，用真值移除模型残差 | 47 | 0 |
| 仅移除热噪声，保留物理模型残差 | 2436 | 0 |
| 同时移除热噪声与模型残差 | 0 | 0 |

四组控制共用同一参考矩阵、原 30 dB RF 方差及 LMMSE 正则项；残差相减使用参考真值，
仅用于归因诊断，不能作为实际接收机算法或替换正式 BER 结果。两个信道全部符号的正式误码
均被独立复现；参考矩阵对 `clean-model_residual` 的最大相对平方预测误差小于 `8e-31`。
诊断结果保存于 `outputs/smoke/four_data_perfect_csi_audit_20261125/`。

因此，当前完美 CSI 曲线需按“完美单符号 CSI、保留实际跨符号干扰”解释。
它不能用于给两种信道的固有难度排序，也不能作为无跨符号失配的整帧性能上界。
若要构造后者，需要覆盖充分时延的 CP 或包含跨符号耦合的联合均衡模型，并重新评测。
