# 四符号双信道 CRENet 训练与完整测试

本轮使用已验收的 `data/processed/{nof1,ncs1}_dddd/`，NOF1、NCS1 分别从头训练，
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
