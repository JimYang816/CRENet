# 固定 12 Hz、Q24、L176 的无 P 四数据符号小实验

这是独立诊断实验：四个 OFDM 符号均为 D，使用本次接收机直接估计的 CSI。
不修改正式 PDDP 配置、已验收数据、compensated CSI 目标或主对比结果。

## 固定设置与指标

- 结构为 D1–D2–D3–D4，没有整符号训练 P。沿用 FFT1024、CP384、提前32点、6/96 kHz 采样及14 kHz载频。
- 每个 D 保留512个未知 QPSK 数据载波、128个公开导频及256个保护载波；64个边缘空载波/侧。
- DPSS 覆盖包含四个 CP 的5632个时域样本，固定**半带宽12 Hz、Q=24、L=176**。
  Q 直接指定，不继续按旧的1e-5尾比选择；实际尾比保存到结果。12 Hz 是估计基的半带宽，不改变 WATERMARK 实测信道。
- 冻结先验从051–055记录的2、14秒窗口拟合，共10个窗口；延迟协方差块宽32，正则化1e-6。
- 四个 D 的导频及保护载波联合形成1536维估计观测，未知数据引起的 ICI 作为有色干扰进入协方差。
  不接收标签，不迭代反馈，不做理想 CSI 补偿，也不使用 P1/P2 功率误差下限。
- NOF1、NCS1 分别测试既定测试记录011/015/019/023/027/031/035/039/043/047，
  每条取2、14、26秒，共30帧，每 SNR 为120个 D、122880比特。每窗一次独立载荷/噪声，跨 SNR 共用标准噪声。
- SNR 为0、5、10、15、20、25、30 dB。以全部四个干净接收 FFT 窗的平均功率定义 SNR，
  使用已知仿真 RF 方差及接收滤波后的有色噪声协方差。
- 单抽头 MMSE 使用数据载波上的原始接收值及 CSI 对角线；完整矩阵 MMSE 使用全部接收频点，
  先消除公开导频贡献，再联合求解512个数据符号。复用现有均衡器实现。
- 两种均衡器共用相同 CSI，因此每个信道仅有一条 NMSE 曲线。NMSE 对**完整1024×1024参考信道**计算，
  包含 L=176 之外的遗漏能量；先平均所有帧、所有 D 的线性相对平方误差，再转为 dB。
- 实际96 kHz实带通回放保留信道长尾和符号间泄漏；估计器及均衡器不读取参考 CSI。
  完整参考 CSI 仅用于计算 NMSE 和物理诊断。

## 运行

本地 CPU 小样本（每信道两个先验窗口、一帧测试，保留全部七个 SNR）：

```powershell
python -m pytest -q outputs/smoke/data_generation_tests/test_four_data.py
python -m scripts.evaluate.four_data_mmse --smoke --device cpu --channels NCS1 --threads 4
```

服务器完整小实验：

```bash
cd /home/fay/JimYang/CRENet
git pull --ff-only
/home/fay/anaconda3/envs/jimyang/bin/python -m scripts.evaluate.four_data_mmse \
  --device cuda --threads 4 \
  --output outputs/experiment/four_data_12hz_q24_l176_20261119
```

不传 `--device` 时，Windows 使用 CPU，服务器在 CUDA 可用时使用 GPU。
全部计算使用 complex128。CPU 的特征分解使用 EVR，以规避已复现的本机 LAPACK EVD 复数特征向量精度问题。

输出包含汇总 `metrics.json`、`metrics.csv`、`comparison.png/pdf`、`ber_snr.png/pdf`、`nmse_snr.png/pdf`，
各信道保留逐帧误码计数、NMSE、符号间泄漏、支撑外能量及物理模型残差。
`frozen_prior.h5` 和 `receiver_estimates.h5` 保存冻结先验、逐帧估计系数、公开信号、干净观测、标准噪声及标签，供复核。
结果按窗口断点续跑，代码、配置、源包或运行环境身份变化时拒绝混用已有目录。

独立测试将联合估计与显式构造的稠密 LMMSE 比较，将分块回放及参考核与完整物理算子比较，
并检查所有四个 D、隐藏数据、支撑外 NMSE 能量和完整矩阵 MMSE 的解调结果。
小样本曲线用于观察该固定设置，不替代已验收数据上的正式主对比。
