# CP=80 ms 的双信道 LMMSE 小实验

入口为 `scripts/evaluate/four_data_cp80.py`，配置为
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
