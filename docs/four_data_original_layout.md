# 恢复原载波掩码后的12 Hz、无P、4D实验

本实验按用户最新要求固定原始载波分布，禁止以增加导频、去除边缘空载波或错位四D掩码来改善性能。
当前默认入口为`python -m scripts.evaluate.four_data_covariance`；历史全频带入口只能重画已完成结果。

## 不变的发射与观测协议

FFT1024，CP384，FFT提前32点，12 Hz半带宽，D1–D2–D3–D4。
每D为128导频、256保护、512数据，两侧各64个固定空载波；中间896载波重复`DDZPZDD`128次。
四D使用相同载波掩码；导频和数据QPSK能量均为1，无额外发射增益。
掩码、公开导频的值与幅度、完整512符号数据载荷均有运行时和测试约束。

测试直接使用原Q24/L176的`receiver_estimates.h5`中的干净观测、标准实RF噪声、RF方差和全部载荷，
不使用任何全频带实验观测。每帧重新生成原始波形，核对全部比特和种子，并记录原波形实际RF能量。
因此三个比较方案具有完全相同的发射载波、数据速率、实际波形、功率、接收观测和噪声。
每信道30帧、每SNR122880比特，SNR为0:5:30 dB，均评测单抽头及完整矩阵LMMSE。

## 表示、先验与欠定性

此次只恢复载波协议，其余表示参数保持当前值：NOF1 Q_eff=5、L=512；NCS1 Q_eff=12、L=224。
共同原始DPSS字典仍为12 Hz、Q27，各信道统计时间旋转保持冻结。
时间函数和系数仅从独立物理信道投影设计缓存读取，不继承任何全频带波形或掩码。

原掩码选取每D的128导频和256保护，共1536观测。
NOF1有2560个系数，NCS1有2688个系数，均大于1536，因此纯导频线性系统欠定。
原Q24/L176的4224个系数同样欠定。此处运行带独立先验的贝叶斯LMMSE，不宣称无先验可辨识或满列秩。
没有为了保留先前的满秩结论再次改变载波，也没有悄悄调整Q_eff或L。

三种对照为：

- 原Q24/L176结构先验，复用原始正式结果；
- 当前Q_eff/L结构先验，在原掩码下重新计算全部CSI与两种BER；
- 当前Q_eff/L联合协方差修正，在原掩码下重新标定、选择参数并计算全部CSI与两种BER。

后两者的表示和先验来源相同，只比较协方差建模；前者与后两者的差异包含时间子空间和延迟支撑。
标定仍为051–055各2/14秒，验证为056–060各2/14秒。
只在独立验证上从预设alpha∈{0.1,0.4,0.7,1.0}、rho∈{0.05,0.2,0.5,0.8}选择平均完整参考NMSE最小者。
本次重新标定原掩码的导频残差、完整数据干扰及交叉/跨D矩，不沿用全频带实验的参数选择结果。
测试CSI仅用于计算指标；先验及验证接口限制可读记录，估计器在读测试观测前序列化冻结并校验哈希。

## 运行与留存

```bash
python -m pytest -q outputs/smoke/data_generation_tests/test_four_data_original.py \
  outputs/smoke/data_generation_tests/test_four_data_covariance.py \
  outputs/smoke/data_generation_tests/test_four_data.py \
  outputs/smoke/data_generation_tests/test_four_data_rotated.py \
  outputs/smoke/tests/evaluation/test_four_data_design.py \
  outputs/smoke/tests/evaluation/test_four_data_rotated_audit.py
python -m scripts.evaluate.four_data_covariance --device cuda --threads 4
python -m scripts.evaluate.four_data_covariance --report-only
```

默认独立时间基来源`outputs/experiment/four_data_12hz_design_20261119`，
原始波形缓存来自`outputs/experiment/four_data_12hz_q24_l176_20261119`，
新结果写入`outputs/experiment/four_data_original_layout_joint_20261119`。
本地只做CPU smoke，并使用对应smoke设计及原始波形缓存。
完整HDF5留服务器；图、精确CSV、逐帧JSON、参数选择、标定和来源记录带回本地。
历史全频带结果保留为归档，不与本轮原载波条件的曲线混合绘制。
