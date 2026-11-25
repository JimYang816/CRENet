# 三算法统一评测

项目仅保留以下三种均衡算法：

| 算法 | 实现 | CSI 处理 |
|---|---|---|
| Single-tap LMMSE | `src/algorithms/single_tap_mmse/equalizer.py` | 固定输入 CSI，每个数据载波独立检测 |
| Full-matrix LMMSE | `src/algorithms/full_matrix_lmmse/equalizer.py` | 固定输入 CSI，使用全部接收行、数据列和完整热噪声协方差 |
| CRENet | `src/algorithms/crenet/` | 共享 CSI 修正器与默认 8 层残差 MMSE 展开联合检测 |

## 当前数据与接入状态

当前正式数据使用 D-D-D-D 帧与 192 个估计时延抽头，保存在服务器
`data/processed/{nof1,ncs1}_dddd/`，由 `src.data.FourDataDataset` 读取。
默认 `reference_assisted` 视图使用完整参考矩阵构造补偿 CSI；`raw` 视图保留原始接收机估计。
协议、目标曲线和验收结果见 [四符号数据说明](four_data_lmmse.md)。

两种 LMMSE 已通过四符号小实验和正式数据验收中的 BER 诊断。
CRENet 保留模型、训练、消融与评测实现，但这些学习流程仍使用旧两符号读取器，
须适配 `FourDataDataset`、四个数据符号及新的输入视图后，才能用于本批正式数据。
旧数据的权重、缓存和误码计数不能直接作为当前三算法的新结果。

## 入口与配置

统一入口为 `scripts.evaluate.compare`，配置为 `configs/evaluation/ber_comparison.json`。
传统算法标识只有 `single_tap_mmse` 和 `full_matrix_lmmse`，均无额外检测参数。
`checkpoints` 仅接受每信道的 `crenet` 检查点；当前配置留空，尚未绑定新帧权重。
不能仅把旧入口的数据路径指向四符号文件就开始正式评测。

以下命令描述保留的入口功能，计算命令仍受上述数据协议限制：

```bash
# 仅两种 LMMSE；不会训练模型
python -m scripts.evaluate.compare --baselines-only --device cuda --threads 4
# 已完成协议适配并绑定相符 CRENet 权重后，评测三种算法
python -m scripts.evaluate.compare --device cuda --threads 4
# 从已保存的完整报告重新绘图，不读取数据或运行检测
python -m scripts.evaluate.compare --plot-only
```

`--force` 只接受 `single_tap_mmse`、`full_matrix_lmmse`、`crenet` 和 `all`。
完整 CRENet 流水线 `scripts.run.crenet_experiment` 只训练 CRENet 的共同预训练、
固定 CSI 与联合分支，然后评测两种 LMMSE 和 CRENet。
效率入口 `scripts.evaluate.crenet_efficiency` 只接受一个 CRENet 检查点，默认测量三种算法。

## 指标与来源

所有方法必须使用同一数据划分、SNR 网格、接收观测、初始 CSI 与 QPSK 映射。
每次报告至少包含 BER-SNR 和平均 NMSE-SNR；NMSE 先对完整矩阵的线性相对平方误差求均值，再转换为 dB。
两种固定 CSI 的 LMMSE 共用输入 NMSE，CRENet 另外报告修正后的完整矩阵 NMSE。
零误码的数值仍为零，对数图以 `0.5 / total_bits` 处的空心三角显示。

离线重画仅提取保留的三种算法，保留其原始误码计数、总比特数及来源；从旧报告导出时使用新目录。
历史实验结果、检查点和配置备份保留为来源记录，不能解释为当前新帧的计算结果。
缓存依照实际的数据、输入视图、算法依赖、权重和数值环境校验，不跨协议复用。
