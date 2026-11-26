# LMMSE 基线评测

项目当前提供两种均衡算法。

| 算法 | 实现 | CSI 处理 |
|---|---|---|
| Single-tap LMMSE | `src/algorithms/single_tap_mmse/equalizer.py` | 固定输入 CSI，每个数据载波独立检测 |
| Full-matrix LMMSE | `src/algorithms/full_matrix_lmmse/equalizer.py` | 固定输入 CSI，使用全部接收行、数据列和完整热噪声协方差 |

## 当前数据与接入状态

当前正式数据使用 D-D-D-D 帧与 192 个估计时延抽头，默认生成目录为
`data/processed/dddd_pilot224/{nof1,ncs1}_dddd/`，由 `src.data.FourDataDataset` 读取。
默认 `reference_assisted` 视图使用完整参考矩阵构造补偿 CSI；`raw` 视图保留原始接收机估计。
协议、目标曲线和验收结果见 [CP80 四符号数据说明](four_data_cp80.md)。

CP80 四符号小实验用两种 LMMSE 对比原始估计和完美 CSI；
正式数据验收对原始与参考辅助 CSI 抽样统计两种 LMMSE 的 BER，并汇总各划分的平均 NMSE。

## 入口与配置

`scripts.evaluate.four_data_cp80` 运行当前四符号链路的小样本 LMMSE 对比。
`scripts.evaluate.compare` 读取旧版 `CompactDataset` 的原始 CSI 测试划分，
配置为 `configs/evaluation/ber_comparison.json`。算法标识为 `single_tap_mmse` 和 `full_matrix_lmmse`。

```bash
# 四符号链路小样本
python -m scripts.evaluate.four_data_cp80 --smoke --device cpu
# 旧版 CompactDataset 测试划分
python -m scripts.evaluate.compare --device cuda --threads 4
# 从已保存的报告重画基线
python -m scripts.evaluate.compare --plot-only
```

`--force` 接受 `single_tap_mmse`、`full_matrix_lmmse` 和 `all`。
传入含旧检查点的配置进行计算时，使用 `--baselines-only` 运行两种 LMMSE。

## 指标与来源

所有方法必须使用同一数据划分、SNR 网格、接收观测、初始 CSI 与 QPSK 映射。
每次报告至少包含 BER-SNR 和平均 NMSE-SNR；NMSE 先对完整矩阵的线性相对平方误差求均值，再转换为 dB。
两种 LMMSE 共用输入 CSI 和对应的平均 NMSE。
零误码的数值仍为零，对数图以 `0.5 / total_bits` 处的空心三角显示。

从旧报告导出两条基线曲线时，以 `--from-run` 指定来源，在新目录保留原始误码计数、总比特数及来源。
旧版 CRENet 的代码与文档可在提交 `38c1b00` 中查阅。历史实验结果、检查点和配置备份用于复核当时的实验。
缓存按数据、输入视图、算法依赖和数值环境校验。
