# 指标图绘制

绘图统一使用 Python。BER 图显示 Single-tap LMMSE 和 Full-matrix LMMSE，
NMSE 图显示两种基线共享的输入 CSI。

已保存的完整比较报告可通过下列入口重新绘图，不运行训练或推理：

```bash
python -m scripts.evaluate.compare --plot-only
```

`configs/evaluation/ber_comparison.json` 的 `output` 指定报告目录。
从历史报告导出时，用 `--from-run` 指定来源，输出目录必须与来源不同；
保留方法的原始误码计数、比特数和来源继续保存，不能把旧协议结果当成当前四符号结果。
算法接入状态见 [统一评测](equalizer_comparison.md)。

本批正式四符号数据的平均 NMSE 使用全量验收统计单独绘制：

```bash
python -m scripts.evaluate.four_data_dataset_nmse
```

图保存在 `outputs/experiment/dddd_delay192_assisted_20261124/nmse_snr_combined.{png,pdf,svg}`。
默认参考辅助视图的训练、验证、测试分别校准，因此三条平均曲线重合。
NMSE 先按完整矩阵线性相对平方误差求均值，再转换为 dB。
零误码仍保存为零，对数 BER 图以 `0.5 / total_bits` 处的空心三角表示。

需要用于论文的图片，应在审查协议、数值与来源后，直接放在 `outputs/paper/main.tex` 同目录。
旧论文图与历史实验数值作为来源记录保留，不由清理算法的操作覆盖。
