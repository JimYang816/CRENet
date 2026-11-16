# ICC 论文结果绘图（MATLAB）

当前工作区保留 `plot_nof1_ncs1_ber.m`、`plot_nof1_ncs1_nmse.m` 和 `export_icc_metric_pdfs.m`；下文原 `plot_icc_results` 及其消融表入口已不在当前工作区，相关用法仅作历史记录。现有脚本读取已保存报告，并兼容报告中的旧算法标识。

在项目根目录的 MATLAB 命令窗口运行：

```matlab
addpath('scripts/visualize');
[figures, summary] = plot_icc_results;
```

只显示四个独立窗口，不自动导出、保存或修改论文：

- `figures.ber_nof1`：NOF1 的 BER–SNR，纵轴为对数坐标。
- `figures.ber_ncs1`：NCS1 的 BER–SNR。两个 BER 图分别根据各自数据确定纵轴范围。
- `figures.nmse`：同一坐标轴内的 NOF1/NCS1 初始及修正后 CSI，共四条 NMSE–SNR 曲线。
- `figures.ablation`：单栏 A–D 表格，分别列出两个信道的平均及 30 dB BER、CSI NMSE。

图窗默认宽度为 3.5 inch；两个 BER 图的尺寸可在脚本的 `style.berSizes` 中分别调整，不要求一致。使用 Times New Roman，刻度、图例及表格为 9 pt，坐标标签为 10 pt。颜色、线型与标记共同区分曲线；消融表按各指标独立加粗最低值，包括并列值。图例中的 CRENet 对应报告中的 `crenet`，与当前论文名称一致。

默认主对比从 `configs/evaluation/ber_comparison.json` 的 `output` 读取，当前包含六种算法。正式消融从 `outputs/runs/icr_ad_current_20261031_133445/test/{nof1,ncs1}/comparison.json` 读取，不混用旧消融。也可显式指定两个结果根目录：

```matlab
[figures, summary] = plot_icc_results(mainRun, ablationRun);
```

输入须为完整的 compensated CSI 测试报告，SNR 网格为 `0:5:30` dB。脚本校验 BER 与误码计数、NMSE 线性值与 dB 值，以及输入 CSI 的验收端点。只读取已有结果，不运行训练和推理。

“平均”表示全部 7 个 SNR 点等权统计，不是额外的 15 dB 测量：BER 对逐 SNR BER 求算术平均；CSI NMSE 对逐 SNR 的线性均值再次求平均后转换为 dB，不能直接平均 dB 值。30 dB 列直接读取该 SNR 的测量值。`summary` 保留未舍入的 MATLAB 数值表；图中 BER 显示四位有效数字，NMSE 保留两位小数。

若出现零误码，原始 BER 仍为零；对数图仅在 `0.5/total_bits` 处用空心倒三角表示，并中断该点的连线。

样式依据：[IEEE 图形及字体指南](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/file-formatting/)、[IEEE 线图可读性建议](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/)。3.5 inch 和具体线型是本项目为 IEEE 单栏排版选定的绘图设置。
