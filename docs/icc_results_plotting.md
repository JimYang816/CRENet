# ICC 论文指标图绘制

> 本文记录旧 P-D-D-P / 参考辅助补偿条件下的设计或结果，仅供历史参考。当前实验使用 D-D-D-D 帧和未经补偿的 CSI；范围及运行方式见 [四数据符号 LMMSE 小实验](four_data_lmmse.md)。

绘图脚本读取 `configs/evaluation/ber_comparison.json` 的 `output` 指向的双信道七算法报告。当前结果位于 `outputs/experiment/equalizer_comparison_dd_ce_20261102/{nof1,ncs1}/metrics.json`；脚本只读取已有结果，不运行训练或推理。

## 查看图形

在项目根目录的 MATLAB 命令窗口运行：

```matlab
addpath('scripts/visualize');
plot_nof1_ncs1_ber;
plot_nof1_ncs1_nmse;
```

前者显示 NOF1 和 NCS1 各一张七算法 BER–SNR 图，后者在一张图中显示两信道的初始 CSI、DD-DPSS 修正 CSI 和 CRENet 修正 CSI 的 NMSE–SNR 曲线。这两个脚本只显示图形，不导出文件。

输入须为完整的 compensated CSI 测试报告，SNR 网格为 `0:5:30` dB。BER 脚本核对误码数、总比特数及 BER；NMSE 脚本核对数值完整性和输入 CSI 的验收端点。DD-DPSS 结果还通过同目录上级的 `verification.json` 或源码指纹验证。历史报告中的 `icr_mmse` 标识按 CRENet 读取。

NMSE 曲线为完整矩阵逐块线性 NMSE 的平均值，再转换为 dB。若出现零误码，原始 BER 仍为零；对数图在 `0.5/total_bits` 处用空心倒三角表示，并中断该点的连线。图形使用 3.5 × 2.8 英寸画布及 Arial 字体。

## 导出论文图

以下命令显式重新生成并覆盖 `outputs/paper/` 中的 `NOF1-BER.pdf`、`NCS1-BER.pdf` 和 `NMSE-SNR.pdf`：

```matlab
addpath('scripts/visualize');
export_icc_metric_pdfs;
```

论文正式 A–D 消融数值位于 `outputs/experiment/icr_ad_current_20261031_133445/test/{nof1,ncs1}/comparison.json` 和对应 CSV；当前三个指标图脚本不生成消融表。消融定义及结果见 [CRENet 说明](crenet.md#5-比较消融与输出)。
