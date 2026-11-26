# 带状 LMMSE 评测

当前算法为 `src/algorithms/banded_lmmse/` 中的 Banded LMMSE，半带宽固定为 16。
先用完整输入 CSI 消除已知导频，再按实际 FFT 索引截断数据通道至 ±16，
最后使用全部接收行与完整热噪声协方差完成 LMMSE 求解。

令 D 为数据载波的实际 FFT 索引，使用 `r = y - H_est x_known` 消除导频，
构造 `A[i,j] = H_est[i,D[j]] * 1(abs(i-D[j]) <= 16)`。
在噪声白化之前截断通道，再求解
`x_hat = (A^H R^-1 A + variance I)^-1 A^H R^-1 r`。
其中 R 为单位 RF 噪声经过真实接收滤波后的完整协方差。
每个数据列最多保留 33 个接收载波，导频间隙也计入 FFT 索引距离。

正式数据由 `src.data.FourDataDataset` 读取，目录为
`data/processed/dddd_pilot224/{nof1,ncs1}_dddd/`。
默认数据视图 `reference_assisted` 使用完整参考 CSI；评测入口明确选择 `raw` 原始接收机 CSI。
每帧包含四个数据符号，正式测试划分每个 SNR 有 600 帧。

已有正式测试数据的增量评测使用：

```bash
python -m scripts.evaluate.compare --device cuda --threads 4
python -m scripts.evaluate.compare --force banded_lmmse
python -m scripts.evaluate.compare --plot-only
```

配置为 `configs/evaluation/ber_comparison.json`，算法标识为 `banded_lmmse`，
参数为 `half_bandwidth=16`。缓存按数据内容、CSI 视图、算法依赖和数值环境校验。
每份报告包含 BER-SNR 与平均 NMSE-SNR。NMSE 使用完整参考的线性误差均值转 dB。
零误码保留数值零，在图上按半个误码的计数下限显示。

Single-tap LMMSE、Full-matrix LMMSE 及其旧实验入口的来源版本为 `a08592b`。
历史报告保留原始误码和名称，不转换成新的带状 LMMSE 结果。
