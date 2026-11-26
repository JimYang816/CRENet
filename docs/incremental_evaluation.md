# LMMSE 增量评测

Single-tap LMMSE 和 Full-matrix LMMSE 直接使用接收机的输入 CSI 进行检测。
当前四符号数据和小实验见 [CP80 四符号协议](four_data_cp80.md)。

`scripts.evaluate.compare` 在 `CompactDataset` 的测试划分上对两种 LMMSE 使用逐块校验缓存。
缓存身份由数据、CSI 视图、数值环境和实际计算依赖决定，公共输入或评分代码变更时重新计算受影响的方法。

`--force` 可指定 `single_tap_mmse`、`full_matrix_lmmse` 或 `all`。
离线重画使用 `--plot-only`。从旧报告导出两条基线曲线时，以 `--from-run` 指定源报告目录，另设输出目录。

缓存内的计时保留原始测量来源，不构成新的效率实测。
完整数据留在服务器，图、指标和报告通过 SSH 回传本地。
