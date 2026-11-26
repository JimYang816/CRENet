# 带状 LMMSE 增量评测

`scripts.evaluate.compare` 对 `FourDataDataset` 测试划分所选的 CSI 视图运行 B=16 带状 LMMSE，
每帧四个数据符号分别评分，使用逐块校验缓存。
缓存身份包含数据文件哈希、CSI 视图、实际计算依赖、算法参数和数值环境。
默认视图为明确使用完整参考矩阵的 `reference_assisted`，原始 CSI 使用 `--input-view raw`。

`--force banded_lmmse` 或 `--force all` 重新计算当前算法；`--plot-only` 根据完整报告重画。
`--from-run` 可从已有带状 LMMSE 报告读取曲线并另设输出目录。
历史单抽头与完整矩阵报告不提供当前算法的误码，不能据其生成带状 LMMSE 曲线。

当前默认数据路径为 `data/processed/dddd_pilot224/{nof1,ncs1}_dddd/test.h5`。
完整数据留在服务器，指标与图表通过 SSH 回传本地。
