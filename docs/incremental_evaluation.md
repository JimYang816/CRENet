# 独立训练与增量评测

当前只比较 Single-tap LMMSE、Full-matrix LMMSE 和 CRENet。
两种 LMMSE 无需训练；唯一学习模型为 CRENet。
四符号正式数据已生成，CRENet 的旧训练和评测入口仍须适配，状态见
[统一评测](equalizer_comparison.md) 和 [四符号协议](four_data_lmmse.md)。

`scripts.evaluate.compare` 只评测或绘图，不会自动调用训练。
`configs/evaluation/ber_comparison.json` 中的 `checkpoints` 仅支持 `crenet`；
更换模型时应绑定新实验的权重，并验证相同的数据、输入视图、模型与代码身份。
完整训练入口 `scripts.run.crenet_experiment` 仅训练 CRENet 的共同预训练、固定 CSI 与联合分支。

两种 LMMSE 使用逐块校验缓存；CRENet 完整报告缓存位于
`outputs/experiment/_cache/crenet_reports/`，校验数据、权重、推理与评分源码、bootstrap 设置和全量比特覆盖。
单独修改 CRENet 不要求重算两种 LMMSE；修改公共输入或评分代码时，受影响的方法须重新计算。
更换数据协议或 CSI 视图必须重新匹配身份，不能复用旧权重和结果。

`--force` 可同时指定 `single_tap_mmse`、`full_matrix_lmmse` 和 `crenet`，
`--force all` 刷新所有已配置方法。`--baselines-only` 只运行两种 LMMSE。
离线重画使用 `--plot-only`；需要保留旧报告时，使用 `--from-run` 读取历史目录并输出到不同目录。
绘图仅保留三种算法的曲线，原始历史结果不删除。

缓存内的计时保留原始测量来源，不构成新的效率实测。
完整数据和权重留在服务器，图、指标、训练记录和报告通过 SSH 回传本地。
