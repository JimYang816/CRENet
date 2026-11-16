# 当前 CRENet 训练和推理性能口径

当前统一设计及运行入口见 [CRENet说明](crenet.md)，公式和历史冻结预算结果见 [算法设计](crenet_algorithm_design.md#38-历史全参数预算修复的冻结结果)。当前正式权重来自 `outputs/experiment/icr_full_20261031_095549/`；下方保留的预算表属于更早的预算修复实验，不能直接当作这些正式权重的实测耗时。历史batch扫描、缓存迁移和旧门控结果已清理，不再作为当前默认配置。

两信道均8层、完整物理信道作用、修正半带宽16、窗口32/中心16/隐藏48，均学习全部物理参数和局部网络。训练为 `complex64`、`dense_device`、batch 8、SNR交错、预取2批、8192 MiB有界RAM缓存；不启用AMP或缩减物理基。`dense_device`在设备上准备完整矩阵后转为模型精度，正式输入不投影回DPSS系数空间。

参考矩阵缓存按需建立、受主存上限约束，不写入数据或检查点。它只供独立监督与指标计算。续训仍检查数据、控制参数、优化器和源码身份。

```bash
python -m scripts.evaluate.crenet_training_perf --channel NOF1 --device cuda \
  --output outputs/experiment/crenet_training_performance_current/nof1 \
  --frames 128 --validation-frames 4 --repeats 3 --threads 4 \
  --reference-cache-mib 4096 --batch-sizes 2 4 8 16
```

入口从唯一配置解析对应信道的模型；测量后端、预取和缓存由该CLI参数决定。batch选择只读train/validation，不使用test选型。历史8层旧模型的吞吐量不可当作当前残差展开训练的实测速度。

历史预算复核采用RTX4080、complex64、batch2、4个CPU线程，计入输入重建、全部展开层及CSI修正；一次性加载和冻结准备不计入稳态逐块预算。另以11次轮换模型计时的中位数复核，不以训练吞吐量替代推理延迟。来源为已清理的 `icr_budgeted_unified_20261031` 及其核验复用的NCS1历史报告。

| 设置 | 原展开→预算修复版总耗时 ms/块 | 修复版/原预算 |
|---|---:|---:|
| NCS1固定CSI | 6.837→6.618 | 96.79% |
| NOF1固定CSI | 11.707→6.706 | 57.29% |
| NCS1迭代修正 | 9.091→8.875 | 97.63% |
| NOF1迭代修正 | 17.909→8.885 | 49.61% |

该历史复核中的NOF1与NCS1均执行局部神经网络。NCS1预算余量较小，其他设备或配置应重新测量；固定CSI的单次完整矩阵LMMSE仍快于8层展开。预算基准是原Unfolding。NOF1原22层、修复版8层，不能把该表当作等深度的正式A–D消融。

完整效率入口 `scripts.evaluate.crenet_efficiency` 默认测量七种算法，接受明确的数据、正式比较报告及三个学习检查点，每算法独立进程测量，并记录初始化、稳态、参数量和峰值内存；当前正式路径的完整命令见 [CRENet说明第6节](crenet.md#6-验证与运行)。性能实验至少输出BER–SNR及平均NMSE–SNR，原始报告和不满足预期的指标均保留。

## 冻结推理的等价优化

`prepare_frozen_inference` 在既有标量/窗口缓存上提供两项优化，训练模型、配置及检查点权重不变。默认启用残差复用；静态卷积缓存需显式设置 `cache_static_correction=True`，不根据其理论乘加节省宣称实测加速：

- 共享 CSI 修正器第一层卷积的 467 个通道中，168 个通道在单次 forward 内不变。其卷积贡献只计算一次，与每轮动态贡献相加后再激活。缓存每次 forward 都清空，不跨样本、SNR 或 D 块复用。
- CSI 修正后的残差由修正前残差减去实际带状增量作用得到。仍保留全部初始 CSI 耦合；均衡器重复执行且软状态已变时重新计算残差。

两项运算重排允许浮点舍入差异，不能声称逐位相同。调用 `prepare_frozen_inference(model, cache_static_correction=False, reuse_correction_residual=False)` 可保留优化前的冻结部署路径；切换选项须重新加载同一检查点。旧训练源码身份检查继续生效。

正式2026-10-31检查点还早于提交 `027ae0e`：该提交只向 `numerics.py` 新增自动选择设备的函数，未改变数值算子。兼容校验明确列举该版本的完整源码指纹及改名后指纹，不忽略不匹配项，也不改写历史检查点。

```bash
python -m scripts.evaluate.crenet_inference_optimization \
  --output outputs/experiment/crenet_inference_optimization_<run_id> \
  --device cuda --batch-size 2 --threads 4 --repeats 11
```

该入口默认比较两信道当前正式联合权重。先在完整验证集检查所有层的概率、log概率、软符号、方差和 CSI 增量，再在完整测试集逐 SNR 比较 BER、完整矩阵平均 NMSE 与硬判决差异，输出 BER–SNR 和 NMSE–SNR 图。数值容限在正式评测前固定为 `atol=rtol=1e-4`，CSI 增量的绝对容限按模型既有的逐样本信道尺度归一化；同时保存物理量的实际最大误差，不以容限替代精度指标。该容限依据真实验证小样本的精度审计：NCS1 原 FP32 路径相对相同权重的 FP64 路径已有约2.5e-5的概率误差，而 FP64 优化前后的概率差约8.3e-14。`scripts.evaluate.crenet_frozen_equivalence --double` 可对相同权重执行双精度代数核验，不改变正式权重。

GPU 的 cuDNN 默认允许 TF32，不同卷积形状可能采用不同截断精度的内核。拆分缓存路径明确使用完整 FP32 卷积；数值等价及四种实现的速度比较均使用相同 FP32 条件。另以 `legacy_tf32` 保留原默认卷积设置下的 BER、NMSE 和计时审计，避免把精度设置变化混入优化收益。

首轮全量 NOF1 审计 `crenet_inference_20261116_c6a0fb3` 保留为未通过预设软状态容限的结果：两项同时启用时，8,601,600比特硬判决全部一致、逐SNR BER相同、NMSE差小于1e-7 dB，但最大概率差为3.25e-4。静态缓存单独启用也未带来实测加速，因此不默认启用。后续入口分别记录 `residual_only` 和 `both` 的全部数值检查、硬判决及指标；不因一个候选未通过就遗漏另一个信道，也不自动放宽容限。`complete` 仅表示评测完成，部署路径是否通过另见 `default_numerical_checks_passed`。

计时另用各 SNR、D1/D2 的真实验证输入，预热3轮、轮换实现顺序并取11轮中位数，分别报告原路径、仅静态缓存、仅残差复用和两者启用。纯前向包含八层全部计算与每次样本缓存准备；重建加前向另计主机到设备传输及既有完整 CSI 重建。两者均排除磁盘读取、模型加载与监督评分，不能与包含磁盘读取的旧报告混用。结果记录检查点、数据、源码身份、数值精度和设备设置。
