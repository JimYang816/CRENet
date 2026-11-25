# CRENet：当前统一算法与运行方式

> 本文中的训练命令和数值对应旧 P-D-D-P 协议，作为历史设计与结果保留。当前 D-D-D-D 正式数据默认使用参考辅助补偿 CSI；CRENet 流程须适配四符号后再运行，旧权重不可复用。项目现仅保留两种 LMMSE 与 CRENet，当前状态见 [四符号协议](four_data_lmmse.md) 和 [统一评测](equalizer_comparison.md)。

当前设计为 **8层残差 MMSE Unfolding + 共享带状 CSI 修正器**。NOF1、NCS1 都学习逐层物理参数和局部神经校准网络，分别训练权重。公式、参数和冻结测试见 [算法设计](crenet_algorithm_design.md)。

## 1. 输入与模型

统一使用已验收的 compensated CSI 和既定划分。NOF1 输入平均 NMSE 为 −12.70→−18.30 dB，NCS1 为 −6.20→−10.67 dB，SNR为0→30 dB。算法输入包含接收观测、完整初始CSI、公开导频、物理位置/掩码和噪声参数；真实符号与参考CSI仅供监督及预测后的指标计算。

默认 `complex64`、对角噪声加权、修正半带宽16、局部窗口32、中心16、隐藏宽度48。修正器采用 `14(2B+1)+5` 个输入通道，修正仅作用于带内元素，所有带外耦合继续参与检测。

每层由 LMMSE 正规方程导出预条件残差方向，加入非负可学习 CSI 加载，并以精确二次方向线搜索和 `(0,2)` 内的可学习系数控制更新。QPSK距离 logits 与局部网络校准相加，概率阻尼使用稳定的 `log_softmax/logaddexp`。每层学习步长系数、加载、温度、不确定度和软阻尼，以及局部 MLP；共197,544个检测器参数，其中40个物理参数。两信道都执行完整神经分支。

## 2. 当前配置和训练

唯一训练配置为 `configs/training/crenet.json`。共同结构放在 `model`，共同训练设置放在 `training`，`channel_overrides` 记录信道独立的加载初值、学习率和监督权重；禁止仅为某个信道冻结神经网络。旧标量配置及分散的 `crenet_budgeted_*.json` 已合并清理。

训练入口依次执行共同预训练、固定CSI分支和联合分支。两分支共享同一个阶段A检查点：

| 分支 | 阶段/轮数 | 学习内容 |
|---|---|---|
| `warmup` | A：10 epoch | 固定CSI，全部物理参数和神经网络 |
| `fixed` | 复用A，继续10 epoch | 固定CSI，全部检测器参数 |
| `joint` | 复用A，B：10、C：10 epoch | B仅CSI修正器；C学习检测器与修正器 |

检测损失为比特交叉熵，末层权重0.9、其余7层各0.1/7。阶段C的完整矩阵CSI NMSE使用相同层权重，再乘以 `lambda_h`；阶段B仅对所有层和样本的CSI NMSE均匀平均。阶段A按验证BER选型，阶段B优先验证NMSE，阶段C优先考虑CSI是否改善，再比较验证BER和NMSE。训练种子20261028，batch 8，SNR交错，`dense_device`，预取2批，8192 MiB有界主存缓存。具体分组学习率见算法设计第3.7节。

```bash
# 服务器：依次训练两信道及两分支
python -m scripts.run.crenet --device cuda --threads 4
# 只训练一个信道，或恢复同一配置
python -m scripts.run.crenet --channel NOF1 --device cuda --resume
# 单独执行某个阶段，fixed/joint要求先有同根目录下的warmup
python -m scripts.train.crenet --channel NCS1 --stage warmup --device cuda
python -m scripts.train.crenet --channel NCS1 --stage joint --device cuda
```

新训练目录为 `outputs/experiment/crenet_current/{warmup,fixed,joint}/<channel>/`。`--output` 指定新的实验根目录，所有分支及共享预训练路径共同切换。续训校验实际训练、数据和源码身份；不能用整理后的新配置直接续接历史身份不同的目录。

本地仅运行CPU小样本验证；完整训练仍在服务器进行。性能测量入口为 `scripts.evaluate.crenet_training_perf`，它会解析对应信道的实际模型设置，计时和缓存参数由自己的CLI控制。

## 3. 当前正式检查点与历史预算复核

历史正式训练保存在 `outputs/experiment/icr_full_20261031_095549/`，两信道均包含共同预训练、固定CSI分支和联合分支，`progress.json` 标记 `completed`。下表记录当时模型，当前新帧配置未绑定这些旧权重：

| 信道 | 固定CSI | 完整模型 |
|---|---|---|
| NOF1 | `icr_full_20261031_095549/fixed/nof1/icr_mmse.pt` | `icr_full_20261031_095549/joint/nof1/icr_mmse.pt` |
| NCS1 | `icr_full_20261031_095549/fixed/ncs1/icr_mmse.pt` | `icr_full_20261031_095549/joint/ncs1/icr_mmse.pt` |

路径均相对 `outputs/experiment/`。共同预训练位于该实验的 `warmup/<channel>/warmup.pt`，正式主比较结果见第4节，等深度A–D结果见第5节。推理通过 `prepare_frozen_inference` 缓存固定参数变换和窗口几何；数值等价检查不修改推理公式或训练权重。

历史实验目录已迁至 `outputs/experiment/`，报告内记录的原生成路径保留不变。旧检查点只有在迁移前源码身份与当前重命名实现均严格匹配时才可用于推理，并继续校验 CSI 视图和数据身份；其他版本不会因改名而自动通过校验。历史报告可以离线按 CRENet 标签重画，但训练续接、消融及计算缓存仍需满足各自的完整源码与数据身份校验。完整权重位于服务器，本次迁移没有重新训练模型。

历史预算修复使用 `icr_budgeted_unified_20261031` 的NOF1权重和 `icr_budgeted_20261031` 的NCS1权重。相关历史结果和检查点已清理，保留的数值记录见 [算法设计第3.8节](crenet_algorithm_design.md#38-历史全参数预算修复的冻结结果)；复核需要先恢复同身份的历史产物。

该历史复核的四组平均BER均优于历史和同修正器权重LMMSE，实测总耗时不超过原Unfolding预算。全部展开层及CSI修正计入稳态耗时，一次性加载/准备单列。历史NOF1为22层、修复版8层，NCS1均8层，因此这批预算修复结果不替代相同深度的正式A–D消融，也不是当前正式权重的耗时实测。

该历史复核中，NOF1固定CSI相对未训练初始化没有显著学习收益，部分0 dB点仍差于LMMSE，联合CSI NMSE也未全面改善。单种子、物理记录配对区间不能证明训练随机性稳定或神经校准的独立贡献。完整负面结果和区间保存在历史冻结报告中。

## 4. 三算法主比较

当前仅比较 Single-tap LMMSE、Full-matrix LMMSE 和 CRENet，配置为 `configs/evaluation/ber_comparison.json`。当前两种 LMMSE 已接入四符号实验；统一入口及 CRENet 流程仍需按新协议适配，运行边界见 [统一评测](equalizer_comparison.md)。

```bash
python -m scripts.evaluate.compare --device cuda --threads 4
```

历史完整报告保留原始误码计数及输入/修正 CSI NMSE。离线绘图只提取当前三种方法，不改变原始来源；后续更新检查点时，必须重新校验计算身份并完成所需评测，才能引用新结果。

## 5. 比较、消融与输出

论文核心消融采用下列四组，与训练阶段A/B/C区分：

| 版本 | CSI处理 | 均衡器 |
|---|---|---|
| A | 固定CSI | Full-matrix LMMSE |
| B | 固定CSI | 当前MMSE Unfolding |
| C | 迭代CSI修正 | Full-matrix LMMSE |
| D | 迭代CSI修正 | 当前MMSE Unfolding |

B/A检查固定CSI下的检测器替换，C/A检查传统检测器下的修正收益，D/B检查展开检测器下的修正收益，D/C检查已有修正时的检测器作用。C保留多轮CSI修正与软反馈，A只做一次有效求解。B/D使用同一展开深度及共同预训练，C/D修正结构及轮数一致。参数和检查点只由训练/验证确定，不预设每个SNR均改善。

旧 `icr_ad_ablation_20261030` 检测器初始化审计显示最终B/D未实际学习，不能将其差异归因于参数学习。旧消融及单次修正、去不确定度、去门控和等预算一次修正等非论文核心诊断产物已清理。

正式A–D自动运行入口如下。`--trained-root` 指向当前配置完成的共同预训练及固定/联合分支，入口核验训练源码、数据、配置和逐层参数学习后复用B/D；A没有待学习检测器，C独立训练具有LMMSE软反馈的CSI修正器，沿用D的8轮修正及B/C阶段预算。所有模型先冻结，再运行验证集和测试集，测试结果不参与选型，也不要求预设的性能排序。

```bash
python -m scripts.run.crenet_ablation --trained-root outputs/experiment/crenet_current --output outputs/experiment/crenet_ad_current_<run_id> --device cuda --threads 4
# 中断后按同一代码、配置、数据和权重恢复
python -m scripts.run.crenet_ablation --trained-root outputs/experiment/crenet_current --output outputs/experiment/crenet_ad_current_<run_id> --device cuda --threads 4 --resume
```

输出包含 `progress.json`、冻结权重清单、检测器学习审计、每信道/划分/组别的逐层指标、A–D汇总JSON/CSV、BER/NMSE对比图、四项物理记录配对BER差异区间及可传回本地的 `reports.zip`。失败时进度记录异常，既有检查点和历史实验均保留。

当前正式A–D实验 `outputs/experiment/icr_ad_current_20261031_133445/` 已完成两信道的验证集和全量测试，根目录 `progress.json` 标记 `completed`。B/D复用第3节的固定/联合检查点，C单独训练；冻结权重和实际参数学习记录分别见 `frozen_checkpoints.json`、`detector_learning_audit.json`。每信道测试覆盖7个SNR、8,400个D块，每SNR为1,228,800比特。下表平均BER为7个SNR的算术平均，数值来自 `results_digest.json`：

| 信道 | 组别 | 平均BER | 30 dB BER | 30 dB CSI NMSE / dB |
|---|---|---:|---:|---:|
| NOF1 | A | 0.04411807 | 0.00444661 | −18.3000 |
| NOF1 | B | 0.04327765 | 0.00326904 | −18.3000 |
| NOF1 | C | 0.04176851 | 0.00251302 | −20.8804 |
| NOF1 | D | 0.04086403 | 0.00164632 | −20.8493 |
| NCS1 | A | 0.06927525 | 0.02085612 | −10.6700 |
| NCS1 | B | 0.06623024 | 0.01541829 | −10.6700 |
| NCS1 | C | 0.05469180 | 0.00918620 | −15.1969 |
| NCS1 | D | 0.05196987 | 0.00641846 | −14.3630 |

各信道的 `test/<channel>/comparison.json`、`comparison.csv` 保存完整逐SNR数值及四项配对区间，同目录包含BER/NMSE图；验证结果在对应 `validation/` 目录。D的测试平均BER最低，但30 dB的CSI NMSE仍弱于C，不能表述为所有指标均改善。该结果仍为单训练种子，配对区间反映物理记录间差异，不代表多种子稳定性。

A–D内部统一使用冻结配置的 `complex64` 和对角噪声权重；主比较中的传统Full-matrix LMMSE使用完整热噪声协方差及双精度。两类报告须分别引用，不能跨表替换A或D的误码计数。

每次完整评测至少生成BER–SNR、平均完整矩阵NMSE–SNR，保留实际误码数、逐层结果、数据/代码/权重摘要、计时及物理记录配对统计。预算复核保留的 `legacy_20261030.py` 与 `legacy_lmmse_20261030.py` 是原推理快照，不能删除或当成活动训练模型。

## 6. 验证与运行

本地CPU验证覆盖新配置解析、两分支共享预训练、分组学习率、逐层参数学习审计及推理等价性。服务器按 `main` 同步代码后读取原数据和检查点。Git只保存代码、配置、测试与文档；数据、训练权重、旧结果和论文产物留在各自输出目录。

保留的效率入口默认对三种算法分别启动独立进程，只需一个 CRENet 检查点。以下为旧协议的运行示例，必须使用与比较报告匹配的数据和权重：

```bash
python -m scripts.evaluate.crenet_efficiency \
  --dataset data/processed/nof1_compact_20261026_final/test.h5 \
  --comparison-report outputs/experiment/crenet_comparison/nof1/metrics.json \
  --checkpoint outputs/experiment/icr_full_20261031_095549/joint/nof1/icr_mmse.pt \
  --output outputs/experiment/crenet_efficiency_<run_id>/nof1.json \
  --device cuda --batch-size 2 --threads 4
```

NCS1使用 `data/processed/ncs1_compact_20261026_nmse/test.h5`，其余信道路径切换为 `ncs1`。初始化、稳态耗时、参数量及峰值内存按实际报告引用；第3节的历史预算表或缓存内原始耗时不能替代当前主比较的独立效率测量。
