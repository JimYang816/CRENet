# 四符号 CRENet 均衡器训练诊断

本页记录 2026-11-26 对 CP80 双信道 CRENet 的验证集诊断。输入继续使用
`reference_assisted`，其构造依赖完整参考信道；结果不能解释为无参考 CSI 性能。
原正式实验及数据口径见 [四符号 CRENet 训练与测试](four_data_crenet.md)。

## A 阶段为何选择第 0 轮

当前 A 阶段以逐比特交叉熵训练，但验证选模按 BER 优先、交叉熵次之；
第 0 轮的原始模型也参与竞争。每个验证 BER 包含 2,150,400 bit，
7 个 SNR 每点 307,200 bit。

| 信道 | 第 0 轮 BER / 交叉熵 | 训练后 BER 最佳轮 | 该轮 BER / 交叉熵 | 相对第 0 轮误码变化 |
| --- | --- | ---: | --- | ---: |
| NOF1 | 0.05181966 / 0.123387 | 2 | 0.05234468 / 0.111956 | +1,129 |
| NCS1 | 0.07985863 / 0.207864 | 3 | 0.08022414 / 0.183235 | +786 |

因此两条信道都不是没有训练梯度，而是交叉熵改善没有转化为硬判决 BER 改善，
连续 4 轮未超过第 0 轮后早停。上述两轮相对第 0 轮按 0、5、…、30 dB 的误码变化分别为：

- NOF1：`[769, 236, 69, 0, 28, 0, 27]`。
- NCS1：`[749, 177, 207, 110, -134, -155, -168]`。

每个 SNR 的 bit 数相同；低 SNR 的实际退化较大，并非选模对低 SNR 额外加权。
直接改成按交叉熵选择会选中验证 BER 更差的模型。

## 验证集上的分组与目标对照

诊断从各信道原 `best_A.pt` 第 0 轮精确恢复，使用原训练记录、完整验证集、batch=32、
相同随机批次及原 `reference_assisted` 视图。每个诊断间隔抽取 50 个训练批次，
最多 3 个间隔；这是配对的小规模优化诊断，不是完整训练轮。
初始验证逐 SNR 误码与原检查点完全一致。各方案保留自己验证 BER 最好的训练检查点，
并与第 0 轮比较：

| 信道 | 方案 | 更新次数 | 验证 BER | 相对初始误码变化 |
| --- | --- | ---: | ---: | ---: |
| NOF1 | 全部均衡器参数，Adam 1e-3 | 50 | 0.05265113 | +1,788 |
| NOF1 | 仅物理标量，Adam 1e-3 | 50 | 0.05181083 | -19 |
| NOF1 | 仅局部神经网络，Adam 1e-3 | 50 | 0.05268927 | +1,870 |
| NCS1 | 全部均衡器参数，Adam 1e-3 | 50 | 0.08091239 | +2,266 |
| NCS1 | 仅物理标量，Adam 1e-3 | 100 | 0.07972331 | -291 |
| NCS1 | 仅局部神经网络，Adam 1e-3 | 50 | 0.08109235 | +2,653 |

只训练局部神经网络时，NOF1/NCS1 分别有 1,366/2,228 个新增误码来自 0 dB。
这表明该学习率下神经分支的更新与主要 BER 退化相关；它不单独证明更深层的统计原因。
将全部参数学习率降到 1e-4 只能减轻短程退化。另试用平滑比特误判目标加 0.1 倍交叉熵：
NOF1 未超过初始 BER；NCS1 在物理参数 1e-3、神经参数 1e-4 时，150 次更新比初始少错
367 bit，但 0 dB 多错 456 bit。不能只凭该平均值宣称各 SNR 均有改善。

诊断脚本早期版本切换参数组时没有清除被冻结参数残留的 `.grad`；这会污染
部分 `first_gradient_l2` 日志，但冻结参数不在优化器及梯度裁剪列表中，
不影响参数更新、保存的权重或 BER/NMSE。后续代码已清除全部旧梯度并增加回归检查。

## CSI 修正后的均衡器微调

从原 `best_B.pt` 出发，冻结 CSI 修正器，启用其反馈并仅微调均衡器；
仍按 C 阶段的验证规则检查每个 SNR 的修正后 CSI 不劣于输入 CSI。
NCS1 原最终 C 阶段验证 BER 为 0.06170898，短程微调均未超过该结果。

NOF1 原最终模型即 B 阶段权重，验证 BER 为 0.04837519。延长的配对诊断每间隔
抽取 100 个训练批次，最多 5 个间隔：

| 方案 | 最佳更新次数 | 验证 BER | 相对原最终模型误码变化 | 参数变化 |
| --- | ---: | ---: | ---: | --- |
| 仅物理标量 1e-3 | 400 | 0.04832310 | -112 | 40/40 个标量张量变化 |
| 物理标量 1e-3，神经参数 1e-5 | 500 | 0.04830822 | -144 | 40/40 个标量及 32/32 个神经张量变化 |

两个候选均满足当前逐 SNR CSI 约束，但验证增益很小；不能仅据此更改正式默认训练配置。

## NOF1 冻结候选的完整配对测试

仅按验证结果固定上表的 500 次联合微调候选，然后测试全部 7 个 SNR、每点
2,457,600 bit。测试前保存候选、原模型、验证结果、数据和源码身份；没有使用测试结果选模。
原模型与候选使用相同物理帧、接收观测和噪声，三条非 CRENet 对照的逐符号误码完全一致。

| SNR（dB） | 原 CRENet BER | 候选 BER | 候选减原模型误码 |
| ---: | ---: | ---: | ---: |
| 0 | 0.19218099 | 0.19231527 | +330 |
| 5 | 0.08948853 | 0.08947428 | -35 |
| 10 | 0.03392008 | 0.03385091 | -170 |
| 15 | 0.01256470 | 0.01252197 | -105 |
| 20 | 0.00544637 | 0.00545085 | +11 |
| 25 | 0.00318563 | 0.00321248 | +66 |
| 30 | 0.00246989 | 0.00253296 | +155 |

全部 17,203,200 bit 中，原模型错 833,756 bit，候选错 834,008 bit，
总体多错 252 bit。10 条测试物理记录中 4 条误码减少、6 条增加；
每记录包含多个相关窗口和重复，不能把所有 bit 当成独立样本。
30 dB 的修正后平均 NMSE 从 -18.70966 dB 变为 -18.74564 dB，
但 0 和 5 dB 分别变差约 0.056 和 0.080 dB。

因此这个候选证明均衡器两个参数组可以得到非零更新及验证集收益，
但其验证收益没有稳定推广到本次测试集；保留结果，不替换原正式模型或默认配置。
测试集已经用于本次核验，进一步调参后的新候选应使用新的独立留出数据确认。

后续训练可先以物理标量学习率 `1e-3`、局部神经学习率 `1e-5` 作为对照，
在验证集按更新次数保留第 0 轮及中间检查点，并同时检查逐 SNR BER、逐记录误码及 CSI NMSE。
如需进一步约束低 SNR 的局部网络更新，应在训练/验证数据上预先定义约束，
再用新的留出记录检验；上述学习率是诊断起点，不是已经通过测试的默认配置。

## 可复核材料

- `outputs/experiment/four_data_stage_a_diagnosis_20261126/`：原目标的参数分组对照。
- `outputs/experiment/four_data_stage_a_softber_20261126/`：平滑误判目标对照。
- `outputs/experiment/four_data_detector_after_b_20261126/`：双信道修正后短程微调。
- `outputs/experiment/four_data_detector_nof1_longer_20261126/`：NOF1 修正后延长微调。
- `outputs/experiment/four_data_detector_nof1_holdout_20261126/`：冻结候选的完整配对测试。

各目录保存完整验证指标、候选检查点以及 BER–SNR、平均 NMSE–SNR 图。
上述四组共 63 个文件已从服务器回传并逐文件核对 SHA-256。
配对测试另外 10 个文件也已逐文件核对 SHA-256。
脚本入口为 `python -m scripts.evaluate.four_data_stage_a`；
各次结果的 `summary.json` 保存实际源代码哈希、数据身份、设备、采样种子及超参数。
配对测试入口为 `python -m scripts.evaluate.four_data_detector_holdout`，
结果包含 `frozen_selection.json`、`comparison.json`、逐帧数据及 BER/NMSE 图。

例如，在服务器复跑 NOF1 的修正后分组诊断，可使用新的输出目录：

```bash
cd /home/fay/JimYang/CRENet
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/fay/anaconda3/envs/jimyang/bin/python -m scripts.evaluate.four_data_stage_a \
  --previous outputs/experiment/four_data_crenet_cp80_20261126 \
  --channel NOF1 --start B --steps 100 --intervals 5 \
  --variants scalars_1e-3 scaled_neural_1e-2 --device cuda --threads 4 \
  --output outputs/experiment/four_data_detector_nof1_repeat
```

重放已经冻结的候选配对测试如下；这只用于复核，不提供新的独立留出集：

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/fay/anaconda3/envs/jimyang/bin/python -m scripts.evaluate.four_data_detector_holdout \
  --previous outputs/experiment/four_data_crenet_cp80_20261126 \
  --channel NOF1 --variant scaled_neural_1e-2 --device cuda --threads 4 \
  --diagnosis outputs/experiment/four_data_detector_nof1_longer_20261126 \
  --output outputs/experiment/four_data_detector_nof1_holdout_repeat
```
