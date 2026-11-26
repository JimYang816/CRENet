# 项目说明

本项目用于水声通信 AI 信道均衡算法研究，基于 WATERMARK 信道构建真实接收机链路数据集，使用接收机估计得到的不完美 CSI 对 MMSE 基线与学习型均衡算法进行统一指标对比。

## 文档与当前实验范围

- 当前均衡算法为半带宽 16 的 Banded LMMSE；按真实 FFT 索引保留数据通道的 ±16 耦合，先消除已知导频，再使用完整热噪声协方差求解。Single-tap LMMSE 和 Full-matrix LMMSE 源码及旧实验入口已归档至提交 `a08592b`，历史产物仅保留为来源记录。
- 当前默认使用 D-D-D-D 帧、FFT=1024、循环前缀 80 ms（基带 6 kHz 下为 480 点）、12 Hz / Q=24、192 个物理时延抽头。每符号使用 512 个数据子载波、224 个梳状导频、224 个带内保护空载波和两端各 32 个边缘空载波。基础帧为 `configs/data/{nof1,ncs1}/dddd.json`，当前入口将 `delay_length` 设为 192。入口及结果定义见 `docs/four_data_cp80.md`。
- `raw` 视图保留接收机原始 CSI。正式数据默认使用 `reference_assisted` 视图，当前指定 NOF1 平均 NMSE 从 -11.41 到 -15.77 dB，NCS1 从 -5.37 到 -8.92 dB，采用尺度 10 dB 的先陡后缓曲线并保留两位小数；训练、验证、测试分别按全量线性 NMSE 均值校准。该视图明确使用完整参考矩阵，属于理想辅助数据，不能解释为真实接收机的无参考估计性能。
- 正式数据生成入口为 `data_generation/scripts/four_data.py`，配置为 `configs/data/four_data_dataset.json`，默认输出到 `data/processed/dddd_pilot224/{nof1,ncs1}_dddd/`，读取器为 `src.data.FourDataDataset`。每信道的训练、验证和测试分别含 3,675、525 和 4,200 个帧样本（计入 7 个 SNR，每样本 4 个数据符号）。全帧 DPSS 基和物理通道先验系数来自独立记录，来源为已验收的 `outputs/smoke/four_data_cp80_20261125/`。
- 当前 224 导频正式数据已于 2026-11-27 生成并全量验收，双信道约 19.78 GiB，记录为 `outputs/experiment/dddd_cp80_delay192_pilot224_assisted_20261127/`。256 导频的旧 CP80 正式数据约 19.72 GiB 已删除，旧验收记录保留在 `outputs/experiment/dddd_cp80_delay192_assisted_20261126/`；历史权重及结果禁止用于新的默认实验。
- 现有正式数据的 CSI 重校准与完整 B16 测试入口为 `data_generation/scripts/recalibrate_four_data.py`；本次 NCS1 使用 `--channels NCS1`，记录保存到 `outputs/experiment/dddd_assisted_b16_ncs1_892_20261127/`，NOF1 沿用 `outputs/experiment/dddd_assisted_b16_20261127/`。重校准只修改补偿元数据，原始生成身份保留，并单独记录校准身份和旧补偿参数；完整测试使用参考辅助视图，每信道每 SNR 共 600 帧。
- 旧版 CRENet（8 层残差 MMSE 展开）的源码、训练与评测配置及消融记录见提交 `38c1b00`。本地备份为 `outputs/experiment/archive/crenet_before_redesign_20261127_38c1b00.zip`，旧权重和结果仅用于复核历史实验。
- `README.md` 仅保留项目简要说明及文件目录结构；未经用户明确允许，不得修改。
- NCS1 与 NOF1 使用两份 D-D-D-D 配置并共用同一实际 DPSS 时间基；各自冻结 192 个物理时延位置，保留真实索引。`pddp.json` 仅为历史帧配置。
- 原始 WATERMARK 源包放在 `data/raw/watermark/`；生成数据、缓存和实验结果不得提交到 Git。

## 实现约定

- 按功能将代码及相关文件放入对应目录，命名清晰恰当，不随意散放文件。
- 整个项目统一使用 **Python** 实现，包括数据生成、信道均衡算法、学习型算法训练、评测指标计算与性能对比。
- 数据生成部分独立放在根目录 `data_generation/`，数据生成代码放在 `data_generation/src/`，包括发射信号生成、WATERMARK 信道仿真、接收机处理、CSI 估计及数据整理与导出；运行入口放在 `data_generation/scripts/`，测试源码放在 `outputs/smoke/data_generation_tests/`。
- 数据读取与预处理代码放在 `src/data/`，算法、训练及评测代码分别放在 `src/algorithms/`、`src/training/` 和 `src/evaluation/`；对应运行入口放在根目录 `scripts/`。
- 每种均衡算法使用 `src/algorithms/` 下独立的同名目录，相应算法测试放在 `outputs/smoke/tests/algorithms/` 下的同名目录；NCS1、NOF1 数据配置分别放在 `configs/data/ncs1/`、`configs/data/nof1/`。
- `data/processed/` 保存真实接收机链路数据及估计得到的不完美 CSI，供训练与评测使用。
- 正式实验、消融及相关缓存放在 `outputs/experiment/`，测试源码、小样本数据、临时文件及测试结果放在 `outputs/smoke/`；测试源码纳入 Git，数据、缓存和结果不提交。每次实验完成后清理项目中确认无用的文件和结果，保留论文所需结果、模型及可复核来源。
- 论文主稿与相关 TeX 材料统一存放在 `outputs/paper/`；论文中使用的所有图片必须直接放在 `main.tex` 同目录下，不放入图片子目录。
- 每次实验结果指标图至少包括BER-SNR、平均NMSE-SNR。

## 开发与运行方式

- **本地**：在 `D:\SCI\code\CRENet` 生成和修改代码，仅用 CPU 和少量测试数据验证能跑通即可。服务器运行默认在 CUDA 可用时使用 GPU，CUDA 不可用时使用 CPU。
- **Git 提交**：提交信息统一使用中文。
- **代码同步**：本地验证通过后推送至 GitHub 仓库 `https://github.com/JimYang816/CRENet.git`，服务器拉取对应版本后再正式运行。
- **远程运行**：通过 `ssh lab448` 连接 Linux 服务器，项目目录为 `/home/fay/JimYang/CRENet`。服务器使用公共 Conda 环境 `jimyang`，解释器为 `/home/fay/anaconda3/envs/jimyang/bin/python`。运行前检查可用资源，根据实际配置使用 GPU 进行正式计算。
- **数据与结果**：完整数据留在服务器，实验结果通过 SSH 文件传输带回本地；GitHub 只同步代码、测试、配置和文档。
