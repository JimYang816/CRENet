# 项目说明

本项目用于水声通信 AI 信道均衡算法研究，基于 WATERMARK 信道构建真实接收机链路数据集，使用接收机估计得到的不完美 CSI 对 MMSE 基线与学习型均衡算法进行统一指标对比。

## 文档与当前实验范围

- 当前只保留 Single-tap LMMSE、Full-matrix LMMSE 和 CRENet 三种均衡算法；新增评测、配置和图表均使用这三种算法，历史产物仅保留为来源记录。
- 当前单符号实验采用 D-D-D-D 帧、FFT=1024 和 80 ms 循环前缀（6 kHz 下为 480 点）。NOF1 与 NCS1 的 `configs/data/{nof1,ncs1}/dddd.json` 共用 64 簇保护导频。每个带内组为 4 个数据、1 个空载波、4 个导频、1 个空载波、4 个数据，频带两端各有 64 个空载波。单符号入口 `scripts/evaluate/four_data_single_symbol.py` 默认按该布局重放接收波形，使用 12 Hz / Q=5 / 256 个物理时延抽头。入口和复现方式见 `docs/four_data_cp80.md`。
- `raw` 视图保留接收机原始 CSI。原梳状导频正式数据默认读取 `reference_assisted` 视图。NOF1 平均 NMSE 从 -9.69 到 -16.05 dB，NCS1 从 -5.24 到 -9.13 dB，采用尺度 10 dB 的先陡后缓曲线。训练、验证、测试分别按全量线性 NMSE 均值校准。该视图使用完整参考矩阵，属于理想辅助数据。
- 原梳状导频的 CP=80 ms 参考辅助数据已在服务器 `data/processed/{nof1,ncs1}_dddd/` 生成并全量验收，生成代码为 `5ee9af5`，使用 Q=24 的整帧估计和 192 个物理时延抽头。复现入口为 `data_generation/scripts/four_data.py`，配置为 `configs/data/four_data_dataset.json`，读取器为 `src.data.FourDataDataset`。先验来源为已验收的 `outputs/smoke/four_data_cp80_20261125/`。每信道训练、验证、测试含 3,675、525、4,200 个帧样本，计入 7 个 SNR，每样本 4 个数据符号。双信道约 19.72 GiB，运行及验收记录在 `outputs/experiment/dddd_cp80_delay192_assisted_20261126/`。这些 HDF5 为原布局复现及新布局重放提供物理参考和配对序列。旧 CP=256 数据已删除，历史权重及结果禁止用于新的默认实验。
- CRENet 的原有8层残差 MMSE Unfolding 与论文消融方案可作为后续设计参考，但不得直接复用旧参数条件下的权重或结果。
- `README.md` 仅保留项目简要说明及文件目录结构；未经用户明确允许，不得修改。
- NCS1 与 NOF1 共用同一个单符号 DPSS 时间基，各自从独立先验冻结 256 个真实时延位置。原整帧实验的 192 点支撑及 `pddp.json` 是历史来源。
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
