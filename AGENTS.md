# 项目说明

本项目用于水声通信 AI 信道均衡算法研究，基于 WATERMARK 信道构建真实接收机链路数据集，使用接收机估计得到的不完美 CSI 对 MMSE 基线与学习型均衡算法进行统一指标对比。

## 文档与当前实验范围

- 当前阶段先验证 D-D-D-D 帧的 Single-tap LMMSE 与 Full-matrix LMMSE，使用 `configs/data/{nof1,ncs1}/dddd.json`：每符号 512 数据和 256 梳状导频，DPSS 半带宽 12 Hz、固定 Q=24、L=176。小实验入口及结果定义见 `docs/four_data_lmmse.md`。
- CSI 直接使用接收机原始估计，不根据参考 CSI 或固定 NMSE 目标调整。参考 CSI 仅用于独立先验构造、理想参考均衡、训练监督和误差评分。
- 历史 P-D-D-P 数据集、补偿视图、训练检查点及缓存不适用于新的 D-D-D-D 帧。后续学习型算法实验须先生成并验收新帧数据，再适配四个数据符号的训练与评测代码；当前小实验不代表正式数据集已完成。
- CRENet 的原有8层残差 MMSE Unfolding 与论文消融方案可作为后续设计参考，但不得直接复用旧参数条件下的权重或结果。
- `README.md` 仅保留项目简要说明及文件目录结构；未经用户明确允许，不得修改。
- NCS1 与 NOF1 使用 `configs/data/` 中的两份 D-D-D-D 配置，共用同一实际 DPSS 时间基与延迟支撑；`pddp.json` 仅为历史帧配置。
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
