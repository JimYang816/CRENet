# 项目说明

本项目用于水声通信 AI 信道均衡算法研究，基于 WATERMARK 信道构建真实接收机链路数据集，使用接收机估计得到的不完美 CSI 对 MMSE 基线与学习型均衡算法进行统一指标对比。

## 文档与当前实验范围

- 数据生成已完成，后续所有均衡算法统一使用当前已验收的数据集及既定划分进行训练和测试。
- 当前 CRENet 算法使用统一8层残差 MMSE Unfolding：NOF1、NCS1 均学习逐层物理参数和局部神经校准网络，禁止按信道冻结神经分支。唯一训练配置为 `configs/training/crenet.json`，允许各信道独立权重及验证确定的学习率/加载初值。当前设计见 `docs/crenet_algorithm_design.md`。
- 主对比包含 Single-tap LMMSE、Full-matrix LMMSE、MMSE-DFE、DD-CE + Full-matrix LMMSE、UDNet、OAMP-Net2 和 CRENet。DD-CE 从相同 compensated CSI 起步，单独报告修正后 CSI 的平均 NMSE。EP、Robust EP 不再参与性能或效率对比。
- 论文正式消融采用 A–D 四组：A＝固定 CSI + Full-matrix LMMSE，B＝固定 CSI + MMSE Unfolding，C＝迭代 CSI 修正 + Full-matrix LMMSE，D＝迭代 CSI 修正 + MMSE Unfolding（完整模型）。单次修正、去不确定度、去门控及等预算单次修正保留为补充诊断，不列入正文核心消融；定义与结果使用边界见 `docs/crenet.md` 第5节。
- 默认训练、验证、测试必须使用已验收的 **compensated CSI**：NOF1 平均 NMSE 随 SNR 0→30 dB 为 **−12.70→−18.30 dB**，NCS1 为 **−6.20→−10.67 dB**。未经用户明确要求，不得改为 raw CSI 或修改这些目标。raw 数据保留，仅供显式诊断；不得复用 raw 视图的训练检查点、特征或评测缓存。
- `README.md` 仅保留项目简要说明及文件目录结构；未经用户明确允许，不得修改。
- NCS1 与 NOF1 使用 `configs/data/` 中的两份固定配置，共用同一实际 DPSS 时间基与延迟支撑。
- 当前固定12 Hz、无P的4D实验必须保持原载波掩码：每D 128导频、256保护、512数据，两侧各64空载波，四D位置相同；不得为改善可辨识性而增加导频、移除边缘空载波或错位掩码。全频带历史实验仅作归档，当前入口为 `scripts/evaluate/four_data_covariance.py`。恢复掩码后若Q_eff×L大于1536，必须如实报告欠定性。
- 原掩码下重新选择非欠定Q_eff/L使用上述入口的 `--select-ql`，只用051–055先验和056–060独立验证；必须同时满足Q_eff×L≤1536及实际公开导频矩阵满列秩（SVD相对阈值1e-10），两信道参数冻结后才进入测试，不依据测试CSI或测试BER反向调参。
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
