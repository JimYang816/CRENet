# CRENet

基于 WATERMARK NCS1／NOF1 的水声通信 AI 信道均衡研究项目，统一使用 Python。
基于已验收的 compensated CSI，对比 Single-tap LMMSE、Full-matrix LMMSE、MMSE-DFE、
DD-DPSS、UDNet、OAMP-Net2 和 CRENet；CRENet 默认采用 8 层联合 CSI 修正与 MMSE 展开。

## 文件目录结构

```text
CRENet/
├── configs/
│   ├── data/                 # ncs1/、nof1/ 固定数据生成配置
│   ├── training/             # CRENet、UDNet、OAMP-Net2 训练配置
│   └── evaluation/           # 统一评测配置
├── data_generation/
│   ├── scripts/              # 数据生成与物理链路检查入口
│   ├── src/                  # 发射、WATERMARK 信道回放、接收机及数据集
│   └── tests/                # 数据生成测试
├── src/
│   ├── algorithms/           # 每种均衡算法独立目录
│   │   ├── single_tap_mmse/  # 单抽头 LMMSE
│   │   ├── full_matrix_lmmse/ # 完整矩阵 LMMSE
│   │   ├── mmse_dfe/         # MMSE-DFE
│   │   ├── dd_ce_lmmse/      # DD-DPSS
│   │   ├── udnet/            # UDNet
│   │   ├── oamp_net2/        # OAMP-Net2
│   │   └── crenet/           # CRENet
│   ├── common/               # 数值后端与实验元数据
│   ├── data/                 # 数据读取与预处理
│   ├── training/             # 学习型算法训练与验证选型
│   └── evaluation/           # 指标计算、算法对比与效率评测
├── scripts/
│   ├── train/                # 独立训练入口
│   ├── run/                  # 训练、选型及消融流水线
│   ├── evaluate/             # 统一评测与必要诊断入口
│   ├── infer/                # 推理入口
│   └── visualize/            # 论文算法图生成入口
├── tests/                    # 算法测试按算法分目录；另含数据、训练和评测测试
├── docs/                     # 设计与运行说明
├── data/
│   ├── raw/watermark/        # 原始 WATERMARK 源包
│   ├── interim/              # 可再生成的校准缓存
│   └── processed/            # 接收机数据与估计 CSI
├── outputs/
│   ├── runs/                 # 正式实验、消融及保留的历史结果
│   └── paper/                # 论文主稿 main.tex、图片及历史稿 archive/
├── AGENTS.md                 # 项目开发约定
└── pyproject.toml            # Python 依赖与测试配置
```
