# 原七算法计算开销统计

本统计比较 Single-tap LMMSE、Full-matrix LMMSE、MMSE-DFE、DD-DPSS、UDNet、OAMP-Net2 和 CRENet。
使用完整保留七种算法的历史提交 `fe873ac5bb3a57d0e6d8e6ed042e97de655e19f4`，在结果目录中导出独立源码快照；
历史代码仍称 CRENet 为 `ICR-MMSE-Net`。不将已归档算法恢复为项目默认算法。

旧版完整数据已清理。本次从当前 224 导频 NOF1、NCS1 测试数据准备统一的接收机输入，
仅测计算开销，不测 BER、NMSE，也不据此评价历史权重在新数据上的精度。
三种学习算法使用各信道原有正式权重，保留历史加载器的源码身份校验，不重新训练或改写检查点。
历史训练数据身份与当前输入不同，因此不调用用于精度评测的历史数据身份验证；这一差异在结果中明确记录。

## 计时口径

- 服务器在 GPU 可用时使用 CUDA，batch 为 1；主表单位为一个 OFDM 数据符号的毫秒数。
- 当前输入尺寸为 FFT 1024、512 数据载波和 224 导频，使用参考辅助 CSI 视图。
  参考矩阵只在输入准备阶段用于构造这一既有视图，不进入计时进程。
- 每信道每 SNR 均匀取 4 个帧位置，分别使用第 1 至第 4 个数据符号；7 个 SNR 共 28 个块。
- 每个算法、信道在独立进程运行。输入预先驻留设备，预热 5 次，然后完整遍历这 28 块 7 次。
  每轮采用确定的随机次序，每次调用前后同步 GPU。主表为七轮平均单符号耗时的中位数。
- 计入导频消除、逐样本白化/统计量构造、UDNet 的局部伪逆、OAMP-Net2 的 Gram 特征分解、
  DD-DPSS 的软判决重估与第二次检测、所有展开层，以及算法接口返回主存输出的开销。
  每次调用使用新的接收机批对象，禁止跨次复用 Gram、特征分解、局部 SVD 或检测结果。
- 不计磁盘读取、完整 CSI 重建、初始输入传输、加载权重、公共噪声白化矩阵和 DD 先验模式的一次性构造。
  一次性准备耗时另存于各算法 JSON 的 `setup_seconds`。
- 保留历史实现精度：传统算法和 OAMP-Net2 使用 complex128/float64；
  UDNet 预处理为 complex128/float64，网络为 float32；CRENet 为 complex64/float32。
  关闭 TF32；CPU 和 BLAS 线程限制为 4。该表比较这些实际实现的开销。

DD-DPSS 保留 rank 32、prior scale 0.03 的原算法。当前先验采用 192 个物理时延位置，
将原有模式构造器输出的紧凑时延槽映射回这些真实位置；模式选择仍基于公开接收机先验。

## 复杂度口径

复杂度按实际实现的主导运算阶数给出，包含逐样本预处理，不将仅支持部分算子的 FLOPs 工具输出称为完整接收机 FLOPs。
另以 forward hook 精确统计神经网络 Linear/Conv1d 乘加次数，保存于各算法 JSON 的
`neural_linear_convolution_macs`；该值不包含物理矩阵运算、SVD 或特征分解。

记 `N=1024`、`D=512`，`F=N²D+ND²+D³`。
`N²D` 来自完整噪声白化矩阵乘数据通道，`ND²` 来自 Gram 构造，`D³` 来自线性求解或矩阵分解。
公共白化矩阵自身的分解属于一次性准备；每个新信道样本与它的乘法仍计入。

- Single-tap LMMSE 仅在数据位置逐点运算，复杂度 `O(D)`。
- Full-matrix LMMSE 为 `O(F)`；MMSE-DFE 另有 `O(D²)` 的顺序硬反馈，保持原载波检测次序。
- DD-DPSS 包含两次完整似然构造、一次初始协方差求逆和一次最终 LMMSE 求解，另有
  rank `r=32` 的拟合与完整 ICI 更新：`O(F+N²r+Nr²+r³+N²log N)`；`O` 记号省略两次检测的常数。
- UDNet 使用长度 `w=32`、步长 16 的局部窗口，隐藏宽度 `h=48`、层数 `T_u=22`。
  `W_u` 为实际覆盖数据载波的窗口数。完整导频消除和局部白化、Gram、伪逆及展开合计
  `O(N²+W_u[w³+T_u(w²+wh)])`。
- OAMP-Net2 预先对完整 `D×D` Gram 做一次特征分解，并在 `T_o=10` 层复用谱表示，
  因此为 `O(F+T_o D²)`，不是每层重新求逆。
- CRENet 使用 `T_c=8` 层、带宽 `b=2×16+1=33`、卷积宽度 `c=32`、
  长度 `w=32`/中心 16/隐藏宽度 `h=48` 的 `W_c=64` 个局部窗口。
  卷积核长度固定为 3，完整信道作用、局部修正和判决校准合计
  `O(T_c[N²+Nbc+Nc²+W_cwh])`。共享修正器在参数量中只计一次，在推理中按 8 次计算。

参数量统计原模型的全部可学习参数，排除固定物理先验、缓存和非参数缓冲区；
冻结后的 `requires_grad=False` 不将已学习参数变成零。

## 运行及产物

```bash
# 本地：CPU 合成小样本，验证全部七种算法；不作为正式计时结果
python -m scripts.evaluate.seven_algorithm_efficiency --smoke --device cpu \
  --output outputs/smoke/seven_algorithm_efficiency_smoke

# 服务器：使用当前正式接收机输入和各信道历史权重
python -m scripts.evaluate.seven_algorithm_efficiency --device cuda \
  --output outputs/experiment/seven_algorithm_efficiency_20261127
```

配置为 `configs/evaluation/seven_algorithm_efficiency.json`。
结果包含一张 `comparison.md`/`comparison.csv` 对比表、完整 `summary.json`、输入抽样与身份、
每算法每信道的重复计时及权重摘要、进度记录和不可变历史源码归档。
正式结果与输入缓存不提交 Git。
