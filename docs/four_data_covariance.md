# 固定12 Hz、无P、4D的独立先验联合协方差修正

## 固定实验协议

NOF1仍为Q_eff=5、L=512，NCS1仍为Q_eff=12、L=224。
时间基复用修正前冻结的12 Hz/Q27字典旋转；每D为512导频、256保护、256数据，
全频带`DDZPPPPZ`布局和四D偏移0/1/3/5均不变。
修正后直接复用修正前保存的干净接收观测、标准实RF噪声、RF方差、发射增益和载荷，
因此每个SNR、每个配对帧的波形、发射功率、噪声和比特完全相同。

改变的部分仅为CSI估计的冻结联合二阶矩。输出仍为同一Q_eff×L系数及相同逐D循环CSI，
两种均衡器、导频消除方式和实际热噪声方差保持原实现。
NOF1的跨符号响应纳入CSI的观测统计，不额外增加CSI输出维度，也不改为跨D联合检测器。
原PDDP验收数据和compensated CSI不变；此实验按用户要求使用直接接收机估计CSI。

## 独立标定与验证边界

- 协方差标定仅用原先验051–055记录，各2/14秒，共10帧。
  接口通过`RecordSubset`限制可读记录；被冻结时间基和系数的来源与修正前一致。
- 收缩参数仅在独立056–060记录，各2/14秒，共10帧验证，不用其样本重新拟合协方差。
  目标为全部七个SNR、所有验证帧和四D的平均完整参考CSI线性NMSE。
  热噪声误差通过精确协方差解析平均，不依赖一次验证噪声抽样。
- 参数网格在最终测试前固定：alpha∈{0.1,0.4,0.7,1.0}、rho∈{0.05,0.2,0.5,0.8}。
  两信道各16种组合；alpha为经验系数矩与结构先验的混合比例，rho为导频残差形状收缩比例。
  所有候选保留非零系数—残差交叉项及非零经验残差结构。
- 测试仍为011/015/019/023/027/031/035/039/043/047，各2/14/26秒，共30帧；
  SNR为0:5:30 dB。拟合函数不接收测试CSI；先写出完整冻结估计器及验证选择记录，
  再读取测试观测并计算CSI，最后才读取测试参考信道计算指标。
  测试前后的冻结估计器SHA256必须相同。

本地smoke仅用051的两个先验窗口、056@2秒验证及011@2秒测试，固定alpha=0.4、rho=0.5，
用于CPU流程验证，不作为正式参数选择或性能结果。

## 完整物理数据与导频响应

对每个独立先验帧，以有限长实RF链路构造四个接收FFT窗口的
`y=A_frame x_bb+B_frame x_bb*`。保留所有四个发射D到所有接收D的响应及CP、保护区、
滤波和共轭项；不把NOF1的跨符号响应折叠成逐D循环近似。
独立检查要求该算子对固定公开导频的输出与实际RF回放一致。

令c_i为与修正前相同定义的完整自符号参考信道投影，A_p为当前BEM公开导频观测矩阵。
取实际导频响应y_pi，定义r_i=y_pi−A_p c_i。
r包含时间/延迟表示残差及物理跨D公开导频响应、共轭路径的剩余部分，
其与目标c的经验交叉二阶矩直接保留。

对每个发射D的数据载波，构造到全部接收D选取行的直接列F_i和共轭列G_i。
给定先验样本分布，未知proper单位QPSK数据的协方差解析积分为

\[
R_D=\frac1{N_p}\sum_i(F_iF_i^H+G_iG_i^H).
\]

R_D保留完整的跨D非对角块。若写F_i=F_BEM,i+E_i，
该计算同时包含空间内项、空间外项及F_BEM E^H+E F_BEM^H交叉项；
没有将这些项分别当作独立噪声。标定记录保留这四项的功率迹恒等式、
跨符号数据响应能量以及最终R_D的跨D非对角块范数。
NCS1也使用同一有限算子实现，其跨D项可为零或很小，不按信道切换算法。

## 保证半正定的联合矩标定

仅向原观测协方差添加残差协方差，再直接拼接经验交叉项，未必得到合法联合协方差。
这里用同一潜在先验样本构造目标系数与导频残差，保证一致性。

记C=[c_i]，R=[r_i]，U=A_p C，P_b为原块宽32的结构先验，R_n为选取行的RF热噪声协方差，
W R_n W^H=I。令

\[
\tau=\|W R\|_F^2/(N_p M),\qquad
P_c=(1-\alpha)P_b+\alpha CC^H/N_p,
\]
\[
C_{cr}=\sqrt{\alpha(1-\rho)}\,CR^H/N_p,
\qquad R_r=(1-\rho)RR^H/N_p+\rho\tau R_n.
\]

等价生成模型为c=sqrt(alpha)c_emp+sqrt(1−alpha)c_struct，
r=sqrt(1−rho)r_emp+sqrt(rho)r_iso，其中经验c、r共享样本潜变量，另外两项独立。
由此联合二阶矩半正定；目标系数总能量和导频残差的白化总能量不随收缩比例改变。

最终冻结的目标—观测交叉矩及无热噪声观测矩为

\[
C_{cy}=P_c A_p^H+C_{cr},
\]
\[
C_{yy}=A_pP_cA_p^H+A_pC_{cr}+C_{cr}^HA_p^H+R_r+R_D.
\]

估计器直接使用C_cy(C_yy+(sigma_RF²+epsilon)R_n)^−1 y。
数值ridge沿用原估计器的1e-6相对规则所得固定值，与实际热噪声及残差统计分开记录。
发射增益g仍吸收到有效CSI；所有信道相关矩均随g²缩放，运行时等价地将RF方差除g²，
不改变观测y，也不使用参考CSI缩放误差。

## 运行与复核文件

```bash
python -m pytest -q outputs/smoke/data_generation_tests/test_four_data.py \
  outputs/smoke/data_generation_tests/test_four_data_rotated.py \
  outputs/smoke/data_generation_tests/test_four_data_covariance.py \
  outputs/smoke/tests/evaluation/test_four_data_design.py \
  outputs/smoke/tests/evaluation/test_four_data_rotated_audit.py
python -m scripts.evaluate.four_data_covariance --device cuda --threads 4
python -m scripts.evaluate.four_data_covariance --report-only
```

正式输出默认为`outputs/experiment/four_data_joint_covariance_20261119/`。
根目录保存三方案精确CSV、JSON及`comparison`、`ber_snr`、`nmse_snr`的PNG/PDF；
每信道保存`frozen_moments.h5`、`calibration.json`、独立`validation.h5`、全部候选`selection.json`、
`frozen_estimator.h5`、测试CSI系数与逐帧误码记录，便于检查参数是在测试前冻结的。
修正前及原Q24/L176结果只读，完整数据留在服务器，图表和指标传回本地。

修正前后每SNR均61440比特；原Q24/L176为122880比特，原布局数据吞吐量为两倍。
三方案比较需同时保留这一开销差异，不能仅把曲线差异归因于协方差。
