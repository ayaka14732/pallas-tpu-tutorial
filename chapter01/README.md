# 第一章：基本操作的 TC 实现

本章只用一颗芯片上的一个 TensorCore，研究每种基本操作在硬件上由哪些指令完成。每节先给出一个最小、完整的 kernel，再一次只改变一个条件：dtype、shape、操作轴、对齐或写法，观察哪些指令随之改变、代价从哪里增加，并与原生 XLA 生成的指令对照。

读完本章，应当能够：

- 写出一个完整的单 TensorCore Pallas kernel，并读懂它的机器清单；
- 知道 HBM、TC VMEM、TC VREG、SMEM 之间的数据怎样移动，以及 DMA 的粒度与限制；
- 知道向量单元、EUP、XLU、MXU 各自能做什么、怎样取回结果，哪些操作是一条指令、哪些要拼出来；
- 遇到 Mosaic 拒绝编译或公开接口出错时，能用 tpuasm 判断硬件本身能否做到，并在需要时直接改写机器清单。

本章不讨论运行时间。清单给出的是指令条数和 bundle 数；它们怎样换算成周期，第三章再讲。也不讨论第二个 TensorCore、Megacore Shared CMEM 和多芯片，这些是第二章的内容。正文遇到这些概念时，会先给出一句“暂且可以理解为”，并指明详细介绍的位置。

## 本章目录

- [01 · 第一个 Pallas TPU kernel](01_first_kernel/README.md)
- [02 · TensorCore 与机器清单](02_machine_listing/README.md)
- [03 · HBM 与 TC VMEM 之间的 DMA](03_local_dma/README.md)
- [04 · 向量布局与打包](04_vector_layout/README.md)
- [05 · 类型转换](05_dtype_conversion/README.md)
- [06 · 逐元素运算](06_elementwise/README.md)
- [07 · 标量单元与控制流](07_scalar_control_flow/README.md)
- [08 · gather 与 scatter](08_gather_scatter/README.md)
- [09 · 矩阵转置](09_transpose_xlu/README.md)
- [10 · 矩阵乘法与 MXU](10_matmul_mxu/README.md)
- [11 · 归约](11_reduction/README.md)
- [12 · 前缀扫描](12_prefix_scan/README.md)
- [13 · 拼接](13_concatenate/README.md)
- [14 · Top-k](14_top_k/README.md)

## 本章发现的硬件能力与工具链缺口

以下几处，公开接口的行为与硬件能力不一致，正文分别给出了实验与处理方法：

| 位置 | 现象 | 处理 |
| --- | --- | --- |
| 第 3 节 | 多列 tile 数组的非对齐行窗口被 Mosaic 拒绝 | 用 tpuasm 改写为两条 `dma.strided` |
| 第 6 节 | 无符号整数的 `maximum` 无法编译；32 bit 整数乘法要 33 条指令 | 改用有符号比较；设计算法时避开整数乘法 |
| 第 8 节 | sublane 方向的 gather 被拒绝 | 用 sublane 循环移位与选择合成 |
| 第 10 节 | `matmul_push_rhs`/`matmul_lhs_fifo` 在 TPU v4 上结果错误；int8 矩阵乘法无法编译 | 用 tpuasm 插入 `vdwg` 修正；int8 尚未处理 |
| 第 12 节 | `jnp.cumsum` 无法降低 | 手写扫描，或用 `stride=0` 的逐行广播 |
| 第 14 节 | `top_k` 在有效值少于 k 个时返回重复下标 | 手写：用掩码记录已选位置，代价与 `top_k` 相当 |
