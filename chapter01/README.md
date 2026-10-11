# 第一章：基本操作的 TC 实现

本章只用一颗芯片上的一个 TensorCore，研究每种基本操作在硬件上由哪些指令完成、要多少周期。每节先给出一个最小、完整的 kernel，再一次只改变一个条件：dtype、shape、操作轴、对齐或写法，观察哪些指令随之改变、周期数从哪里增加，并与原生 XLA 生成的指令和周期数对照。

周期数从第 3 节开始贯穿全章。TensorCore 上的一切都按时钟周期进行，一段代码要多少周期直接在设备上读出来：数据已经在 TC VMEM 中，做完这一步结果也留在 TC VMEM 中，中间隔了多少个周期。每认识一个硬件单元，就同时测出它的延迟和发射间隔；到最后一节，用这些数完整地分析一个 kernel。

读完本章，应当能够：

- 写出一个完整的单 TensorCore Pallas kernel，并读懂它的机器清单；
- 用设备上的周期计数器 LCC 测一段手写的指令或一段编译出的计算，把原生 XLA 与 Pallas kernel 放进同一个 TC VMEM 边界比较；
- 知道 HBM、TC VMEM、TC VREG、SMEM 之间的数据怎样移动，DMA 的粒度、限制与开销；
- 知道向量单元、EUP、XLU、MXU 各自能做什么、怎样取回结果、延迟和发射间隔各是多少，哪些操作是一条指令、哪些要拼出来；
- 看到一个周期数，能判断它受哪一类资源限制：往返延迟、发射间隔，还是指令的条数与依赖链；
- 遇到 Mosaic 拒绝编译或公开接口出错时，能用 tpuasm 判断硬件本身能否做到，并在需要时直接改写机器清单。

本章不讨论第二个 TensorCore、Megacore Shared CMEM 和多芯片，这些是第二章的内容；计时方法背后的规则（两级发射的细节、发射模型、全局时钟）在第三章。正文遇到这些概念时，会先给出一句“暂且可以理解为”，并指明详细介绍的位置。

## 本章目录

- [01 · 第一个 Pallas TPU kernel](01_first_kernel/README.md)
- [02 · TensorCore 与机器清单](02_machine_listing/README.md)
- [03 · 周期数：用 LCC 计时](03_cycle_counting/README.md)
- [04 · HBM 与 TC VMEM 之间的 DMA](04_local_dma/README.md)
- [05 · 向量布局与打包](05_vector_layout/README.md)
- [06 · 类型转换](06_dtype_conversion/README.md)
- [07 · 逐元素运算](07_elementwise/README.md)
- [08 · 标量单元与控制流](08_scalar_control_flow/README.md)
- [09 · gather 与 scatter](09_gather_scatter/README.md)
- [10 · 矩阵转置](10_transpose_xlu/README.md)
- [11 · 矩阵乘法与 MXU](11_matmul_mxu/README.md)
- [12 · 归约](12_reduction/README.md)
- [13 · 前缀扫描](13_prefix_scan/README.md)
- [14 · 拼接](14_concatenate/README.md)
- [15 · Top-k](15_top_k/README.md)

## 本章测得的周期数

硬件单元的参数。“延迟”是结果可以被下一条指令使用，或者可以从结果队列取回之前的周期数；“间隔”是同一个单元（或同一个队列）相邻两次提交之间的最小周期数：

| 单元或通路 | 延迟 | 间隔 | 位置 |
| --- | ---: | ---: | --- |
| 发射一个 bundle | — | 1 | 第 3 节 |
| 向量 ALU：整数运算、位运算、移位、比较、选择、`vmax`/`vmin` | 1 | 1 | 第 7 节 |
| 向量 ALU：f32 的加、减、乘，`vcvt`，打包与展开，`vrot.slane.down` | 2 | 1 | 第 7、12 节 |
| EUP：`vpow2`、`vlog2`、`vrcp`、`vrsqrt`、`vtanh` | 7 | 2 | 第 7 节 |
| 标量：`sld` 读 SMEM | 4 | 4 | 第 8 节 |
| XLU：通道重排 `vperm`、循环移位 `vrot` | 69 | 8 | 第 9 节 |
| XLU：转置，每提交或取回一个 TC VREG | — | 8 | 第 10 节 |
| MXU：`vmatmul` | 83（MXU2、MXU3 为 101） | 8，取回也是 8 | 第 11 节 |
| XLU：跨通道归约 `vadd.xlane`、`vmax.xlane`、`vmax.index.xlane` | 79 | 8 | 第 12、15 节 |
| 向量 → 标量：`vpush` → `spop` | 43 | — | 第 12 节 |
| 按索引读写：`vld.iar`、`vst.iar`；装入索引 `vsetiar` | 1；多 4 个周期 | 1 | 第 9 节 |
| DMA，HBM → TC VMEM | 约 460 + 每 KiB 1.1 | — | 第 4 节 |
| DMA，TC VMEM → HBM | 约 410 + 每 KiB 1.1 | — | 第 4 节 |

TC VMEM 边界内的周期数（第 3 节的测法；每个数包含外壳固定的 19 至 39 个周期），只列有代表性的几项：

| 运算 | 输入 | 原生 XLA | Pallas | 位置 |
| --- | --- | ---: | ---: | --- |
| `x * 2` | `f32[8,128]`、`f32[1024,128]` | 42、170 | 42、170 | 第 3 节 |
| int8 → int32 | `int8[64,128]` | 59 | 114 | 第 6 节 |
| 64 个 tile 逐个乘 2：展开，`pl.loop` | `f32[512,128]` | — | 106，424 | 第 8 节 |
| lane gather | `f32[8,128]` | 9213 | 101 | 第 9 节 |
| 转置：一个，两个 | `f32[128,128]` | 268，527 | 267，285 | 第 10 节 |
| 矩阵乘法 | `bf16[128,128] @ bf16[128,128]` | 211 | 211 | 第 11 节 |
| `sum(x, axis=1)` | `f32[8,128]` | 1045 | 100 | 第 12 节 |
| `sum(x, axis=0)`：一条累加链，两条 | `f32[1024,128]` | 303 | 295，180 | 第 12 节 |
| 沿通道的前缀和：Hillis–Steele，MXU | `f32[8,128]` | 430 | 531，173 | 第 13 节 |
| top-8：`lax.top_k`，抬高 | `f32[8,128]` | 2565 | 685，695 | 第 15 节 |
| top-32：`lax.top_k`，秩计数 | `f32[8,128]` | 2565 | 2653，894 | 第 15 节 |

这张表里有三种情形：两边的指令相同、周期数也相同（逐元素运算、矩阵乘法）；Pallas 选对了硬件通路，比 XLA 的通用实现快一到两个数量级（gather、沿通道的归约）；以及 XLA 更快，或者 Pallas 的直接写法并不好（int8 转换、Hillis–Steele、k 较大的 top-k）。原生 XLA 是必须越过的底线，越没越过，要靠周期数说话。

## 本章发现的硬件能力与工具链缺口

以下几处，公开接口的行为与硬件能力不一致，正文分别给出了实验与处理方法：

| 位置 | 现象 | 处理 |
| --- | --- | --- |
| 第 4 节 | 多列 tile 数组的非对齐行窗口被 Mosaic 拒绝 | 用 tpuasm 改写为两条 `dma.strided` |
| 第 7 节 | 无符号整数的 `maximum` 无法编译；32 bit 整数乘法要 33 条指令 | 改用有符号比较；设计算法时避开整数乘法 |
| 第 9 节 | 子通道方向的 gather 被拒绝 | 用子通道循环移位与选择合成（62 个周期）；或用 tpuasm 写按索引读，不含外壳只要 11 个周期 |
| 第 11 节 | `matmul_push_rhs`/`matmul_lhs_fifo` 在 TPU v4 上结果错误；int8 矩阵乘法无法编译 | 用 tpuasm 插入 `vdwg` 修正；int8 尚未处理 |
| 第 13 节 | `jnp.cumsum` 无法降低 | 手写扫描，或用 `stride=0` 的逐行广播 |
| 第 15 节 | `top_k` 在有效值少于 k 个时返回重复下标，对 NaN 与 ±0 的处理也与原生 XLA 不同 | 重复下标用“抬高”修正，只多约 10 个周期；k 较大时改用秩计数；逐位正确的写法见 top-k 论文 |
