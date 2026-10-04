# 附录：术语中英文对照

正文中的术语以中文为主，硬件单元、内存和指令的名字保留英文原名。下表按主题列出正文使用的写法、对应的英文和首次详细介绍的位置。带 * 的中文说法是本教程自己的叫法，没有通行的英文名，英文一栏给出的是意思相近的写法。

## 芯片与内存

| 正文写法 | 英文 | 含义 | 位置 |
| --- | --- | --- | --- |
| 芯片 | chip | 一颗 TPU v4 芯片，含两个 TensorCore | 第一章第 1 节 |
| TensorCore、TC | TensorCore | 芯片上执行 kernel 的处理器核心 | 第一章第 1 节 |
| Megacore | megacore | 一颗芯片作为一个 device、两个 TensorCore 执行同一份程序的组织方式 | 第二章第 1 节 |
| split-chip | split-chip | 一颗芯片的两个 TensorCore 各作为一个 device 的组织方式 | 第二章第 1 节 |
| HBM | high-bandwidth memory | 芯片的主存，JAX 数组所在的内存 | 第一章第 1、3 节 |
| TC VMEM | vector memory | 每个 TensorCore 私有的片上内存，16 MiB | 第一章第 3 节 |
| TC VREG | vector register | 向量寄存器 `v0`–`v31`，每个 8×128 个 32 bit | 第一章第 4 节 |
| SMEM | scalar memory | 标量单元使用的内存，1 MiB | 第一章第 7 节 |
| Megacore Shared CMEM | common memory | 一颗芯片上两个 TensorCore 共享的片上内存，128 MiB | 第二章第 3 节 |
| 标量寄存器 | scalar register | `s0`–`s31`，32 位 | 第一章第 7 节 |
| 谓词寄存器 | predicate register | `p0`–`p14`，存放一个比较结果，用于条件执行 | 第一章第 2 节 |
| 向量掩码寄存器 | vector mask register | `vm0`–`vm7`，每个元素一个 bit | 第一章第 4、6 节 |
| 同步标志 | sync flag（sflag） | 用于等待的计数器；Pallas 的信号量就是其中之一 | 第一章第 2、3 节 |
| 信号量 | semaphore | Pallas 对同步标志的抽象 | 第一章第 1 节 |
| 主机 | host | 管理芯片的 CPU 机器 | 第二章第 8 节 |
| 切片 | slice | 分配给一个作业的一组芯片，例如 TPU v4 (2x2x2) | 第二章第 5 节 |
| ICI | inter-chip interconnect | 芯片之间的互联链路 | 第二章第 5 节 |

## 数据布局

| 正文写法 | 英文 | 含义 | 位置 |
| --- | --- | --- | --- |
| tile | tile | 数组切成的 8×128 的块，一个 f32 tile 装满一个 TC VREG | 第一章第 1、4 节 |
| sublane、子通道 | sublane | TC VREG 的行方向，共 8 个 | 第一章第 4 节 |
| lane、通道 | lane | TC VREG 的列方向，共 128 个 | 第一章第 4 节 |
| granule | granule | DMA 的长度单位，512 B | 第一章第 3 节 |
| 布局 | layout | 数组在内存中的存放方式，如 `T(8,128)` | 第一章第 1 节 |
| 打包 | packing | 两个 16 bit 或四个 8 bit 元素共用一个 32 bit 位置 | 第一章第 4、5 节 |
| 窗口 | window | Ref 的一部分，`ref.at[...]` | 第一章第 3 节 |

## 指令与执行

| 正文写法 | 英文 | 含义 | 位置 |
| --- | --- | --- | --- |
| 机器清单、清单 | listing | tpuasm 反汇编出的程序文本 | 第一章第 2 节 |
| bundle、指令包 | bundle | 一条 VLIW 长指令字，最多 12 条指令 | 第一章第 2 节 |
| 发射槽、槽 | slot | bundle 中的一个位置，连着一类硬件单元 | 第一章第 2 节 |
| 助记符 | mnemonic | 指令的名字，如 `vmul.8x128.f32` | 第一章第 2 节 |
| 立即数 | immediate | 写在指令中的常数 | 第一章第 2 节 |
| 谓词 | predicate | `@p0` 这样的条件执行前缀 | 第一章第 2 节 |
| 标量单元 | scalar unit | 做地址计算、控制流，发起 DMA | 第一章第 7 节 |
| 向量单元 | vector unit | 在 TC VREG 上做逐元素运算 | 第一章第 6 节 |
| EUP | extended unary pipeline | 计算倒数、指数、对数等超越函数的单元 | 第一章第 6 节 |
| XLU、跨通道单元 | cross-lane unit | 做转置、lane 方向的重排与归约的单元 | 第一章第 8、9 节 |
| MXU、矩阵乘法单元 | matrix multiply unit | 做矩阵乘法的单元 | 第一章第 10 节 |
| 提交—取回 * | push / pop | 先把操作送入某个单元，再从结果队列取回结果 | 第一章第 6 节 |
| 结果队列 | result FIFO | 各单元存放结果、等待 `vpop` 的队列 | 第一章第 6 节 |
| 发射 | issue | 把指令交给执行单元；分为标量发射与向量发射 | 第三章第 4 节 |
| 发射间隔 * | issue interval | 同一单元相邻两条指令的最小距离 | 第三章第 5 节 |
| VIF、向量发射队列 | vector issue FIFO | 标量发射之后、向量发射之前，向量指令排队的队列 | 第三章第 4 节 |
| 计分板 | scoreboard | 硬件记录各寄存器、各单元何时就绪的机构 | 第三章第 5 节 |
| 发射模型 * | issue model | 从清单推算周期数的模型 | 第三章第 5 节 |
| 载体 * | carrier | 供 tpuasm 插入或改写指令的已编译 kernel | 第一章第 3 节 |

## 数据搬运与同步

| 正文写法 | 英文 | 含义 | 位置 |
| --- | --- | --- | --- |
| DMA | direct memory access | 在内存之间搬运数据的机构 | 第一章第 3 节 |
| remote DMA | remote DMA | 目的在另一个 TensorCore 或另一颗芯片上的 DMA | 第二章第 4、5 节 |
| 固定开销 | fixed overhead | 一次 DMA 与数据量无关的那部分时间 | 第二章第 2 节 |
| 双缓冲 | double buffering | 用两个 buffer 让搬运与计算重叠 | 第二章第 2 节 |
| 软件流水线 | software pipelining | 把搬入、计算、搬出错开重叠的写法 | 第二章第 2 节 |
| 汇合 | barrier | 几方都到达同一位置之后才继续 | 第二章第 1 节 |
| 集合通信 | collective communication | all-reduce、all-gather 等多芯片之间的数据交换 | 第二章第 7 节 |
| 跨程序预取 | cross-program prefetch | XLA 在程序开始前把输入提前搬进片上内存 | 第三章第 2 节 |
| 越界检查 | bounds check | DMA 地址超出 buffer 时停机的检查 | 第一章第 1 节 |
| 融合、fusion | fusion | XLA 把几个运算合成一段代码 | 第一章第 2 节 |

## 计时

| 正文写法 | 英文 | 含义 | 位置 |
| --- | --- | --- | --- |
| 周期 | cycle | TensorCore 时钟的一个周期，约 0.95 ns | 第三章第 3 节 |
| LCC、本地周期计数器 | local cycle counter | 每个 TensorCore 的周期计数器 | 第三章第 3 节 |
| GTC、全局时间计数器 | global time counter | 整个切片共用的时间计数器 | 第三章第 6 节 |
| 时钟同步树 | clock synchronization tree | 各芯片的 GTC 沿 ICI 链路对齐到时间源的树 | 第三章第 6 节 |
| 时间源 | global leader | 同步树的根，GTC 由它自己的时钟推进 | 第三章第 6 节 |
| 跟随者 * | follower | 同步树上时间源之外的芯片 | 第三章第 6 节 |
| 偏移（GTC 的低 4 位） | offset | 高位不变时区分本地周期的序号 | 第三章第 6 节 |
| XProf | XProf | JAX 自带的 profiler | 第三章第 7 节 |
| trace 标记 | trace marker（`vtrace`） | 记录一个操作数与当时 GTC 的指令 | 第三章第 7 节 |

## 数值与随机数

| 正文写法 | 英文 | 含义 | 位置 |
| --- | --- | --- | --- |
| 舍入 | rounding | 就近舍入、偶数优先为 round to nearest, ties to even | 第一章第 5 节 |
| 随机舍入 | stochastic rounding | 按小数部分为概率进位，期望等于原值 | 第一章第 5 节 |
| 饱和 | saturation | 超出范围时取最近的端点 | 第一章第 5 节 |
| 归约 | reduction | 沿一个轴求和、求最大值等 | 第一章第 11 节 |
| 前缀扫描 | prefix scan | 累加和这类前缀运算 | 第一章第 12 节 |
| 硬件随机数生成器 | hardware PRNG | 向量单元内置的伪随机数生成器 | 第四章第 1 节 |
| 计数器式生成器 | counter-based generator | 由计数器和密钥计算随机数的生成器，如 threefry | 第四章第 4 节 |
