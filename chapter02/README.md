# 第二章：多 TC 与 DMA

第一章只用一个 TensorCore，数据只在 HBM、TC VMEM、SMEM 与 TC VREG 之间移动。本章扩展到一颗芯片的两个 TensorCore、两个 TensorCore 共享的 Megacore Shared CMEM、多颗芯片，以及主机内存。每一节先讲硬件允许的操作：数据能从哪里到哪里、用什么同步、开销多大、彼此怎样竞争；再据此设计流水线、集合通信和大矩阵乘法这类算法。算法不是标准答案，而是用硬件参数推导出的一种设计。

读完本章，应当能够：

- 用 `TensorCoreMesh(num_cores=2)` 写出两个 TensorCore 分工的 kernel，知道入口与出口汇合的开销与去除条件；
- 用 DMA 的固定开销与带宽设计双缓冲流水线，选择 tile 的大小；
- 使用 Megacore Shared CMEM，并在公开接口不支持时用 tpuasm 直接访问；
- 在 TensorCore 之间、芯片之间发起 remote DMA，用信号量建立正确的先后顺序；
- 按芯片的物理拓扑安排通信，由开销模型比较不同的集合通信设计；
- 让 kernel 读写主机内存，并知道这条通路的开销；
- 读懂 XLA 对大矩阵乘法的安排，并写出不慢于它的 Pallas kernel。

本章引用的时延来自 [tpu-v4-latency-numbers](../../tpu-v4-latency-numbers/README.md)；它们是怎样测出来的，第三章详细介绍。

## 本章目录

- [01 · Megacore：一颗芯片两个 TensorCore](01_megacore/README.md)
- [02 · DMA 的开销与软件流水线](02_dma_pipeline/README.md)
- [03 · Megacore Shared CMEM](03_cmem/README.md)
- [04 · TensorCore 之间的 remote DMA 与同步](04_core_remote_dma/README.md)
- [05 · 多芯片：shard_map、拓扑与 ICI](05_multi_chip/README.md)
- [06 · 跨芯片的 Megacore Shared CMEM](06_remote_cmem/README.md)
- [07 · 由硬件推导集合通信](07_collectives/README.md)
- [08 · 主机内存](08_host_memory/README.md)
- [09 · 综合：单芯片双 TC 大矩阵乘法](09_large_matmul/README.md)

## 本章发现的硬件能力与工具链缺口

| 位置 | 现象 | 处理 |
| --- | --- | --- |
| 第 1 节 | 两个 TensorCore 互不依赖时，入口汇合是多余的等待 | 用 tpuasm 把入口汇合的三条指令换成 `vnop` |
| 第 3 节 | Mosaic 不允许在 TPU v4 上分配 CMEM scratch | 用 tpuasm 手写 CMEM 的 DMA 与 `cld` |
| 第 6 节 | 公开接口不能表达 CMEM → 另一颗芯片 CMEM 的 remote DMA；只改端点会使对方停机 | 用 tpuasm 改写端点，并清除 `ici_dest` 中的 TensorCore 字段 |
| 第 6 节 | tpuasm 的 canonical 重新编码改变了 102 个 bundle 中不出现在文本里的位，程序挂起 | 改写时只重新编码被修改的 bundle |
| 第 8 节 | 运行时页号写主机内存触发 LLO_CHECK（JAX issue #40200） | 用 tpuasm 把页号改为从 SMEM 读出的寄存器 |
| 第 9 节 | 一次 `jnp.dot` 的操作数整个展开为 TC VREG，块大时溢出区超过 TC VMEM；512 行一次的程序无法序列化，读不到清单 | 每次 `jnp.dot` 只算 256 或 512 行 |
