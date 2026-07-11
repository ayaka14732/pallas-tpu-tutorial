# Pallas TPU Kernel 开发教程

从零开始学习 JAX Pallas TPU Kernel 开发 — 面向有 JAX 经验但无 kernel 开发经验的开发者。

## 目标读者

- 熟悉 JAX 基本用法（`jax.jit`、`jax.vmap`、`jax.numpy`）
- 了解 Python / NumPy 编程
- 不需要 CUDA 或 C++ 经验

## 教程结构

### 第一部分：基础概念

| 章节 | 标题 | 核心知识点 |
| :--- | :--- | :--- |
| 1 | TPU 硬件架构与执行模型 | TPU vs GPU 本质区别、内存层级（HBM/VMEM/SMEM/VREG）、8x128 Tiling、顺序执行 |
| 2 | Pallas Hello World | `pallas_call`、Kernel 函数、Ref 类型、向量加法 |
| 3 | Grid 与 BlockSpec 深入解析 | 循环心智模型、`index_map`、块索引 vs 元素切片、广播、squeeze、越界填充 |
| 4 | TPU 内存空间与分配 | `memory_space` 标注、Scratch Buffers、信号量、生命周期 |

### 第二部分：流水线与性能优化

| 章节 | 标题 | 核心知识点 |
| :--- | :--- | :--- |
| 5 | 软件流水线 | 双缓冲、`emit_pipeline`、`pl.Buffered`、计算与传输重叠 |
| 6 | 实战矩阵乘法 | 分块矩阵乘法、MXU 精度控制、VMEM 预算、嵌套流水线 |
| 7 | 性能分析与调优 | `interpret=True`、JAX Profiler、Roofline 模型、关键指标 |

### 第三部分：经典算子实现

| 章节 | 标题 | 核心知识点 |
| :--- | :--- | :--- |
| 8 | RMSNorm | 内存密集型算子、归约性能、多行批处理、算子融合 |
| 9 | Softmax | 数值稳定性、在线 Softmax、Running max/sum |
| 10 | 标量预取与动态稀疏索引 | `PrefetchScalarGridSpec`、SMEM、动态块提取、手动 DMA |

### 第四部分：进阶主题

| 章节 | 标题 | 核心知识点 |
| :--- | :--- | :--- |
| 11 | FlashAttention | 分块注意力、在线 Softmax + 累加器、BlockSizes、反向传播 |
| 12 | 分布式 Kernel 与多核编程 | Megacore、`dimension_semantics`、`core_map`、Remote DMA、集合通信 |
| 13 | 源码剖析：Ragged Paged Attention | 生产级 Kernel 全解析：手动 DMA、`while_loop`、双缓冲、动态分页 |

## 本地构建文档

```bash
cd docs
pip install -r requirements.txt
sphinx-build -b html source build
# 打开 build/index.html 查看
```

## 参考资源

- [JAX Pallas 官方文档](https://docs.jax.dev/en/latest/pallas/index.html)
- [JAX GitHub 仓库](https://github.com/jax-ml/jax)（Pallas TPU tests 和 production kernels）
- [Ragged Paged Attention 源码](https://github.com/jax-ml/jax/tree/main/jax/experimental/pallas/ops/tpu/ragged_paged_attention)

## 许可证

本教程基于 Apache License 2.0 发布。
