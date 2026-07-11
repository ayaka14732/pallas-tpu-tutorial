# 第 7 章：性能分析与调优

编写 Kernel 只是第一步。要让 Kernel 达到硬件的峰值性能，我们需要学会使用 Profiler 来识别瓶颈。本章将介绍如何在 TPU 上进行性能分析。

## 使用 `interpret=True` 进行正确性验证

在 TPU 上调试 Kernel 的第一步，是使用 `interpret=True` 参数运行 `pallas_call`。这会让 Kernel 在 CPU 上以解释模式执行，方便使用标准的 Python 调试工具。

```python
result = pl.pallas_call(
    my_kernel,
    out_shape=...,
    in_specs=...,
    out_specs=...,
    grid=...,
    interpret=True  # 在 CPU 上模拟执行，方便调试
)(inputs)
```

## JAX Profiler

JAX 内置了与 TensorBoard 集成的 Profiler。你可以使用 `jax.profiler.trace` 来捕获 TPU 上的执行轨迹。

```python
with jax.profiler.trace("/tmp/jax-trace"):
    result = jax.jit(my_pallas_fn)(inputs)
    result.block_until_ready()
```

## 关键性能指标

在分析 TPU Kernel 性能时，需要关注以下指标：

| 指标 | 含义 | 优化方向 |
| :--- | :--- | :--- |
| MXU 利用率 | 矩阵乘法单元的活跃时间占比 | 增大 Tile 大小，减少 DMA 等待 |
| HBM 带宽利用率 | 实际 HBM 吞吐 / 理论峰值 | 使用流水线隐藏延迟 |
| VMEM 溢出 | 是否有寄存器溢出到 VMEM | 减小 Block 大小或中间变量数量 |
| Pipeline 气泡 | 流水线中的空闲周期 | 调整 buffer_count，平衡计算与传输时间 |

## Roofline 模型

对于任何 Kernel，你都可以计算它的**算术强度（Arithmetic Intensity）**：

$$ \text{AI} = \frac{\text{FLOPs}}{\text{Bytes Transferred}} $$

如果 AI 高于 TPU 的 Compute/Bandwidth 比值（即 Roofline 的拐点），那么你的 Kernel 是**计算密集型（Compute-bound）**的，瓶颈在 MXU。否则，它是**内存密集型（Memory-bound）**的，瓶颈在 HBM 带宽。

矩阵乘法通常是计算密集型的，而 RMSNorm、Softmax 等逐元素操作通常是内存密集型的。
