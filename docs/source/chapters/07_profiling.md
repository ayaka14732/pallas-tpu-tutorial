# 第 7 章：性能分析与调优

编写 Kernel 只是第一步。要让 Kernel 达到硬件的峰值性能，我们需要学会使用 Profiler 来识别瓶颈。在 TPU 上，性能调优的方法论与 GPU 类似，但关注的具体指标和工具链有所不同。

本章将介绍如何在 TPU 上进行性能分析，并引入指导性能优化的核心理论：**Roofline 模型**。

## 使用 `interpret=True` 进行正确性验证

在 TPU 上调试 Kernel 的第一步，绝不是直接看性能，而是验证正确性。由于 TPU 编译器（Mosaic）在后台做了大量复杂的降级（Lowering）工作，如果代码有 bug，直接在 TPU 上运行可能会导致静默错误或难以理解的硬件挂起。

Pallas 提供了一个非常强大的调试工具：`interpret=True`。它会让 Kernel 在 CPU 上以纯 Python 解释模式执行，跳过所有底层硬件指令的生成。

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

**为什么这很重要？**
在 `interpret=True` 模式下，你可以使用标准的 Python 调试器（如 `pdb`），甚至可以直接在 Kernel 函数中插入 `print()` 语句！这在真实的 TPU 执行中是绝对不可能的。一旦在 CPU 上验证了逻辑的正确性，再去掉 `interpret=True` 交给 TPU 编译。

## JAX Profiler 与 TensorBoard

验证正确性后，我们需要分析性能。JAX 内置了与 TensorBoard 深度集成的 Profiler。它可以捕获 TPU 上的执行轨迹（Trace），展示 HBM 读写、MXU 利用率和流水线状态。

### 捕获 Trace

```python
import jax

# 1. 预热 (Warmup)：触发 JIT 编译，避免将编译时间计入性能分析
z = jax.jit(my_pallas_fn)(inputs)
z.block_until_ready() # 确保异步执行完成

# 2. 捕获 Trace
with jax.profiler.trace("/tmp/jax-trace"):
    for _ in range(10): # 运行多次以获得稳定的平均值
        z = jax.jit(my_pallas_fn)(inputs)
        z.block_until_ready()
```

### 查看 Trace

捕获后，启动 TensorBoard 并加载 Trace 目录：
```bash
pip install tensorboard-plugin-profile
tensorboard --logdir=/tmp/jax-trace
```
在浏览器中打开 TensorBoard，导航到 "Profile" 选项卡，查看 "Trace Viewer"。

## Roofline 模型：寻找性能天花板

在分析 Trace 时，你可能会问："这个算子到底能跑多快？现在的瓶颈是计算还是内存？" 
**Roofline 模型**是回答这个问题的标准理论框架。

对于任何 Kernel，我们都可以计算它的**算术强度（Arithmetic Intensity, AI）**：

$$ \text{AI} = \frac{\text{总计算量 (FLOPs)}}{\text{总内存传输量 (Bytes Transferred)}} $$

AI 衡量了每从 HBM 搬运 1 Byte 数据，我们能执行多少次浮点运算。

硬件也有自己的两个关键指标：
1. **理论峰值算力 ($\pi$)**：例如 TPU v4 的 MXU 峰值为 275 TFLOPs。
2. **理论峰值内存带宽 ($\beta$)**：例如 TPU v4 的 HBM 带宽为 1200 GB/s。

硬件的**机器平衡点 (Machine Balance)** 定义为 $\frac{\pi}{\beta}$。对于 TPU v4，这个值大约是 $275 \times 10^{12} / 1200 \times 10^9 \approx 229$ FLOPs/Byte。

### 性能瓶颈的判定

将你的 Kernel 的 AI 与硬件的机器平衡点进行比较：

1. **如果 AI < 机器平衡点 (Memory-bound)**：
   你的 Kernel 是**内存密集型**的。它的性能被 HBM 带宽限制，也就是撞到了 Roofline 模型的"斜屋顶"。
   - *典型算子*：RMSNorm, Softmax, Element-wise add, Vector-Matrix Multiply (GEMV)。
   - *优化方向*：减少 HBM 读写次数。例如，算子融合（将多个操作合并到一个 Kernel 中，数据留在 VMEM 中传递）、提高缓存命中率。

2. **如果 AI > 机器平衡点 (Compute-bound)**：
   你的 Kernel 是**计算密集型**的。它的性能被 MXU 算力限制，也就是撞到了 Roofline 模型的"平屋顶"。
   - *典型算子*：Matrix-Matrix Multiply (GEMM, MatMul), 大规模卷积。
   - *优化方向*：提高 MXU 利用率。例如，增大 Tile 大小（如 `BM`, `BN`），确保维度是 128 的倍数，消除流水线气泡（Pipeline bubbles）。

## 关键性能指标排查清单

在 Trace Viewer 中，你应该重点关注以下几个指标：

| 现象 | 可能的原因 | 解决策略 |
| :--- | :--- | :--- |
| **HBM 带宽利用率极低，且计算有大量空隙** | 流水线未生效，或者 DMA 延迟未被隐藏 | 检查 `emit_pipeline` 的 `buffer_count`，或者尝试增大 Block 大小以增加单次计算耗时。 |
| **MXU 利用率低 (Compute-bound 算子)** | Tile 未对齐 128，导致大量 Padding | 调整 `BlockSpec` 的形状，确保最后两维是 128 的倍数。 |
| **VMEM 溢出 (OOM)** | Scratch Buffer 或 Pipeline Buffer 太大 | 减小 `BM`, `BN`，或者减少 `buffer_count`。 |
| **寄存器溢出 (Register Spilling)** | 内部循环中局部变量太多，或者存在大小为 1 的冗余维度 | 移除冗余的单元素维度（使用 `None` squeeze），简化内部逻辑。 |

掌握了 Roofline 模型和 Profiler 工具，你就不再是盲目地试错，而是能够精确地定位瓶颈并有的放矢地优化。在接下来的章节中，我们将实战分析几个经典的内存密集型算子（RMSNorm 和 Softmax），看看如何在 TPU 上突破内存墙。
