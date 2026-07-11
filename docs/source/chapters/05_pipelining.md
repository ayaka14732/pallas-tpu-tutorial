# 第 5 章：软件流水线 (Software Pipelining)

流水线是 TPU 极致性能的秘密武器。在没有流水线的情况下，TPU 会等待数据从 HBM 拷贝到 VMEM，然后执行计算，最后再把结果拷贝回 HBM。这种串行执行会导致计算单元（MXU/VPU）在等待内存传输时处于空闲状态。

本章我们将学习如何使用双缓冲（Double-buffering）和 `emit_pipeline` 来重叠计算与内存传输。

## 什么是双缓冲 (Double-buffering)？

假设我们要处理一个包含 10 个数据块的序列。

**串行执行：**
1. DMA 拷贝块 0
2. 计算块 0
3. DMA 拷贝块 1
4. 计算块 1 ...

**双缓冲执行：**
我们在 VMEM 中分配**两个**缓冲区（Buffer A 和 Buffer B）。
1. 初始：DMA 拷贝块 0 到 Buffer A
2. 循环开始：
   - 当计算单元在处理 Buffer A（块 0）时，DMA 引擎**同时**将块 1 拷贝到 Buffer B。
   - 当计算单元在处理 Buffer B（块 1）时，DMA 引擎**同时**将块 2 拷贝到 Buffer A。

通过这种方式，内存传输的延迟被完全隐藏在了计算时间之中。

## Pallas 中的自动流水线

好消息是，对于简单的 `pallas_call`，JAX 编译器通常会自动为你应用流水线优化。只要你的 `Grid` 遍历顺序是连续的，编译器会尝试预取下一个块。

但在复杂的算子中（比如我们马上要写的矩阵乘法），我们需要显式地控制流水线。

## 使用 `emit_pipeline` 构建嵌套流水线

在 TPU 上实现高性能矩阵乘法（MatMul）的标准模式是：在外部 `pallas_call` 的 Kernel 内部，使用 `pltpu.emit_pipeline` 启动一个内部流水线，专门用于沿着归约维度（Reduction dimension，即 K 维度）进行分块累加。

### 为什么需要嵌套流水线？

在 MatMul `C = A @ B` 中，计算一个输出块 `C[i, j]` 需要遍历 `A` 的第 `i` 行块和 `B` 的第 `j` 列块。
如果我们将 K 维度也放在外部的 `pallas_call` Grid 中，那么每次迭代都会触发 C 块从 HBM 加载和写回，这会产生巨大的 HBM 带宽开销。

**正确做法：**
1. 外部 `pallas_call` Grid 负责 `(M, N)` 维度。
2. 外部 Kernel 分配一个 VMEM 累加器（Scratch buffer），初始化为 0。
3. 外部 Kernel 内部调用 `emit_pipeline`，沿着 `K` 维度循环。
4. 内部流水线不断将 `A` 和 `B` 的切块加载到 VMEM，相乘并累加到 VMEM 累加器中。
5. 内部流水线结束后，将累加器的最终结果写回 HBM 的 `C` 块。

### `emit_pipeline` 的基本语法

```python
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as pltpu

def outer_kernel(a_ref, b_ref, c_ref, acc_scratch):
    # a_ref 和 b_ref 这里不再是普通数据块，而是包含了整个 K 维度的"大块"
    
    # 初始化累加器
    acc_scratch[...] = 0.0
    
    def body_fn(k_idx, a_vmem_ref, b_vmem_ref):
        # 这个函数在内部流水线中执行
        # a_vmem_ref 和 b_vmem_ref 是已经被 DMA 预取到 VMEM 的切块
        acc_scratch[...] += a_vmem_ref[...] @ b_vmem_ref[...]
        
    # 启动流水线
    pltpu.emit_pipeline(
        body_fn,
        num_iterations=K_BLOCKS,
        # 告诉编译器使用双缓冲预取 a_ref 和 b_ref
        inputs=[
            pl.Buffered(a_ref, buffer_count=2),
            pl.Buffered(b_ref, buffer_count=2)
        ]
    )
    
    # 流水线结束后，将累加结果写回输出
    c_ref[...] = acc_scratch[...]
```

在这个例子中，`pl.Buffered(..., buffer_count=2)` 是关键。它告诉编译器在 VMEM 中为这个输入分配两个缓冲区，并自动生成底层的 DMA 异步拷贝和信号量（Semaphore）同步指令。你只需要专注于编写 `body_fn` 中的计算逻辑。

在下一章中，我们将把这些概念结合起来，实现一个达到硬件峰值性能的 TPU 矩阵乘法 Kernel。
