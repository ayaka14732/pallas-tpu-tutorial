# 第 4 章：TPU 内存空间与分配

在前面的章节中，我们一直默认输入数据位于 HBM 中，而 Kernel 内部操作的是 VMEM 中的引用。但在复杂的算子（如 FlashAttention 或 MatMul）中，我们往往需要显式地控制数据存放的内存层级，并在不同的网格迭代之间传递状态。

本章将介绍如何在 Pallas 中显式指定内存空间，以及如何创建暂存缓冲区（Scratch Buffers）。

## TPU 的物理内存空间枚举

在 `jax.experimental.pallas.tpu` (通常简写为 `pltpu`) 中，定义了 TPU 的物理内存空间：

- `pltpu.VMEM`：向量内存（Vector Memory）。它是 TPU 向量计算单元（VPU 和 MXU）的工作内存。容量通常在 16MB 到 32MB 之间。**所有向量计算的输入和输出都必须在这里。**
- `pltpu.SMEM`：标量内存（Scalar Memory）。附属于标量核心，容量较小（几百 KB），但支持低延迟的 32-bit 细粒度随机访问。通常用于存放控制流变量和动态索引（详见第 10 章）。
- `pltpu.HBM`：高带宽主存。容量大（16GB+），但延迟极高。
- `pltpu.CMEM`：公共内存（Common Memory）。在某些 TPU 代次中可用，作为多核共享的 L3 缓存。
- `pl.ANY`：让编译器自行决定最佳位置。

## 在 BlockSpec 中指定内存空间

默认情况下，`pallas_call` 假设所有输入/输出都位于 HBM 中，并在调用 Kernel 前自动生成 DMA 指令将它们搬运到 VMEM。

但有时，你的输入本身就已经在 VMEM 中（例如在嵌套流水线或复杂的分布式 Kernel 中）。你可以通过在 `BlockSpec` 中指定 `memory_space` 来改变默认行为：

```python
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as pltpu

# 告诉编译器：这个输入不需要从 HBM 搬运，它已经在 VMEM 中了
vmem_block_spec = pl.BlockSpec(
    block_shape=(128, 128),
    index_map=lambda i, j: (i, j),
    memory_space=pltpu.VMEM
)
```

如果你指定了 `memory_space=pltpu.VMEM`，编译器将**不会**生成 HBM 到 VMEM 的 DMA 拷贝指令，它期望传入 `pallas_call` 的对应参数本身就是一个驻留在 VMEM 中的引用。这在第 5 章的 `emit_pipeline` 中非常关键。

## Scratch Buffers (暂存缓冲区)

在编写 Kernel 时，我们经常需要一些临时的内存空间来存储中间结果，例如：
- 矩阵乘法中的累加器（Accumulator）
- FlashAttention 中的 running max 和 running sum

这些中间结果不需要写回 HBM，它们只在 Kernel 执行期间存在于 VMEM 中。在 CUDA 中，这对应于在 Kernel 内部声明的 `__shared__` 数组。Pallas 提供了 `scratch_shapes` 参数来分配这些缓冲区。

### 分配 Scratch Buffers

在 `pallas_call` 中，你可以传入一个 `scratch_shapes` 列表：

```python
# 分配一个形状为 (128, 128) 的 float32 VMEM 缓冲区
acc_scratch = pltpu.VMEM((128, 128), jnp.float32)

# 分配一个形状为 (8,) 的 int32 SMEM 缓冲区
idx_scratch = pltpu.SMEM((8,), jnp.int32)

kernel = pl.pallas_call(
    my_kernel_fn,
    out_shape=...,
    in_specs=[...],
    out_specs=[...],
    grid=(...),
    scratch_shapes=[acc_scratch, idx_scratch]
)
```

### 在 Kernel 中使用 Scratch Buffers

`scratch_shapes` 中分配的缓冲区，会作为**额外的参数**附加在输入和输出引用之后，传递给你的 Kernel 函数。

```python
def my_kernel_fn(in_ref, out_ref, acc_ref, idx_ref):
    # in_ref, out_ref 对应 in_specs 和 out_specs
    # acc_ref 对应 acc_scratch (VMEM)
    # idx_ref 对应 idx_scratch (SMEM)
    
    # 初始化累加器
    acc_ref[...] = 0.0
    
    # ... 执行计算，将中间结果累加到 acc_ref ...
    
    # 最终结果写回 out_ref
    out_ref[...] = acc_ref[...]
```

### Scratch Buffer 的生命周期与状态传递

**极其重要的一点：** Scratch Buffer 的生命周期是**整个 Grid 执行期间**。

这意味着，当 Grid 从 `i=0` 推进到 `i=1` 时，Scratch Buffer 中的数据**会被保留，不会被清空**。

**这与 GPU 的巨大区别：** 在 CUDA 中，不同的 Thread Block 在不同的流多处理器（SM）上并行执行，它们之间的共享内存（Shared Memory）是完全隔离的。如果你想在 GPU 上跨 Block 累加数据，你必须使用全局内存的原子操作（Atomic Add），这非常慢。

但在 TPU 上，由于 Grid 保证是**按字典序顺序执行**的，同一个物理核心会依次处理 Grid 的每一步。因此，Scratch Buffer 就成为了一个天然的、超低延迟的**状态传递容器**。

最经典的例子就是沿着归约维度（Reduction dimension）进行累加：
如果 Grid 是 `(M, N, K)`，其中 `K` 是归约维度，那么对于相同的 `(M, N)`，随着 `K` 的增加，我们可以不断地向同一个 VMEM `acc_ref` 中累加数据，直到 `K` 遍历完毕，再将 `acc_ref` 写回 HBM。

这正是下一章"流水线与矩阵乘法"的核心机制。
