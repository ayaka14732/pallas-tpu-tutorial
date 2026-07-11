# 第 6 章：实战矩阵乘法 (MatMul)

矩阵乘法是深度学习中最核心的算子，也是检验硬件性能的"试金石"。在 TPU 上，矩阵乘法由专门的硬件单元——**MXU (Matrix Multiply Unit)** 执行。

本章我们将利用前面学到的 Grid、BlockSpec、Scratch Buffers 和 `emit_pipeline`，手写一个高性能的 TPU 矩阵乘法 Kernel。同时，我们将深入探讨 MXU 的工作原理和精度控制。

## MXU：脉动阵列的威力

TPU 的 MXU 本质上是一个巨大的**脉动阵列 (Systolic Array)**。以 TPU v4/v5 为例，一个 MXU 通常包含 128x128 的乘加单元（MACs）。

脉动阵列的设计使得数据像心脏泵血一样在计算单元之间有节奏地流动。当数据在阵列中流动时，每个 MAC 单元在一个时钟周期内执行一次乘法和一次累加。这种设计极大地减少了对寄存器和 VMEM 的访问次数，使得 MXU 能够提供极其恐怖的算力（FLOPs）。

### 硬件对齐约束

由于 MXU 的物理结构是 128x128（或类似的 2 的幂次方），TPU 编译器在将 `jnp.dot` 映射到 MXU 时，有着严格的对齐要求：
- 参与矩阵乘法的内部维度（即 $K$ 维度，缩减维度）通常需要是 128 的倍数。
- 外部维度（$M$ 和 $N$）也最好是 128 或 8 的倍数。
- 如果你的切块大小不满足这些要求，编译器会自动进行填充（Padding），这会导致计算资源的严重浪费。

## 算法设计与流水线

给定矩阵 $A \in \mathbb{R}^{M \times K}$ 和 $B \in \mathbb{R}^{K \times N}$，计算 $C = A \times B$。

我们将矩阵划分为大小为 `(BM, BK)` 和 `(BK, BN)` 的块。
为了避免 HBM Thrashing（第 5 章提到），我们采用**嵌套流水线**策略：

1. **外部 Grid**：大小为 `(M // BM, N // BN)`，遍历输出 $C$ 的块。
2. **Scratch Buffer**：在 VMEM 中分配大小为 `(BM, BN)` 的累加器。
3. **内部 Pipeline**：迭代 `K // BK` 次，每次取 $A_{i, k}$ 和 $B_{k, j}$ 到 VMEM，使用 MXU 相乘，并加到累加器。

## 代码实现

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def matmul_kernel(a_ref, b_ref, c_ref, acc_ref):
    """
    外部 Kernel。
    注意：a_ref 形状为 (BM, K)，b_ref 形状为 (K, BN)。
    K 维度是完整的，我们将在这里使用流水线对 K 维度进行分块读取。
    """
    BM, K = a_ref.shape
    _, BN = b_ref.shape
    BK = 128  # 我们在内部流水线中每次处理的 K 维度大小，完美对齐 MXU
    num_k_blocks = K // BK
    
    # 1. 初始化累加器 (VMEM)
    acc_ref[...] = 0.0
    
    # 2. 定义流水线内部的计算逻辑
    def body_fn(k_idx, a_vmem_block, b_vmem_block):
        # a_vmem_block 形状: (BM, BK)
        # b_vmem_block 形状: (BK, BN)
        # jnp.dot 在 TPU 上会自动映射到 MXU
        # 必须指定 preferred_element_type=jnp.float32 以利用 MXU 的 f32 累加特性
        acc_ref[...] += jnp.dot(
            a_vmem_block[...], 
            b_vmem_block[...], 
            preferred_element_type=jnp.float32
        )
        
    # 3. 启动嵌套流水线 (双缓冲)
    pltpu.emit_pipeline(
        body_fn,
        num_iterations=num_k_blocks,
        inputs=[
            # 沿着 a_ref 的第 1 维 (K 维) 分块
            pl.Buffered(a_ref, buffer_count=2, block_shape=(BM, BK), index_map=lambda k: (0, k)),
            # 沿着 b_ref 的第 0 维 (K 维) 分块
            pl.Buffered(b_ref, buffer_count=2, block_shape=(BK, BN), index_map=lambda k: (k, 0))
        ]
    )
    
    # 4. 将累加结果写回 HBM
    # 必须将 f32 的累加器转换为输出的数据类型
    c_ref[...] = acc_ref[...].astype(c_ref.dtype)


def pallas_matmul(a: jax.Array, b: jax.Array, BM=128, BN=128) -> jax.Array:
    M, K = a.shape
    _, N = b.shape
    
    # 外部 Grid 只负责 M 和 N 维度
    grid = (M // BM, N // BN)
    
    # a_spec: 对于给定的网格 (i, j)，我们需要 A 的第 i 行块，和完整的 K 维度
    a_spec = pl.BlockSpec(
        block_shape=(BM, K), 
        index_map=lambda i, j: (i, 0)
    )
    
    # b_spec: 对于给定的网格 (i, j)，我们需要完整的 K 维度，和 B 的第 j 列块
    b_spec = pl.BlockSpec(
        block_shape=(K, BN), 
        index_map=lambda i, j: (0, j)
    )
    
    # c_spec: 输出块
    c_spec = pl.BlockSpec(
        block_shape=(BM, BN), 
        index_map=lambda i, j: (i, j)
    )
    
    # 分配 VMEM 累加器，MXU 的累加必须是 float32
    acc_scratch = pltpu.VMEM((BM, BN), jnp.float32)
    
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), a.dtype),
        in_specs=[a_spec, b_spec],
        out_specs=c_spec,
        grid=grid,
        scratch_shapes=[acc_scratch]
    )(a, b)
```

## 精度控制与 bfloat16 (极度重要)

这是 TPU 与 GPU 最大的区别之一，也是许多新手踩坑的地方。

在 NVIDIA GPU (Ampere 及以后) 上，TensorCore 可以执行真正的 FP32 乘加，或者显式指定使用 TF32。
但在 TPU 上，**MXU 的乘法器（Multipliers）不支持完整的 float32**。

即使你将 32-bit 的操作数（如 `float32`）传递给 `jnp.dot`，TPU 的硬件行为是：
1. 将输入的 `float32` 强制**截断（Truncate）**为 `bfloat16`。
2. 执行 `bfloat16` $\times$ `bfloat16` 的乘法。
3. 将乘法结果**累加**到 `float32` 的累加器中。

### 为什么是 bfloat16？
`bfloat16` (Brain Floating Point) 是 Google 发明的数据格式。它具有与 `float32` 相同的指数位（8 bits），但尾数位只有 7 bits。这意味着它的**动态范围与 float32 一样大**（不容易溢出），但**精度较低**。对于深度学习来说，动态范围远比精度重要，因此 `bfloat16` 成为了理想的格式。

### 如果我真的需要完整的 float32 精度怎么办？
如果你在进行科学计算，而不是深度学习，截断到 `bfloat16` 可能会导致不可接受的误差。
在 JAX 中，你可以通过设置全局标志来强制 TPU 模拟完整的 `float32` 矩阵乘法：
```python
jax.config.update("jax_default_matmul_precision", "float32")
# 或者
jax.config.update("jax_default_matmul_precision", "tensorfloat32") # 使用 3 次 bfloat16 乘法模拟
```
但这会付出**巨大的性能代价**（通常会慢 3 到 4 倍），因为硬件必须使用多次 `bfloat16` 乘法来拼凑出 `float32` 的精度。

**最佳实践：** 在 Pallas 中编写深度学习算子时，始终接受默认的 `bfloat16` 截断行为，并在 `jnp.dot` 中显式指定 `preferred_element_type=jnp.float32`，以确保累加过程在 `float32` 精度下进行，防止累加误差放大。

## 性能优化总结

1. **Tile 对齐**：确保 `BM`, `BN`, `BK` 是 128 的倍数。
2. **转置融合**：在某些情况下，你可以将输入的转置融合到矩阵乘法中，这在 TPU 上通常是免费的，因为 XLU（Cross-Lane Unit）可以高效地在寄存器级别完成转置。
3. **VMEM 预算控制**：`BM` 和 `BN` 不能无限大。你的 VMEM 需要容纳：
   - `acc_ref`: `BM * BN * 4` bytes (float32)
   - `a_vmem_block`: 双缓冲 `2 * BM * BK * 2` bytes (bfloat16)
   - `b_vmem_block`: 双缓冲 `2 * BK * BN * 2` bytes (bfloat16)
   如果总大小超过 VMEM 容量（通常约 16MB），编译器会报 OOM 错误。因此，寻找最优的 `(BM, BN, BK)` 组合是一个经典的性能调优问题。
