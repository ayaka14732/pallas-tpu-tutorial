# 第 6 章：实战矩阵乘法 (MatMul)

矩阵乘法是深度学习中最核心的算子。本章我们将利用前面学到的 Grid、BlockSpec、Scratch Buffers 和 `emit_pipeline`，手写一个高性能的 TPU 矩阵乘法 Kernel。

## 算法设计

给定矩阵 $A \in \mathbb{R}^{M \times K}$ 和 $B \in \mathbb{R}^{K \times N}$，计算 $C = A \times B$。

我们将矩阵划分为大小为 `(BM, BK)` 和 `(BK, BN)` 的块。
对于输出 $C$ 的每一个块 $C_{i, j}$，它的计算公式为：
$$ C_{i, j} = \sum_{k} A_{i, k} \times B_{k, j} $$

### 数据流设计
1. **外部 Grid**：大小为 `(M // BM, N // BN)`，遍历输出 $C$ 的块。
2. **Scratch Buffer**：在 VMEM 中分配大小为 `(BM, BN)` 的累加器。
3. **内部 Pipeline**：迭代 `K // BK` 次，每次取 $A_{i, k}$ 和 $B_{k, j}$ 到 VMEM，相乘并加到累加器。

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
    BK = 128  # 我们在内部流水线中每次处理的 K 维度大小
    num_k_blocks = K // BK
    
    # 1. 初始化累加器 (VMEM)
    acc_ref[...] = 0.0
    
    # 2. 定义流水线内部的计算逻辑
    def body_fn(k_idx, a_vmem_block, b_vmem_block):
        # a_vmem_block 形状: (BM, BK)
        # b_vmem_block 形状: (BK, BN)
        # jnp.dot 在 TPU 上会自动映射到 MXU (矩阵乘法单元)
        # 推荐指定 preferred_element_type=jnp.float32 以利用 MXU 的 f32 累加特性
        acc_ref[...] += jnp.dot(
            a_vmem_block[...], 
            b_vmem_block[...], 
            preferred_element_type=jnp.float32
        )
        
    # 3. 启动嵌套流水线
    pltpu.emit_pipeline(
        body_fn,
        num_iterations=num_k_blocks,
        inputs=[
            # 沿着 a_ref 的第 1 维 (K 维) 分块，块大小为 BK，双缓冲
            pl.Buffered(a_ref, buffer_count=2, block_shape=(BM, BK), index_map=lambda k: (0, k)),
            # 沿着 b_ref 的第 0 维 (K 维) 分块，块大小为 BK，双缓冲
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

## 精度控制 (Precision Control)

在 TPU 上，矩阵乘法（MXU）**总是产生 float32 格式的结果**。

更重要的是，即使你将 32-bit 的操作数（如 `float32`）传递给矩阵乘法，它们也会被**默认截断为 `bfloat16`** 进行乘法运算，然后将结果累加到 `float32` 中。

这是因为 `bfloat16` 在 TPU MXU 上能提供最高的吞吐量。如果你确实需要完整的 `float32` 精度（代价是速度变慢），你需要设置 `jax.default_matmul_precision("float32")`。对于绝大多数深度学习推理任务，默认的 `bfloat16` 截断是预期且高效的行为。

## 性能优化提示

1. **Tile 对齐**：确保 `BM`, `BN`, `BK` 是 128 的倍数，这样可以完美映射到 MXU 和 8x128 的向量寄存器。
2. **转置融合**：在某些情况下，你可以将输入的转置融合到矩阵乘法中，这通常是免费的。
3. **VMEM 限制**：`BM` 和 `BN` 不能太大。`acc_ref`、双缓冲的 `a_vmem_block` 和 `b_vmem_block` 都会占用 VMEM。如果总大小超过 VMEM 容量（通常约 16MB），编译器会报 OOM 错误。
