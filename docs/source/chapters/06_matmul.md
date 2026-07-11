# 第 6 章：矩阵乘法

## MXU 脉动阵列

TPU 的 MXU（Matrix Multiply Unit）是一个 128×128 的脉动阵列（Systolic Array）。它是 TPU 算力的主要来源。

脉动阵列的工作方式：
- 输入：LHS 矩阵的一行（128 个元素）从左侧流入，RHS 矩阵的一列（128 个元素）从上方流入
- 每个时钟周期，数据向右和向下流动，每个单元执行一次乘加操作
- 经过 128 个时钟周期后，一个 128×128 的输出块计算完成
- MXU 是**流水线化的**：可以连续输入多组数据，不需要等待前一组完成

MXU 支持的精度：
- bf16 × bf16 → fp32（最常用）
- fp32 × fp32 → fp32（通过多次 bf16 乘法模拟，慢 3-4 倍）
- int8 × int8 → int32
- fp8 × fp8 → fp32（v6e+）

## 精度控制

这是 TPU 与 GPU 最重要的区别之一。

**MXU 的乘法器不支持完整的 float32。** 即使传入 float32 操作数，硬件行为是：
1. 将 float32 截断为 bfloat16
2. 执行 bf16 × bf16 乘法
3. 将结果累加到 fp32 累加器

```python
# bf16 输入，fp32 累加（默认，最常用）
jnp.dot(a_bf16, b_bf16, preferred_element_type=jnp.float32)

# 如果需要完整 fp32 精度（慢 3-4 倍）：
jax.config.update("jax_default_matmul_precision", "float32")
```

bfloat16 与 float32 有相同的指数位（8 bits），因此动态范围一样大，只是尾数精度低（7 bits vs 23 bits）。对深度学习来说，动态范围比精度重要，所以 bf16 是理想格式。

## Tiling 约束

MXU 对输入矩阵的形状有严格要求：

```
C[M, N] = A[M, K] @ B[K, N]
```

- M（LHS 的行数）：必须是 8 的倍数（sublane 维度）
- K（收缩维度）：必须是 128 的倍数（lane 维度）
- N（RHS 的列数）：必须是 128 的倍数（lane 维度）

不满足时编译器会自动 padding，浪费计算。block size 应确保 BM 是 8 的倍数，BK 和 BN 是 128 的倍数。

## 分块矩阵乘法策略

对于 `C[M, N] = A[M, K] @ B[K, N]`，标准的两层策略：

1. **外层 `pallas_call`**：Grid 为 `(M//BM, N//BN)`，负责遍历输出块
2. **内层 `emit_pipeline`**：沿 K 维循环 `K//BK` 次，累加到 VMEM 中的 accumulator

为什么不把 K 维放在外层 Grid？因为那样每次 K 迭代都会触发累加器从 HBM 加载和写回（HBM Thrashing）。

## 完整实现

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BM, BK, BN = 128, 128, 128

def inner_kernel(a_vmem_ref, b_vmem_ref, acc_ref):
    """内层流水线 kernel：一次 BM×BK @ BK×BN"""
    acc_ref[...] += jnp.dot(
        a_vmem_ref[...], b_vmem_ref[...],
        preferred_element_type=jnp.float32
    )

def outer_kernel(a_hbm_ref, b_hbm_ref, c_hbm_ref, acc_ref):
    """外层 kernel：管理 K 维的流水线"""
    _, K = a_hbm_ref.shape
    num_k_blocks = K // BK

    acc_ref[...] = jnp.zeros_like(acc_ref[...])

    pltpu.emit_pipeline(
        inner_kernel,
        grid=(num_k_blocks,),
        in_specs=[
            pl.BlockSpec((BM, BK), lambda k: (0, k), memory_space=pltpu.VMEM),
            pl.BlockSpec((BK, BN), lambda k: (k, 0), memory_space=pltpu.VMEM),
        ],
        out_specs=[
            pl.BlockSpec((BM, BN), lambda k: (0, 0), memory_space=pltpu.VMEM),
        ],
    )(a_hbm_ref, b_hbm_ref, acc_ref)

    c_hbm_ref[...] = acc_ref[...].astype(c_hbm_ref.dtype)

def matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    M, K = a.shape
    _, N = b.shape

    return pl.pallas_call(
        outer_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), a.dtype),
        in_specs=[
            pl.BlockSpec((BM, K), lambda i, j: (i, 0)),
            pl.BlockSpec((K, BN), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((BM, BN), lambda i, j: (i, j)),
        grid=(M // BM, N // BN),
        scratch_shapes=[pltpu.VMEM((BM, BN), jnp.float32)],
    )(a, b)
```

## Block Size 选择

| 因素 | 偏好大 block | 偏好小 block |
| :--- | :--- | :--- |
| MXU 利用率 | BM≥128, BK≥128, BN≥128 | - |
| VMEM 容量 | - | 双缓冲需要 2×(BM×BK + BK×BN) + BM×BN |
| HBM 带宽 | 大块减少 DMA 次数 | - |
| 流水线效率 | 大块使计算时间 > DMA 时间 | - |
| 尾部浪费 | - | 小块减少 padding 浪费 |

VMEM 预算计算（bf16 输入，fp32 累加器）：
```
双缓冲 A: 2 × BM × BK × 2 bytes
双缓冲 B: 2 × BK × BN × 2 bytes
累加器:   BM × BN × 4 bytes
总计:     4*BM*BK + 4*BK*BN + 4*BM*BN bytes
```

对于 BM=BK=BN=256：4×256×256 + 4×256×256 + 4×256×256 = 768KB，远小于 16MB VMEM。

## 转置优化

`pltpu.CompilerParams(fuse_transposed_lhs_in_matmul=True)` 允许编译器将 LHS 的转置融合到 MXU 操作中：

```python
compiler_params = pltpu.CompilerParams(fuse_transposed_lhs_in_matmul=True)
```

这在 attention 计算中很有用——`Q @ K^T` 可以通过传入 K 并让编译器处理转置来避免显式的 transpose 操作。

## 与 GPU MatMul 的对比

| 维度 | GPU (CUTLASS/cuBLAS) | TPU (Pallas) |
| :--- | :--- | :--- |
| 计算单元 | Tensor Core (16×16×16) | MXU (128×128) |
| Tile 大小 | 通常 64-256 | 通常 128-512 |
| 共享内存/VMEM | ~160KB | 16MB+ |
| 流水线 | 多级寄存器流水线 | VMEM 双缓冲 |
| 调度 | 多 SM 并行 | 单核顺序 + 软件流水线 |
| 编程复杂度 | 极高（C++ 模板元编程）| 中等（Python）|

## GEMV 场景

当 batch size = 1 时（decode 阶段），矩阵乘法退化为 GEMV。MXU 利用率极低（只使用 128×128 阵列的一行）。

优化策略：
1. **Batching**：合并多个请求，恢复矩阵乘法形状
2. **量化**：int8/fp8 减少内存带宽需求（GEMV 是 memory-bound）
3. **分块归约**：权重沿 K 维分块，多核并行计算部分和

## 低级 MXU 原语

Pallas 还暴露了低级 MXU 控制原语，用于需要精细控制 MXU 流水线的场景：

```python
# 将 RHS 推入 MXU 的权重寄存器
pltpu.matmul_push_rhs(rhs_ref)

# 将 LHS 送入 MXU 并累加到 acc
pltpu.matmul_acc_lhs(lhs_ref, acc_ref)

# 从 MXU 弹出结果
result = pltpu.matmul_pop(acc_ref)
```

这些原语允许手动控制 MXU 的输入输出时序，在需要与其他操作精确交错时有用。但通常 `jnp.dot` 已经足够。
