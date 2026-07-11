# 第 8 章：RMSNorm

## 算子分析

RMSNorm 是 LLaMA、Qwen 等模型中替代 LayerNorm 的归一化操作：

```
RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight
```

**算术强度分析**（假设 bf16 输入，hidden_dim = d）：
- 读取：x (2d bytes) + weight (2d bytes) = 4d bytes
- 写入：output (2d bytes)
- 计算：d 次平方 + d 次求和 + 1 次 rsqrt + d 次乘法 + d 次乘法 ≈ 3d FLOPs
- AI = 3d / 6d = 0.5 FLOPs/byte

**结论**：RMSNorm 是极端的 **memory-bound** 操作。优化目标是最大化 HBM 带宽利用率，减少 HBM 访问次数。

## TPU 上的归约性能

TPU 的归约操作有方向性差异：

- **沿最后一维（lane 维）归约**：需要跨 lane 通信，代价最高
- **沿倒数第二维（sublane 维）归约**：需要 XLU 参与，中等代价
- **沿前导维度归约**：最快，直接对多个 8×128 寄存器块并行累加

对于 RMSNorm，归约维度是 hidden_dim（最后一维）。这是算法决定的，不可避免。但由于 hidden_dim 通常很大（4096-8192），归约本身不是瓶颈——HBM 带宽才是。

## 实现

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import functools

def rmsnorm_kernel(x_ref, weight_ref, out_ref, *, eps: float):
    # x_ref: (hidden_dim,) — 每次处理一行
    x = x_ref[...].astype(jnp.float32)
    w = weight_ref[...].astype(jnp.float32)

    # 计算 RMS
    mean_sq = jnp.mean(x * x)
    # 使用硬件近似倒数，比 1/sqrt(x) 快
    inv_rms = pl.reciprocal(jnp.sqrt(mean_sq + eps), approx=True)

    out_ref[...] = (x * inv_rms * w).astype(out_ref.dtype)

def rmsnorm(x: jax.Array, weight: jax.Array, eps: float = 1e-6):
    num_rows, hidden_dim = x.shape

    return pl.pallas_call(
        functools.partial(rmsnorm_kernel, eps=eps),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[
            pl.BlockSpec((None, hidden_dim), lambda i: (i, 0)),  # squeeze batch dim
            pl.BlockSpec((hidden_dim,), lambda i: (0,)),          # weight 广播
        ],
        out_specs=pl.BlockSpec((None, hidden_dim), lambda i: (i, 0)),
        grid=(num_rows,),
    )(x, weight)
```

## pl.reciprocal

`pl.reciprocal(x, approx=True)` 使用 TPU 的硬件近似倒数指令，比 `1.0 / x` 快得多。精度约 12 bits，对 bf16 足够。

```python
# 精确除法（VPU 多周期）
result = x / rms

# 近似倒数（单周期硬件指令）
result = x * pl.reciprocal(rms, approx=True)
```

RPA v3 kernel 中所有除法都使用 `pl.reciprocal(approx=True)`。

## 与 GPU 实现的对比

| 维度 | GPU (CUDA) | TPU (Pallas) |
| :--- | :--- | :--- |
| 归约实现 | Warp shuffle + block reduce | 向量寄存器内归约（硬件自动）|
| 同步 | `__syncthreads()` | 不需要（单线程顺序执行）|
| 内存访问 | 需要 coalesced access | 不需要（DMA 自动处理）|
| Block size | 受限于共享内存 | 受限于 VMEM（可以很大）|
| 向量化 | 需要手动 `float4` 加载 | 自动（8×128 寄存器）|

TPU 的优势：不需要手动管理线程同步和内存合并访问。

## 优化：多行批处理

如果 hidden_dim 较小（如 4096），单行处理时 DMA 启动开销占比大。可以一次处理多行：

```python
ROWS_PER_BLOCK = 4

def rmsnorm_batched_kernel(x_ref, weight_ref, out_ref, *, eps: float):
    # x_ref: (ROWS_PER_BLOCK, hidden_dim)
    x = x_ref[...].astype(jnp.float32)
    w = weight_ref[...]

    mean_sq = jnp.mean(x * x, axis=-1, keepdims=True)  # (4, 1)
    inv_rms = pl.reciprocal(jnp.sqrt(mean_sq + eps), approx=True)
    out_ref[...] = (x * inv_rms * w[None, :]).astype(out_ref.dtype)
```

注意：`keepdims=True` 产生 `(4, 1)` 形状。最后一维为 1 是 singleton dimension，会被 padding 到 128。如果这导致性能问题，可以手动 squeeze 后再 broadcast。

## 算子融合

RMSNorm 通常与后续的线性层（MatMul）相邻。将 RMSNorm 融合到 MatMul 的输入预处理中，避免一次 HBM 写入 + 读取：

```
未融合：x → [RMSNorm] → y (写回 HBM) → [MatMul] → z
融合后：x → [RMSNorm + MatMul] → z  (y 留在 VMEM)
```

在 emit_pipeline 框架中，将 RMSNorm 作为 MatMul 流水线的 prologue 自然实现。
