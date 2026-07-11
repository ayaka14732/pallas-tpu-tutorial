# 第 8 章：实战 RMSNorm 算子

RMSNorm（Root Mean Square Layer Normalization）是现代大语言模型（如 LLaMA、Qwen）中最常用的归一化方法。它比 LayerNorm 更简单、更高效，因为它省去了均值（Mean）的计算。

本章我们将从数学公式出发，手写一个 TPU 上的 RMSNorm Pallas Kernel。

## 数学公式

给定输入向量 $x \in \mathbb{R}^d$，RMSNorm 的计算为：

$$ \text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma $$

其中 $\gamma \in \mathbb{R}^d$ 是可学习的缩放参数，$\epsilon$ 是一个小常数（防止除零）。

## 性能特征分析

RMSNorm 是一个典型的**内存密集型（Memory-bound）**算子。

对于形状为 `(batch, seq_len, hidden_dim)` 的输入：
- 计算量：约 $3 \times \text{batch} \times \text{seq\_len} \times \text{hidden\_dim}$ 次浮点运算（平方、求和、乘法）。
- 内存访问量：需要读取整个输入，写出整个输出。

由于算术强度很低，优化的关键在于**减少 HBM 访问次数**。

## Kernel 设计

我们的策略是：对于每一行（即 `hidden_dim` 维度），将整行数据一次性加载到 VMEM 中，在 VMEM 中完成所有计算（平方、归约求和、开方、归一化），然后一次性写回。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def rmsnorm_kernel(x_ref, weight_ref, out_ref, *, eps: float):
    # x_ref 形状: (hidden_dim,) — 我们每次处理一行
    # weight_ref 形状: (hidden_dim,)
    # out_ref 形状: (hidden_dim,)
    
    x = x_ref[...].astype(jnp.float32)
    w = weight_ref[...].astype(jnp.float32)
    
    # 计算 RMS
    # 注意：TPU 上的归约操作在最后一个维度上最慢
    # 但这里我们处理的是 1D 数组，所以只有一个维度
    rms = jnp.sqrt(jnp.mean(x * x) + eps)
    
    # 归一化并缩放
    out = (x / rms) * w
    
    out_ref[...] = out.astype(out_ref.dtype)


def pallas_rmsnorm(x: jax.Array, weight: jax.Array, eps: float = 1e-6):
    """
    x: (batch * seq_len, hidden_dim)
    weight: (hidden_dim,)
    """
    num_rows, hidden_dim = x.shape
    
    # 每次处理一行
    x_spec = pl.BlockSpec(
        block_shape=(None, hidden_dim),  # None 表示 squeeze 第一个维度
        index_map=lambda i: (i, 0)
    )
    
    # weight 对所有行都一样（广播）
    weight_spec = pl.BlockSpec(
        block_shape=(hidden_dim,),
        index_map=lambda i: (0,)
    )
    
    out_spec = pl.BlockSpec(
        block_shape=(None, hidden_dim),
        index_map=lambda i: (i, 0)
    )
    
    import functools
    return pl.pallas_call(
        functools.partial(rmsnorm_kernel, eps=eps),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[x_spec, weight_spec],
        out_specs=out_spec,
        grid=(num_rows,)
    )(x, weight)
```

## TPU 上的归约操作性能

在第 1 章中我们提到，TPU 上的归约操作性能与维度有关：
- 归约**前导维度（Leading dimensions）**：最快，且免费。
- 归约**倒数第二个维度（Second-to-last）**：较慢。
- 归约**最后一个维度（Last dimension）**：最慢。

在 RMSNorm 中，我们对 `hidden_dim`（最后一个维度）进行归约。这是不可避免的。但由于我们使用 `None` 将 batch 维度 squeeze 掉了，传入 Kernel 的 `x_ref` 是 1D 的 `(hidden_dim,)`，因此归约发生在唯一的维度上。

## 优化方向

1. **多行批处理**：如果 `hidden_dim` 不大，可以一次处理多行（如 `block_shape=(4, hidden_dim)`），提高 MXU 利用率。
2. **与后续算子融合**：RMSNorm 通常后接矩阵乘法（如 QKV 投影）。如果能将归一化结果直接留在 VMEM 中供后续矩阵乘法使用，可以省去一次 HBM 读写。
