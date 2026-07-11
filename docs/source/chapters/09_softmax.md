# 第 9 章：实战 Softmax 算子

Softmax 是 Transformer 注意力机制的核心组件。在 TPU 上高效实现 Softmax 需要理解数值稳定性和归约操作的性能特征。

## 数学公式

给定输入向量 $x \in \mathbb{R}^d$，Softmax 的计算为：

$$ \text{Softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^{d} e^{x_j - \max(x)}} $$

减去 $\max(x)$ 是为了数值稳定性，防止指数运算溢出。

## 两遍 Softmax vs 在线 Softmax

### 两遍 Softmax（标准实现）

如果整行数据能一次性放入 VMEM，我们可以直接实现标准的两遍算法：
1. 第一遍：计算 $m = \max(x)$
2. 第二遍：计算 $e^{x_i - m}$ 和 $\sum e^{x_i - m}$，然后归一化

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def softmax_kernel(x_ref, out_ref):
    x = x_ref[...].astype(jnp.float32)
    
    # 数值稳定的 Softmax
    m = jnp.max(x, axis=-1, keepdims=True)
    exp_x = jnp.exp(x - m)
    sum_exp = jnp.sum(exp_x, axis=-1, keepdims=True)
    
    out_ref[...] = (exp_x / sum_exp).astype(out_ref.dtype)
```

### 在线 Softmax（Online Softmax）

当序列长度过长，无法将整行数据一次性放入 VMEM 时，我们需要使用**在线 Softmax** 算法。这正是 FlashAttention 的核心思想。

在线 Softmax 维护两个状态变量：
- $m$：当前已见数据的最大值（Running max）
- $l$：当前已见数据的指数和（Running sum）

每当新的数据块到来时，更新规则为：
$$ m_{\text{new}} = \max(m_{\text{old}}, \max(x_{\text{new}})) $$
$$ l_{\text{new}} = l_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + \sum e^{x_{\text{new}} - m_{\text{new}}} $$

```python
def online_softmax_kernel(x_ref, out_ref, m_ref, l_ref):
    """
    x_ref: 当前数据块
    m_ref, l_ref: VMEM Scratch，跨 Grid 迭代保持状态
    """
    x = x_ref[...].astype(jnp.float32)
    
    # 当前块的局部最大值
    m_curr = jnp.max(x, axis=-1, keepdims=True)
    
    # 更新全局最大值
    m_prev = m_ref[...]
    m_new = jnp.maximum(m_prev, m_curr)
    
    # 更新全局指数和
    l_prev = l_ref[...]
    alpha = jnp.exp(m_prev - m_new)
    beta = jnp.exp(m_curr - m_new)
    l_new = l_prev * alpha + jnp.sum(jnp.exp(x - m_new), axis=-1, keepdims=True)
    
    # 保存状态
    m_ref[...] = m_new
    l_ref[...] = l_new
```

## TPU 上的 exp 操作代价

在第 1 章的操作代价表中，`jnp.exp` 被标记为 🌕（中等代价）。这意味着在 TPU 上，指数运算不是免费的，但也不是极其昂贵的。

对于 Softmax 来说，主要的性能瓶颈通常不是 `exp` 本身，而是：
1. **归约操作**：`max` 和 `sum` 在最后一个维度上的归约是最慢的。
2. **内存带宽**：如果序列很长，需要多次从 HBM 加载数据。

## 完整实现

```python
def pallas_softmax(x: jax.Array):
    """
    x: (batch, seq_len) — 对最后一个维度做 Softmax
    """
    batch, seq_len = x.shape
    
    x_spec = pl.BlockSpec(
        block_shape=(None, seq_len),
        index_map=lambda i: (i, 0)
    )
    
    out_spec = pl.BlockSpec(
        block_shape=(None, seq_len),
        index_map=lambda i: (i, 0)
    )
    
    return pl.pallas_call(
        softmax_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[x_spec],
        out_specs=out_spec,
        grid=(batch,)
    )(x)
```

## 与 FlashAttention 的关系

在线 Softmax 是 FlashAttention 的核心子程序。在 FlashAttention 中，我们不仅需要计算 Softmax 的分母（`l`），还需要同时维护加权累加器（`acc`）。当 `m` 更新时，之前累加的结果需要被重新缩放：

$$ \text{acc}_{\text{new}} = \text{acc}_{\text{old}} \cdot \frac{l_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}}}{l_{\text{new}}} + \frac{e^{m_{\text{curr}} - m_{\text{new}}} \cdot (Q K_{\text{curr}}^T) V_{\text{curr}}}{l_{\text{new}}} $$

这正是我们在第 11 章 FlashAttention 和第 13 章 Ragged Paged Attention 中会深入讨论的内容。
