# 第 9 章：实战 Softmax 算子

Softmax 是 Transformer 注意力机制的核心组件。在 TPU 上高效实现 Softmax，不仅需要考虑与 RMSNorm 类似的内存带宽瓶颈和归约性能，还需要特别处理数值稳定性，并掌握**在线 Softmax（Online Softmax）**算法，这是实现 FlashAttention 的前置条件。

## 数学公式与数值稳定性

给定输入向量 $x \in \mathbb{R}^d$，标准的 Softmax 计算为：

$$ \text{Softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{d} e^{x_j}} $$

**问题：** 指数函数 $e^x$ 增长极快。如果 $x_i$ 稍大（例如 50），$e^{50}$ 在 `float32` 中就会溢出（变成 `inf`），导致结果变成 `NaN`。

**解决方案：** 利用 Softmax 的平移不变性 $\text{Softmax}(x) = \text{Softmax}(x - c)$。我们通常取 $c = \max(x)$。这样，指数项的最大值变成了 $e^0 = 1$，彻底消除了溢出风险。

$$ \text{Softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^{d} e^{x_j - \max(x)}} $$

## 两遍 Softmax (Two-pass Softmax)

如果序列长度 $d$ 不大，整行数据能一次性放入 VMEM，我们可以直接实现上述公式，这需要遍历数据两遍：
1. 第一遍：计算局部最大值 $m = \max(x)$
2. 第二遍：计算 $e^{x_i - m}$ 和 $\sum e^{x_i - m}$，然后归一化

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def softmax_kernel(x_ref, out_ref):
    x = x_ref[...].astype(jnp.float32)
    
    # 第一遍：找最大值
    m = jnp.max(x, axis=-1, keepdims=True)
    
    # 第二遍：计算指数和归一化
    exp_x = jnp.exp(x - m)
    sum_exp = jnp.sum(exp_x, axis=-1, keepdims=True)
    
    out_ref[...] = (exp_x / sum_exp).astype(out_ref.dtype)
```

### TPU 上的 `exp` 操作代价

在第 1 章的硬件特性中提到，TPU 上的 `jnp.exp` 是由特殊的超越函数单元（Transcendental Unit）处理的，其吞吐量远低于普通的乘加单元。虽然不是极其昂贵，但在极长的序列中，密集的 `exp` 运算会成为显著的计算瓶颈。

## 在线 Softmax (Online Softmax)：FlashAttention 的基石

当序列长度过长（例如 32K、128K），无法将整行数据一次性放入 VMEM 时，两遍算法就失效了。因为我们无法在不知道全局最大值的情况下，计算正确的指数和。如果在计算出全局最大值之前把数据写回 HBM，再读出来算第二遍，HBM 带宽将被彻底打爆。

**在线 Softmax** 算法（Milakov et al., 2018）完美解决了这个问题。它允许我们在只遍历数据**一遍**的过程中，逐步更新正确的 Softmax 结果。这正是 FlashAttention 的核心思想。

在线 Softmax 维护两个状态变量：
- $m$：当前已见数据的最大值（Running max）
- $l$：当前已见数据的指数和（Running sum）

假设我们已经处理了块 $A$，其状态为 $m_A, l_A$。现在新来了一个块 $B$，其局部最大值为 $m_B$。
我们要合并它们的状态得到 $m_{new}, l_{new}$。

**更新规则：**
$$ m_{\text{new}} = \max(m_A, m_B) $$
$$ l_{\text{new}} = l_A \cdot e^{m_A - m_{\text{new}}} + \sum_{x \in B} e^{x - m_{\text{new}}} $$

关键在于缩放因子 $e^{m_A - m_{\text{new}}}$。如果新块 $B$ 包含了更大的值（$m_B > m_A$），我们需要将之前累加的指数和 $l_A$ "惩罚"（缩小）一个对应的比例，使其与新的全局最大值对齐。

### Pallas 中的在线 Softmax 实现

在 Pallas 中，我们可以利用 Scratch Buffer 在流水线的不同迭代间传递 $m$ 和 $l$ 的状态。

```python
def online_softmax_kernel(x_ref, out_ref, m_ref, l_ref):
    """
    x_ref: 当前加载到 VMEM 的数据块
    m_ref, l_ref: VMEM Scratch Buffers，跨迭代保持状态
    """
    x = x_ref[...].astype(jnp.float32)
    
    # 当前块的局部最大值
    m_curr = jnp.max(x, axis=-1, keepdims=True)
    
    # 获取上一步的全局最大值
    m_prev = m_ref[...]
    
    # 更新全局最大值
    m_new = jnp.maximum(m_prev, m_curr)
    
    # 计算重新缩放因子
    # 注意：如果 m_curr <= m_prev，那么 m_new == m_prev，alpha == 1.0，旧的 l_prev 不变
    # 如果 m_curr > m_prev，那么 m_new == m_curr，alpha < 1.0，旧的 l_prev 被缩小
    alpha = jnp.exp(m_prev - m_new)
    
    # 获取上一步的指数和
    l_prev = l_ref[...]
    
    # 更新全局指数和
    # 新块的指数也必须相对于 m_new 计算，以保证数值稳定
    l_new = l_prev * alpha + jnp.sum(jnp.exp(x - m_new), axis=-1, keepdims=True)
    
    # 保存状态供下一次迭代使用
    m_ref[...] = m_new
    l_ref[...] = l_new
    
    # 注意：真正的在线 Softmax 通常不在这里直接写出 out_ref。
    # 因为此时的 out 只是基于部分数据的 Softmax，并不是最终结果。
    # 在 FlashAttention 中，我们会同时维护一个与 Value 矩阵相乘的累加器。
```

## 与 FlashAttention 的联系

在线 Softmax 的精妙之处在于：**我们不仅可以重新缩放分母（指数和 $l$），还可以重新缩放分子（加权和累加器）**。

在 FlashAttention 中，我们不需要输出 Softmax 的概率矩阵，我们需要的是概率矩阵与 Value 矩阵的乘积：$\text{Softmax}(QK^T) V$。

如果我们维护一个累加器 $O = \sum e^{x_i - m} V_i$，当最大值 $m$ 更新时，我们只需要将旧的累加器也乘以缩放因子 $e^{m_{\text{old}} - m_{\text{new}}}$：

$$ O_{\text{new}} = O_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + \sum e^{x_{\text{new}} - m_{\text{new}}} V_{\text{new}} $$

最终结果就是 $O_{\text{final}} / l_{\text{final}}$。
这就是 FlashAttention 能够在只使用 $O(N)$ 内存的情况下，精确计算 $O(N^2)$ 注意力矩阵的数学基础。我们将在第 11 章详细实现它。
