# 第 8 章：实战 RMSNorm 算子

RMSNorm（Root Mean Square Layer Normalization）是现代大语言模型（如 LLaMA、Qwen）中最常用的归一化方法。它比传统的 LayerNorm 更简单、更高效，因为它省去了均值（Mean）的计算，只计算均方根（RMS）。

本章我们将从数学公式出发，手写一个 TPU 上的 RMSNorm Pallas Kernel，并深入探讨 TPU 上归约操作（Reduction）的性能特征。

## 数学公式

给定输入向量 $x \in \mathbb{R}^d$，RMSNorm 的计算为：

$$ \text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma $$

其中 $\gamma \in \mathbb{R}^d$ 是可学习的缩放参数，$\epsilon$ 是一个小常数（防止除零）。

## 性能特征分析：极端的 Memory-Bound

根据上一章的 Roofline 模型，我们来计算 RMSNorm 的算术强度（AI）：

对于形状为 `(batch, seq_len, hidden_dim)` 的输入，假设我们处理一个大小为 $d$ (`hidden_dim`) 的向量：
- **计算量**：约 $3d$ 次浮点运算（$d$ 次平方、$d$ 次求和、$d$ 次乘法）。
- **内存访问量**：需要读取 $x$（$2d$ Bytes，假设 bfloat16），读取 $\gamma$（$2d$ Bytes），写出输出（$2d$ Bytes）。总计 $6d$ Bytes。

算术强度 $\text{AI} = \frac{3d}{6d} = 0.5$ FLOPs/Byte。
这远远低于 TPU 的机器平衡点（约 229 FLOPs/Byte）。因此，RMSNorm 是一个极其典型的**内存密集型（Memory-bound）**算子。

**优化核心：** 既然算力大量过剩，我们优化的唯一目标就是**减少 HBM 访问次数，并最大化 HBM 带宽利用率**。

## Kernel 设计策略

为了最小化 HBM 访问，我们的策略是：对于每一行（即 `hidden_dim` 维度），将整行数据一次性加载到 VMEM 中，在 VMEM 中完成所有计算（平方、归约求和、开方、归一化），然后一次性写回 HBM。绝不允许在计算过程中将中间结果（如 $x^2$）写回 HBM。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def rmsnorm_kernel(x_ref, weight_ref, out_ref, *, eps: float):
    # x_ref 形状: (hidden_dim,) — 我们每次处理一行
    # weight_ref 形状: (hidden_dim,)
    # out_ref 形状: (hidden_dim,)
    
    # 强制转换为 float32 进行高精度累加
    x = x_ref[...].astype(jnp.float32)
    w = weight_ref[...].astype(jnp.float32)
    
    # 计算 RMS
    # 注意：TPU 上的归约操作在最后一个维度上最慢
    # 但这里我们处理的是 1D 数组，所以只有一个维度
    rms = jnp.sqrt(jnp.mean(x * x) + eps)
    
    # 归一化并缩放
    out = (x / rms) * w
    
    # 转换回原类型并写回 VMEM (随后由 DMA 写回 HBM)
    out_ref[...] = out.astype(out_ref.dtype)


def pallas_rmsnorm(x: jax.Array, weight: jax.Array, eps: float = 1e-6):
    """
    x: (batch * seq_len, hidden_dim)
    weight: (hidden_dim,)
    """
    num_rows, hidden_dim = x.shape
    
    # 每次提取一行。使用 None 挤压掉前面的维度，使其在 Kernel 中表现为 1D 数组
    x_spec = pl.BlockSpec(
        block_shape=(None, hidden_dim),
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

## TPU 上的归约操作性能陷阱

在代码中，我们使用了 `jnp.mean(x * x)`。这是一个归约（Reduction）操作。在 TPU 上，归约操作的性能与被归约的维度（Axis）有着极大的关系。

我们在第 1 章中提到，TPU 向量寄存器的大小通常是 8x128。数组的最后两个维度会被映射到这个物理结构上。
- 归约**前导维度（Leading dimensions，除了最后两维之外的维度）**：最快，几乎是免费的。因为硬件可以简单地对多个完整的 8x128 寄存器块进行并行累加。
- 归约**倒数第二个维度（Sublane dimension）**：较慢，需要跨 Sublane 进行数据洗牌（Shuffle）。
- 归约**最后一个维度（Lane dimension）**：**最慢**。需要昂贵的跨 Lane 通信指令。

在 RMSNorm 中，我们恰好是对 `hidden_dim`（也就是数组的最后一个维度）进行归约。这是算法决定的，不可避免。

**如何缓解？**
由于我们使用 `None` 将 batch 维度 squeeze 掉了，传入 Kernel 的 `x_ref` 是 1D 的 `(hidden_dim,)`。这已经比传入 `(1, hidden_dim)` 要好，因为它避免了硬件在处理单元素维度时的 Padding 惩罚。

## 进阶优化：算子融合 (Operator Fusion)

由于 RMSNorm 受到严重的 HBM 带宽限制，单靠优化 Kernel 内部的计算是无法带来质变的。终极的优化手段是**算子融合**。

在 LLM 的 Transformer 块中，RMSNorm 通常直接连接在注意力机制或 MLP 层的输出之后，或者连接在 QKV 投影矩阵乘法之前。

如果我们在计算完 RMSNorm 后，不将结果写回 HBM，而是**直接将其留在 VMEM 中**，紧接着调用 QKV 的矩阵乘法（利用 MXU），我们就可以完全省去一次 HBM 写入和一次 HBM 读取。

这种将 Memory-bound 算子（RMSNorm）与 Compute-bound 算子（MatMul）融合的技术，是现代大模型推理框架（如 vLLM, TGI）在 TPU/GPU 上压榨极致性能的核心秘诀。在 Pallas 中，这可以通过在同一个 `pallas_call` 的 Kernel 内部依次调用两者的逻辑来实现。
