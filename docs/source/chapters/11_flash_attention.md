# 第 11 章：FlashAttention 在 TPU 上的实现

FlashAttention 是现代大模型推理和训练中最重要的算子之一。它通过分块计算和在线 Softmax，将注意力机制的内存复杂度从 $O(N^2)$ 降低到 $O(N)$，同时保持了精确计算（非近似）。

本章我们将分析 JAX 官方代码库中的 TPU FlashAttention Kernel 的设计思路。

## 标准注意力的问题

标准的注意力计算：
$$ \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V $$

如果 $Q, K, V \in \mathbb{R}^{N \times d}$，那么 $QK^T \in \mathbb{R}^{N \times N}$。当 $N$ 很大时（如 8192 或更长），这个 $N \times N$ 的矩阵无法放入 VMEM（16MB），必须存储在 HBM 中，导致大量的 HBM 读写。

## FlashAttention 的核心思想

FlashAttention 的关键洞察是：我们不需要一次性计算整个 $QK^T$ 矩阵。我们可以将 $K$ 和 $V$ 分成小块，逐块计算部分注意力分数，并使用**在线 Softmax**（第 9 章）来维护正确的归一化因子。

### 分块策略

将 $Q$ 分为 $Q_1, Q_2, \ldots$ 块（沿 seq_len 维度），$K$ 和 $V$ 分为 $K_1, K_2, \ldots$ 和 $V_1, V_2, \ldots$ 块。

对于每个 $Q_i$ 块，我们遍历所有 $K_j, V_j$ 块：
1. 计算 $S_{ij} = Q_i K_j^T / \sqrt{d}$
2. 更新 running max $m$ 和 running sum $l$
3. 重新缩放之前的累加结果，并加上新的贡献

## TPU 上的 FlashAttention 设计

在 TPU 上实现 FlashAttention 与 GPU 版本有显著不同：

| 方面 | GPU (CUDA) | TPU (Pallas) |
| :--- | :--- | :--- |
| 并行模型 | 成千上万线程并行处理不同的 Q 块 | 顺序执行，通过流水线隐藏延迟 |
| 内存管理 | 共享内存 (SRAM) 手动管理 | VMEM 自动管理 + Scratch Buffers |
| 分块大小 | 受限于共享内存大小 (通常 48KB-192KB) | VMEM 很大 (16MB+)，可以用更大的块 |
| 同步 | __syncthreads() | DMA 信号量 |

### 关键代码结构

```python
# JAX 官方 FlashAttention 的简化结构
def flash_attention_kernel(
    q_ref,    # [block_q, head_dim] in VMEM
    k_ref,    # [block_k, head_dim] in VMEM (通过流水线预取)
    v_ref,    # [block_k, head_dim] in VMEM (通过流水线预取)
    o_ref,    # [block_q, head_dim] output
    m_ref,    # [block_q, 128] scratch - running max
    l_ref,    # [block_q, 128] scratch - running sum
):
    # 计算 QK^T (利用 MXU)
    qk = jnp.dot(q_ref[...], k_ref[...].T, preferred_element_type=jnp.float32)
    qk *= sm_scale
    
    # 应用因果掩码 (Causal Mask)
    # 使用 broadcasted_iota 生成位置索引，避免显式构造掩码矩阵
    row_ids = lax.broadcasted_iota(jnp.int32, qk.shape, 0)
    col_ids = lax.broadcasted_iota(jnp.int32, qk.shape, 1)
    causal_mask = row_ids < col_ids
    qk = jnp.where(causal_mask, mask_value, qk)
    
    # 在线 Softmax 更新
    m_curr = jnp.max(qk, axis=-1, keepdims=True)
    m_prev = m_ref[...]
    m_new = jnp.maximum(m_prev, m_curr)
    
    alpha = jnp.exp(m_prev - m_new)
    beta = jnp.exp(m_curr - m_new)
    
    l_prev = l_ref[...]
    s_curr = jnp.exp(qk - m_new)
    l_new = l_prev * alpha + jnp.sum(s_curr, axis=-1, keepdims=True)
    
    # 更新累加器
    # 之前的结果需要乘以 alpha (重新缩放)
    o_prev = o_ref[...].astype(jnp.float32)
    o_new = (o_prev * alpha + jnp.dot(s_curr, v_ref[...], preferred_element_type=jnp.float32)) / l_new
    
    # 保存状态
    m_ref[...] = m_new
    l_ref[...] = l_new
    o_ref[...] = o_new.astype(o_ref.dtype)
```

## BlockSizes 的选择

JAX 官方的 FlashAttention 通过 `BlockSizes` 数据类来参数化所有的分块大小：

```python
@dataclasses.dataclass(frozen=True)
class BlockSizes:
    block_q: int        # Q 的分块大小
    block_k_major: int  # K 的外层分块 (用于流水线)
    block_k: int        # K 的内层分块 (用于 MXU 计算)
    block_b: int        # Batch 的分块大小
```

选择合适的 `BlockSizes` 是性能调优的关键。一般原则是：
- `block_q` 和 `block_k` 应该尽可能大（充分利用 MXU），但不能超过 VMEM 容量。
- `block_k_major` 应该是 `block_k` 的整数倍，用于控制流水线的粒度。

## 反向传播 (Backward Pass)

FlashAttention 的反向传播比前向传播更加复杂。由于前向传播中我们没有保存完整的 $QK^T$ 矩阵（这正是 FlashAttention 省内存的原因），反向传播时需要**重新计算**这些中间结果。

这就是为什么 JAX 官方代码中有 `block_q_major_dkv`、`block_k_major_dq` 等额外的分块参数——它们控制反向传播中重计算的粒度。

## 从 FlashAttention 到 Ragged Paged Attention

FlashAttention 假设所有序列长度相同，且 KV Cache 是连续存储的。在实际的 LLM 推理中，这两个假设都不成立。Ragged Paged Attention（第 13 章）正是在 FlashAttention 的基础上，加入了动态长度处理和分页内存管理。
