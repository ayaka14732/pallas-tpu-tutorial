# 第 11 章：FlashAttention 在 TPU 上的实现

FlashAttention (Dao et al., 2022) 是现代大模型推理和训练中最重要的算子之一。它通过分块计算（Tiling）和在线 Softmax，将注意力机制的内存复杂度从 $O(N^2)$ 降低到 $O(N)$，同时保持了精确计算（非近似）。

本章我们将分析 JAX 官方代码库中的 TPU FlashAttention Kernel 的设计思路，并对比它与 CUDA 版本的实现差异。

## 标准注意力的问题与内存墙

标准的注意力计算公式：
$$ \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V $$

如果序列长度 $N = 8192$，注意力头维度 $d = 128$，批量大小为 1。
计算 $QK^T$ 会产生一个 $8192 \times 8192$ 的浮点矩阵。在 `bfloat16` 下，这需要约 134MB 的内存。
这个矩阵远远超出了 TPU 的 VMEM 容量（通常 16MB 左右），更不用说 GPU 的 Shared Memory（通常 160KB）。

因此，标准实现必须将这个庞大的中间矩阵写回 HBM，然后再读出来计算 Softmax，最后再读出来与 $V$ 相乘。这导致了灾难性的 HBM 读写开销，注意力计算完全被**内存带宽（Memory-bound）**限制。

## FlashAttention 的核心思想

FlashAttention 的关键洞察是：我们不需要一次性计算整个 $QK^T$ 矩阵。我们可以将 $K$ 和 $V$ 分成小块，逐块计算部分注意力分数，并使用**在线 Softmax**（我们在第 9 章学过）来维护正确的归一化因子。

### 分块策略 (Tiling)

将 $Q$ 分为 $Q_1, Q_2, \ldots$ 块（沿 seq_len 维度），$K$ 和 $V$ 分为 $K_1, K_2, \ldots$ 和 $V_1, V_2, \ldots$ 块。

对于每个 $Q_i$ 块，我们遍历所有 $K_j, V_j$ 块：
1. 计算局部注意力分数 $S_{ij} = Q_i K_j^T / \sqrt{d}$
2. 更新 running max $m$ 和 running sum $l$
3. 重新缩放之前的累加结果，并加上新的贡献 $\text{Softmax}(S_{ij}) V_j$

这样，所有的中间计算都保留在超快的片上内存中，HBM 的读写量被大幅削减。

## TPU 上的 FlashAttention 设计：与 GPU 的对比

在 TPU 上实现 FlashAttention 与 GPU 版本有显著不同，主要体现在并行模型和内存管理上。

| 方面 | GPU (CUDA FlashAttention) | TPU (Pallas FlashAttention) |
| :--- | :--- | :--- |
| **并行模型** | 成千上万个线程块 (Thread Blocks) 并发处理不同的 Q 块和注意力头。 | 网格 (Grid) 通常顺序执行。利用 `emit_pipeline` 和 MXU 隐藏延迟。 |
| **内存管理** | 手动管理 Shared Memory (SRAM)。程序员需要精确计算指针偏移。 | 自动分配 VMEM 和 Scratch Buffers。 |
| **分块大小** | 受限于 Shared Memory 容量 (通常 `block_q=128`, `block_k=128`)。 | VMEM 容量极大 (16MB+)，可以使用非常大的块 (例如 `block_q=1024`, `block_k=1024`)。 |
| **同步机制** | 使用 `__syncthreads()` 同步线程块内的线程。 | 使用底层 DMA 信号量 (Semaphores) 同步流水线阶段。 |
| **计算引擎** | Tensor Cores (Warp 级协作)。 | 巨大的 MXU 脉动阵列 (自动映射)。 |

### 关键代码结构 (Pallas 伪代码)

在 Pallas 中，我们通常使用 `emit_pipeline` 沿着 $K, V$ 序列长度维度进行流水线预取。

```python
# JAX 官方 FlashAttention 的简化结构
def flash_attention_kernel(
    q_ref,    # [block_q, head_dim] in VMEM (当前 Q 块)
    k_ref,    # [block_k, head_dim] in VMEM (流水线预取的 K 块)
    v_ref,    # [block_k, head_dim] in VMEM (流水线预取的 V 块)
    o_ref,    # [block_q, head_dim] output
    m_ref,    # [block_q] scratch - running max
    l_ref,    # [block_q] scratch - running sum
):
    # 1. 计算 QK^T (利用 MXU，强制 float32 累加)
    qk = jnp.dot(q_ref[...], k_ref[...].T, preferred_element_type=jnp.float32)
    qk *= sm_scale
    
    # 2. 应用因果掩码 (Causal Mask)
    # 在 TPU 上，避免显式构造巨大的掩码矩阵，使用 broadcasted_iota 即时生成
    row_ids = lax.broadcasted_iota(jnp.int32, qk.shape, 0)
    col_ids = lax.broadcasted_iota(jnp.int32, qk.shape, 1)
    causal_mask = row_ids < col_ids
    qk = jnp.where(causal_mask, mask_value, qk)
    
    # 3. 在线 Softmax 更新
    m_curr = jnp.max(qk, axis=-1, keepdims=True)
    m_prev = m_ref[...]
    m_new = jnp.maximum(m_prev, m_curr)
    
    alpha = jnp.exp(m_prev - m_new)
    beta = jnp.exp(m_curr - m_new)
    
    l_prev = l_ref[...]
    s_curr = jnp.exp(qk - m_new)
    l_new = l_prev * alpha + jnp.sum(s_curr, axis=-1, keepdims=True)
    
    # 4. 更新累加器 (利用 MXU)
    # 之前的结果需要乘以 alpha (重新缩放)
    o_prev = o_ref[...].astype(jnp.float32)
    o_new = (o_prev * alpha + jnp.dot(s_curr, v_ref[...], preferred_element_type=jnp.float32)) / l_new
    
    # 5. 保存状态供下一个 K 块迭代使用
    m_ref[...] = m_new
    l_ref[...] = l_new
    o_ref[...] = o_new.astype(o_ref.dtype)
```

## BlockSizes 的选择与性能调优

JAX 官方的 FlashAttention 通过 `BlockSizes` 数据类来参数化所有的分块大小。由于 TPU 的 VMEM 非常大，调优策略与 GPU 有很大不同。

```python
@dataclasses.dataclass(frozen=True)
class BlockSizes:
    block_q: int        # Q 的分块大小
    block_k_major: int  # K 的外层分块 (用于流水线预取)
    block_k: int        # K 的内层分块 (用于 MXU 计算)
    block_b: int        # Batch 的分块大小
```

**调优原则：**
1. **最大化 MXU 利用率**：`block_q` 和 `block_k` 应该尽可能大，且必须是 128 的倍数。
2. **流水线粒度**：`block_k_major` 应该是 `block_k` 的整数倍。它决定了 DMA 预取的数据块大小。如果太大，可能会导致 OOM；如果太小，流水线气泡会变多。
3. **VMEM 限制**：与 GPU 相比，你可以把 `block_q` 设到 512 甚至 1024。但要注意，在线 Softmax 的状态变量（`m_ref`, `l_ref`）和累加器（`o_ref`）的体积也会随之增大。

## 反向传播 (Backward Pass) 的复杂性

FlashAttention 的反向传播比前向传播复杂得多。

在标准注意力中，前向传播会保存巨大的 $QK^T$ 和 $\text{Softmax}(QK^T)$ 矩阵，反向传播直接使用它们。
但 FlashAttention 为了省内存，**没有**保存这些中间矩阵。因此，在反向传播时，它必须在片上内存中**重新计算**前向传播的 Softmax 结果，然后再计算梯度。

这就是为什么 JAX 官方代码中有 `block_q_major_dkv`、`block_k_major_dq` 等额外的分块参数——它们控制反向传播中重计算和梯度累加的粒度。在 TPU 上，反向传播的调优空间更大，因为我们需要在重计算的开销和 HBM 读写之间找到最佳平衡点。
