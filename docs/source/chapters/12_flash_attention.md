# 第 12 章：FlashAttention

## 动机

标准注意力计算 `O = softmax(Q @ K^T / sqrt(d)) @ V` 需要 O(N^2) 的中间存储（完整的 attention score 矩阵）。对于长序列（N=8192, d=128），这个矩阵在 bf16 下需要约 134MB，远超 VMEM 容量。

FlashAttention 的核心思想：利用在线 Softmax，将注意力计算分块进行，只需要 O(N) 的中间存储，且结果与标准注意力**精确一致**（非近似）。

## 算法

对于单个 query 块 Q_i（形状 BM×d），遍历所有 KV 块：

```
m = -inf          # running max (BM,)
l = 0             # running sum (BM,)
o = 0             # running output (BM, d)

for j in range(num_kv_blocks):
    # Step 1: 局部 attention score (MXU)
    s = Q_i @ K_j^T / sqrt(d)    # (BM, BK)

    # Step 2: 在线 softmax 更新 (VPU)
    m_new = max(m, rowmax(s))     # (BM,)
    correction = exp(m - m_new)   # (BM,)

    # Step 3: 修正之前的累加器 (VPU)
    o = o * correction[:, None]   # (BM, d)
    l = l * correction            # (BM,)

    # Step 4: 当前块贡献 (MXU)
    p = exp(s - m_new[:, None])   # (BM, BK)
    o = o + p @ V_j               # (BM, d)
    l = l + rowsum(p)             # (BM,)

    # Step 5: 更新 max
    m = m_new

# 最终归一化
o = o / l[:, None]
```

## MXU 与 VPU 的交错

每次迭代包含两类操作：
- **MXU**：`Q @ K^T`（Step 1）和 `p @ V`（Step 4）
- **VPU**：exp、max、sum、correction（Step 2-3）

TPU 编译器会尝试将 VPU 操作插入 MXU 流水线间隙。理想情况下，VPU 操作完全被 MXU 操作隐藏。

## 精度处理

- `Q @ K^T`：bf16 × bf16 → fp32（MXU 自动 fp32 累加）
- `p @ V`：p 是 fp32 的 exp 结果，但 MXU 输入必须是 bf16
- 解决：将 p 截断为 bf16 后送入 MXU。精度损失可接受（p ∈ [0,1]）
- `o` 累加器：必须是 fp32（长序列累加需要精度）
- 最终输出：转回 bf16

## Pallas 实现

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BM = 128  # query block size
BK = 128  # kv block size

def flash_attention_kernel(
    q_ref, k_ref, v_ref, o_ref,
    m_scratch, l_scratch, o_scratch
):
    d = q_ref.shape[-1]
    seq_len = k_ref.shape[0]
    num_kv_blocks = seq_len // BK
    scale = jnp.float32(1.0 / jnp.sqrt(jnp.float32(d)))

    # 初始化
    m_scratch[...] = jnp.full((BM,), -jnp.inf, dtype=jnp.float32)
    l_scratch[...] = jnp.zeros((BM,), dtype=jnp.float32)
    o_scratch[...] = jnp.zeros((BM, d), dtype=jnp.float32)

    @pl.loop(0, num_kv_blocks)
    def _(j):
        k_block = k_ref[j*BK:(j+1)*BK, :]  # (BK, d)
        v_block = v_ref[j*BK:(j+1)*BK, :]  # (BK, d)
        q = q_ref[...]                       # (BM, d)

        # Step 1: attention score (MXU)
        s = jnp.dot(q, k_block.T, preferred_element_type=jnp.float32) * scale

        # Step 2: online softmax
        m_old = m_scratch[...]
        m_new = jnp.maximum(m_old, jnp.max(s, axis=-1))
        correction = jnp.exp(m_old - m_new)

        # Step 3: 修正累加器
        o_scratch[...] = o_scratch[...] * correction[:, None]
        l_scratch[...] = l_scratch[...] * correction

        # Step 4: 当前块贡献 (MXU)
        p = jnp.exp(s - m_new[:, None])
        o_scratch[...] = o_scratch[...] + jnp.dot(
            p.astype(jnp.bfloat16), v_block.astype(jnp.bfloat16),
            preferred_element_type=jnp.float32
        )
        l_scratch[...] = l_scratch[...] + jnp.sum(p, axis=-1)

        # Step 5: 更新 max
        m_scratch[...] = m_new

    # 最终归一化
    inv_l = pl.reciprocal(l_scratch[...], approx=True)
    o_ref[...] = (o_scratch[...] * inv_l[:, None]).astype(o_ref.dtype)
```

## 流水线优化

生产级实现使用双缓冲重叠 KV 块的 DMA 和计算：

```python
# 伪代码：双缓冲版本
# prologue: 预取第一个 KV 块
fetch_kv(0, buf=0)

@pl.loop(0, num_kv_blocks)
def _(j):
    buf = j % 2
    wait_kv(buf)

    # 预取下一个 KV 块（与当前计算重叠）
    @pl.when(j + 1 < num_kv_blocks)
    def _():
        fetch_kv(j + 1, buf=1-buf)

    # 计算
    compute_attention_block(q, k_buf[buf], v_buf[buf])
```

## Causal Mask

Decoder 中需要 causal mask（下三角）。实现方式：

```python
# 使用 broadcasted_iota 即时生成 mask（不占 VMEM）
row_ids = jax.lax.broadcasted_iota(jnp.int32, (BM, BK), 0) + q_block_idx * BM
col_ids = jax.lax.broadcasted_iota(jnp.int32, (BM, BK), 1) + j * BK
causal_mask = row_ids >= col_ids
s = jnp.where(causal_mask, s, -jnp.inf)
```

优化：当 query 块的最小位置 > key 块的最大位置时，整个块都是 -inf，可以跳过。

## Block Size 选择

| 参数 | 典型值 | 约束 |
| :--- | :--- | :--- |
| BM (query block) | 128-512 | 8 的倍数，受 VMEM 限制 |
| BK (kv block) | 128-256 | 128 的倍数（MXU K 维）|
| d (head dim) | 128 | 128 的倍数 |

VMEM 预算：
```
q: BM × d × 2 bytes
k_buf (双缓冲): 2 × BK × d × 2 bytes
v_buf (双缓冲): 2 × BK × d × 2 bytes
o_scratch: BM × d × 4 bytes
m_scratch: BM × 4 bytes
l_scratch: BM × 4 bytes
```

对于 BM=128, BK=128, d=128：约 256KB，远小于 16MB VMEM。

## 与 GPU FlashAttention 的对比

| 维度 | GPU (Tri Dao) | TPU (Pallas) |
| :--- | :--- | :--- |
| 内存层级 | SRAM (~160KB) | VMEM (16MB+) |
| 并行度 | 多 SM 并行 | 单核顺序 + 流水线 |
| Block size | BM=BK=64-128（受限于 SRAM）| BM=BK=128-512 |
| 反向传播 | 重计算 | 重计算 |
| 编程难度 | 极高（手写 CUDA + PTX）| 中等（Python + Pallas）|

## 多头注意力与 GQA

```python
# MHA: 每个 head 独立计算
grid = (batch_size, num_heads, seq_len // BM)

# GQA: 多个 query head 共享 KV head
kv_head_idx = q_head_idx // num_groups
```

## 反向传播

FlashAttention 的反向传播不保存中间的 attention score 矩阵，需要在反向时**重计算**。这增加了计算量但大幅减少了内存使用。

在 TPU 上，反向传播的分块策略可能与前向不同（有独立的 `block_q_major_dkv`、`block_k_major_dq` 参数），因为梯度累加的访问模式与前向不同。
