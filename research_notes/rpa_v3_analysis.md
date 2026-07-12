# Ragged Paged Attention v3 Kernel 分析

> 论文：arXiv:2604.15464
> 代码：https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py

## 概述

Ragged Paged Attention (RPA) 是 vLLM 在 TPU 上的核心注意力 kernel。v3 版本支持三种调度模式，针对不同的 batch 特征自动选择最优策略。

## 三种调度模式

| 模式 | 适用场景 | 特点 |
| :--- | :--- | :--- |
| `BATCH_QUERY_PARALLEL` | 多请求、短序列 | 每个 grid 迭代处理一个请求 |
| `SEQUENCE_PARALLEL` | 少请求、长序列 | 一个请求的 KV 分散到多个 grid 迭代 |
| `QUERY_HEAD_PARALLEL` | 多头、中等序列 | 按 query head 分配 |

## 关键设计决策

### 1. `grid=(1,)` 设计

v3 kernel 使用 `grid=(1,)` + 内部手动循环，而不是用 Pallas 的 grid 来分配工作。原因：
- 需要在循环之间维护复杂的状态（多个累加器、多组信号量）
- 需要根据 ragged batch 的实际长度动态决定循环次数
- Pallas 的 grid 是静态的，无法处理 ragged 结构

### 2. 三组双缓冲

kernel 同时维护三组独立的双缓冲：
- **Q 缓冲**：query 数据（在 prefill 模式下需要分块加载）
- **K 缓冲**：key 页（从 paged KV cache 加载）
- **V 缓冲**：value 页（从 paged KV cache 加载）

每组有 buffer_0 和 buffer_1，交替使用。

### 3. 手动 DMA + 信号量

```python
# 典型模式：
# 1. 启动下一块的 DMA
async_copy = pltpu.make_async_copy(src_ref, dst_ref[next_buf], sem)
async_copy.start()

# 2. 计算当前块
compute(dst_ref[curr_buf])

# 3. 等待下一块 DMA 完成
async_copy.wait()

# 4. 交换 buffer
curr_buf, next_buf = next_buf, curr_buf
```

### 4. FlashAttention 分步实现

在每个 KV 块上执行 FlashAttention 的一步：
1. `S = Q @ K^T`（MXU 矩阵乘法）
2. `m_new = max(m_old, row_max(S))`（更新行最大值）
3. `correction = exp(m_old - m_new)`（修正因子）
4. `P = exp(S - m_new)`（注意力权重）
5. `O = O * correction + P @ V`（更新输出累加器）
6. `l = l * correction + row_sum(P)`（更新归一化因子）

最终：`output = O / l`

### 5. Paged KV Cache 访问

KV cache 按页存储（page_size 通常为 128 或 256 tokens）。kernel 通过 page table 索引来定位每一页：

```python
# page_indices: (max_num_pages,) 存储在 SMEM 中
page_idx = page_indices[page_offset]
k_page = k_pages[page_idx]  # 通过 DMA 加载
```

这就是 Scalar Prefetch 的典型用例：page_indices 存在 SMEM 中，用于计算 DMA 地址。

### 6. Strided Load/Store

由于 paged attention 的 KV cache 布局是 `(num_pages, page_size, num_heads, head_dim)`，而 kernel 需要按 head 访问，因此需要 strided DMA：

```python
# 加载特定 head 的 K 数据
# 需要跨越 num_heads 维度的 stride
```

### 7. Bank Conflict 避免

VMEM 有 bank 结构。当多个 sublane 同时访问同一 bank 时会产生冲突。v3 kernel 通过：
- 选择合适的 tile 布局
- 在必要时插入 padding
来避免 bank conflict。

## 代码结构

```
kernel.py
├── _ragged_paged_attention()          # 顶层入口
│   ├── 参数验证和形状计算
│   ├── pl.pallas_call(kernel_fn, ...)  # grid=(1,)
│   └── 后处理（reshape, 归一化）
│
├── kernel_fn()                         # 主 kernel 函数
│   ├── 初始化累加器（O, l, m）
│   ├── 初始化双缓冲
│   ├── 启动第一次 DMA
│   └── 主循环：
│       ├── 等待当前 DMA
│       ├── 计算 S = Q @ K^T
│       ├── 更新 m, l, O（FlashAttention 步骤）
│       ├── 启动下一次 DMA
│       └── 交换 buffer
│
├── _fetch_k_page() / _fetch_v_page()  # DMA 辅助函数
├── _compute_attention_step()           # 单步注意力计算
└── _finalize_output()                  # 最终归一化 O/l
```

## 性能优化要点

1. **DMA/计算完全重叠**：三组双缓冲确保 DMA 引擎始终忙碌
2. **MXU 利用率**：Q@K^T 和 P@V 都是矩阵乘法，充分利用 MXU
3. **最小化 HBM 访问**：Q 只加载一次，KV 流式加载
4. **SMEM 存储索引**：page table 在 SMEM 中，随机访问无代价
5. **在线算法**：不需要存储完整的 attention matrix

## 待深入研究的问题

- [ ] v3 相比 v2 的具体改进点
- [ ] 不同 batch 大小下三种模式的切换阈值
- [ ] 与 JAX 仓库中 `flash_attention.py` 的实现差异
- [ ] 在 v7 TPU（192MB VMEM）上是否可以用更大的 tile
- [ ] GQA (Grouped Query Attention) 的处理方式
