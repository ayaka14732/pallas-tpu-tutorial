# 第 13 章：Ragged Paged Attention (RPA v3)

本章分析 vLLM 项目中的 Ragged Paged Attention v3 kernel（[源码](https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py)，[论文](https://arxiv.org/abs/2604.15464)）。这个 kernel 综合运用了前面所有章节的技术：手动 DMA、双缓冲流水线、标量预取、在线 Softmax、Megacore 并行。

## 问题定义

在 LLM serving 中，一个 batch 内同时存在：
- **Decode 请求**：q_len=1，KV cache 很长（数千 token）
- **Prefill 请求**：q_len 很大（数百到数千），KV cache 从 0 开始增长

传统做法是分开处理（decode batch 和 prefill batch 分别调度），但这浪费了 TPU 算力——decode 是 memory-bound，prefill 是 compute-bound，混合处理可以更好地利用硬件。

## 三种调度模式

RPA v3 根据 batch 组成选择不同的 kernel 配置：

| 模式 | 条件 | q_len 处理 | 优化目标 |
| :--- | :--- | :--- | :--- |
| Decode | 所有请求 q_len=1 | static_q_len=1 | 最大化 HBM 带宽利用 |
| Prefill | 所有请求 q_len 相同 | chunk_prefill_size | 最大化 MXU 利用 |
| Mixed | 混合 | 动态 | 平衡两者 |

每种模式有独立的 block size 配置和 `pallas_call` 调用。

## Grid 设计

```python
grid = (1,)  # 只有一个 grid cell
```

所有循环逻辑在 kernel 内部通过 `pl.loop` 实现。原因：不同序列的工作量不同（ragged），无法用静态 grid 表达。

### 循环结构

```
pl.loop(start_seq_idx, end_seq_idx)          # 遍历序列
    pl.loop(0, num_bq)                        # 遍历 Q 块
        pl.loop(start_bkv_idx, end_bkv_idx)   # 遍历 KV 块
            pl.loop(0, num_loops)              # 计算子块
```

## 内存管理

### 三组双缓冲

```python
# Scratch buffers (VMEM)
bkv_buf: (2, BKV, head_dim)    # KV cache 双缓冲
bq_buf:  (2, BQ, head_dim)     # Query 双缓冲
bo_buf:  (2, BQ, head_dim)     # Output 双缓冲
```

四组 DMA 信号量：
```python
sems: SemaphoreType.DMA((4, 2))
# sems[0]: bkv 信号量
# sems[1]: bq 信号量
# sems[2]: bo 信号量
# sems[3]: kv_cache_update 信号量
```

### Paged KV Cache 布局

```
[total_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
```

K 和 V 交错存储（`num_kv_heads_x2`），并且可能被 pack 到一起（`kv_packing`）以减少 DMA 次数。

## 手动 DMA：_fetch_bkv

```python
def _fetch_bkv(page_indices_ref, kv_cache_ref, bkv_buf, sem, block_idx):
    pages_per_block = BKV // page_size

    # 逐 page 发起 DMA（page 物理不连续）
    for p in range(pages_per_block):
        page_idx = page_indices_ref[block_idx * pages_per_block + p]
        pltpu.make_async_copy(
            kv_cache_ref.at[page_idx, :, head_idx, :, :],
            bkv_buf.at[p * page_size : (p+1) * page_size, :],
            sem,
        ).start()
```

每个 page 需要单独的 DMA 操作，因为物理页不连续。

## FlashAttention 分步实现

RPA v3 将 FlashAttention 拆分为两步，实现 MXU 和 VPU 的重叠：

### Step 1: QK + Softmax (VPU 密集)

```python
def step1_qk_softmax(q_buf, k_buf, m_scratch, l_scratch):
    # QK^T (MXU)
    qk = jnp.dot(q_buf, k_buf.T, preferred_element_type=jnp.float32) * scale

    # Causal mask (VPU)
    mask = lax.broadcasted_iota(...) >= lax.broadcasted_iota(...)
    qk = jnp.where(mask, qk, -jnp.inf)

    # Online softmax (VPU)
    m_old = m_scratch[...]
    m_new = jnp.maximum(m_old, jnp.max(qk, axis=-1))
    correction = jnp.exp(m_old - m_new)
    p = jnp.exp(qk - m_new[:, None])
    l_scratch[...] = l_scratch[...] * correction + jnp.sum(p, axis=-1)
    m_scratch[...] = m_new

    return p, correction
```

### Step 2: PV 累加 (MXU 密集)

```python
def step2_pv(p, v_buf, correction, o_scratch):
    o_scratch[...] = o_scratch[...] * correction[:, None]
    o_scratch[...] += jnp.dot(
        p.astype(jnp.bfloat16),
        v_buf.astype(jnp.bfloat16),
        preferred_element_type=jnp.float32,
    )
```

Step 2 的 MXU 操作可以与下一次 Step 1 的 VPU 操作重叠。

## 主循环流水线

```python
# Prologue
fetch_bkv(0, buf=0)
fetch_bq(0, buf=0)

@pl.loop(0, num_kv_blocks)
def _(i):
    buf = i % 2

    # 等待当前 KV 块就绪
    wait_dma(bkv_sem[buf])

    # 预取下一个 KV 块（与计算重叠）
    @pl.when(i + 1 < num_kv_blocks)
    def _():
        fetch_bkv(i + 1, buf=1-buf)

    # Step 1: QK + softmax
    p, correction = step1_qk_softmax(q, k_buf[buf])

    # Step 2: PV 累加
    step2_pv(p, v_buf[buf], correction, o_scratch)

# Epilogue: 归一化并写回
inv_l = pl.reciprocal(l_scratch[...], approx=True)
o_ref[...] = (o_scratch[...] * inv_l[:, None]).astype(o_ref.dtype)
```

## Strided Load/Store

由于 KV cache 的 packed 布局，需要 strided 操作提取 K 和 V：

```python
def strided_load(src_ref, dst_buf, stride):
    src_reinterpreted = pltpu.bitcast(src_ref, target_dtype)
    dst_buf[...] = src_reinterpreted[::stride, :]
```

## Bank Conflict 避免

当多个 DMA 操作同时访问 VMEM 的同一 bank 时产生冲突：

```python
# 增加 stride 偏移避免 bank conflict
bkv_stride = num_kv_heads_x2_per_kv_packing + 1  # +1 避免冲突
```

## CompilerParams 配置

```python
compiler_params = pltpu.CompilerParams(
    dimension_semantics=(pltpu.GridDimensionSemantics.PARALLEL,),
    vmem_limit_bytes=...,
    disable_bounds_checks=True,
    disable_semaphore_checks=True,
)
```

生产环境关闭检查以提高性能。

## 设计决策分析

### 为什么 grid=(1,)？

Ragged batch 中每个序列的工作量不同。静态 grid 需要 padding 到最大长度，浪费算力。`pl.loop` 在 kernel 内部动态循环，精确处理每个序列的实际长度。

### 为什么三种模式？

不同 q_len 的最优 block size 不同：
- Decode (q_len=1)：BQ 小，BKV 大（memory-bound，最大化带宽）
- Prefill (q_len 大)：BQ 和 BKV 都大（compute-bound，最大化 MXU）

### 为什么手动 DMA？

Paged KV cache 的物理页不连续，且每个序列的页数不同。BlockSpec 只能表达静态、规则的访问模式。手动 DMA 是唯一选择。

### 为什么 K/V packed 存储？

减少 DMA 次数。一次 DMA 同时获取 K 和 V，在 VMEM 中用 strided_load 分离。DMA 带宽是瓶颈时，减少 DMA 次数比增加 VMEM 计算更划算。

## 关键 API 汇总

| API | 用途 |
| :--- | :--- |
| `pl.loop` | Pallas 循环 |
| `pl.when` | 条件执行 |
| `pl.ds` | 动态切片 |
| `pl.reciprocal(x, approx=True)` | 快速倒数 |
| `pltpu.make_async_copy` | 手动 DMA |
| `pltpu.bitcast` | 类型重解释 |
| `pltpu.PrefetchScalarGridSpec` | 标量预取 |
| `pltpu.SemaphoreType.DMA` | DMA 信号量 |
| `pltpu.VMEM` / `pltpu.HBM` | 内存空间标记 |
| `pltpu.CompilerParams` | 编译器参数 |
| `pltpu.GridDimensionSemantics.PARALLEL` | 多核并行 |
| `lax.broadcasted_iota` | 即时生成 mask |
| `input_output_aliases` | 原地更新 |

## 性能优化清单

1. **三级流水线**：DMA 预取、QK 计算、PV 累加三者重叠
2. **双缓冲**：所有数据通路都使用双缓冲
3. **Packed 布局**：减少 DMA 次数
4. **Bank conflict 避免**：stride 偏移
5. **Megacore 并行**：batch × head 维度并行到多核
6. **动态循环**：精确处理 ragged 工作量，无 padding 浪费
7. **编译器提示**：关闭不必要的检查，设置 VMEM 限制
8. **近似倒数**：`pl.reciprocal(x, approx=True)` 避免昂贵的除法
