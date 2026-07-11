# 第 13 章：源码剖析 - Ragged Paged Attention

在前面的章节中，我们学习了 TPU 架构、BlockSpec、流水线、内存空间和标量预取。本章我们将把所有这些知识点融会贯通，剖析 JAX 官方代码库中最复杂、也最具代表性的生产级 Kernel 之一：**Ragged Paged Attention**。

这个 Kernel 旨在解决大语言模型（LLM）推理中的核心痛点：混合 Prefill（预填充）和 Decode（解码）阶段的高效注意力计算，同时支持 Paged KV Cache。

*注：本章基于 JAX GitHub 仓库中的 `jax.experimental.pallas.ops.tpu.ragged_paged_attention.kernel` 源码进行分析。*

## 挑战与需求

在 LLM 推理服务器（如 vLLM）中，批处理（Batching）的请求往往具有不同的序列长度（Ragged），并且它们的 KV Cache 被打散存储在不连续的物理内存页（Pages）中。

要在 TPU 上高效实现这个 Kernel，我们需要解决以下问题：
1. **动态索引**：KV Cache 的物理地址是动态的，需要通过 `page_indices` 查找。
2. **长度不一**：每个请求的 Query 长度和 KV 长度都不同，需要动态控制循环次数。
3. **极高的内存带宽要求**：必须使用双缓冲（Double-buffering）隐藏 KV Cache 的加载延迟。
4. **混合计算**：有的请求在做 Prefill（长 Query），有的在做 Decode（单 Query），计算负载极度不均衡。

## 整体架构与 Grid 设计

```python
# 核心网格设计
grid = (num_heads_blks, num_q_blks)

compiler_params = pltpu.CompilerParams(
    dimension_semantics=("arbitrary", "arbitrary")
)
```

Kernel 的 Grid 维度是 `(头分组数量, Query块数量)`。
非常值得注意的是 `dimension_semantics=("arbitrary", "arbitrary")`。由于每个 Query 块的工作量可能完全不同（取决于它对应的 KV 长度），如果强制顺序执行或固定并行，会导致严重的核心空闲。标记为 `arbitrary` 允许 TPU 编译器（Mosaic）在 Megacore（双核）上自由地进行负载均衡调度。

## 标量预取 (Scalar Prefetch) 的极致应用

由于每个 Sequence 的元数据（长度、页索引等）各不相同，这些数据必须被放入低延迟的 SMEM 中供控制流使用。

```python
scalar_prefetches = (
    kv_lens,         # 每个序列的 KV 长度
    page_indices,    # 物理页映射表
    cu_q_lens,       # Query 长度的前缀和 (用于定位 Q 的起始位置)
    seq_buf_idx_ref, # 跨 Grid 传递状态的计数器
    num_seqs,        # 总序列数
)

grid_spec = pltpu.PrefetchScalarGridSpec(
    num_scalar_prefetch=5, # 前 5 个参数全部进入 SMEM
    in_specs=[q_block_spec, pl.BlockSpec(memory_space=pl.ANY)], # 注意 KV Cache 的空间是 ANY/HBM
    ...
)
```

注意，KV Cache `kv_pages` 被标记为 `memory_space=pl.ANY`（实际上驻留在 HBM），并且**没有给出具体的 BlockSpec 映射**。这是因为它的读取完全是动态的，无法在主机端静态描述。

## Scratch Buffers 与手动 DMA

为了在 Kernel 内部手动控制 KV Cache 的加载和流水线，代码分配了多个 VMEM 缓冲区和 DMA 信号量：

```python
double_buf_scratch = pltpu.VMEM(
    (2, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, head_dim),
    kv_pages.dtype,
)
scratch_shapes = [
    double_buf_scratch,             # kv_bufs: 长度为 2 的双缓冲
    pltpu.SemaphoreType.DMA((2,)),  # 两个 DMA 信号量，用于同步
    lm_scratch,                     # l_ref: FlashAttention 的 running sum
    lm_scratch,                     # m_ref: FlashAttention 的 running max
    acc_scratch,                    # acc_ref: 最终输出累加器
]
```

## Kernel 内部逻辑：The MultiPageAsyncCopyDescriptor

这是整个 Kernel 最硬核的部分。由于 KV Cache 是一页一页离散存储的，Kernel 内部定义了一个 `MultiPageAsyncCopyDescriptor` 类，使用 `pltpu.make_async_copy` 手动发起 HBM 到 VMEM 的异步拷贝。

```python
# 伪代码逻辑展示
for i in range(num_kv_pages_per_blk):
    # 从 SMEM 中的 page_indices 读取物理页号
    physical_page_idx = page_indices_ref[seq_id, logical_page_idx]
    
    # 构造从 HBM 到 VMEM 的异步拷贝指令，并绑定到信号量 sem
    async_copy = pltpu.make_async_copy(
        kv_pages_hbm_ref.at[physical_page_idx],
        vmem_buf.at[i],
        sem
    )
    async_copies.append(async_copy)
```

## 嵌套的 while_loop 与软件流水线

为了处理不同长度的 Sequence，Kernel 没有使用简单的 `for` 循环，而是使用了 `jax.lax.while_loop`。

外层 `while_loop` 遍历当前 Grid 负责的 Query 块，内层 `while_loop` 遍历该 Query 对应的所有 KV 块。

为了实现双缓冲流水线，代码在内层循环中采用了以下精妙的设计：
1. **当前步计算**：等待当前 Buffer 的信号量，获取 KV 数据。
2. **计算 Flash Attention**：调用 `flash_attention` 函数，更新 `m_ref`, `l_ref`, `acc_ref`。
3. **下一步预取**：在计算的同时，计算出下一个 KV 块的逻辑地址，读取物理页号，并向**另一个 Buffer** 发起异步 DMA 拷贝。

```python
# 内层 while_loop 的核心流水线逻辑 (简化版)
def compute_with_kv_blk_in_cur_seq(kv_states):
    # ... 计算下一个块的索引 ...
    
    # 1. 发起下一个块的异步预取 (到 next_buf_idx)
    @pl.when(next_heads_blk_idx < num_heads_blks)
    def prefetch_next_kv_blk():
        next_async_copy_kv.start()
        
    # 2. 等待当前块的 DMA 完成 (cur_buf_idx)
    kv_ref = cur_async_copy_kv.wait()
    
    # 3. 执行核心计算
    flash_attention(q, k, v, l_ref, m_ref, acc_ref)
    
    return kv_blk_idx + 1, next_buf_idx
```

## 总结

Ragged Paged Attention Kernel 是 Pallas 表达能力的巅峰展现。它证明了：
- Pallas 不仅仅能做静态的矩阵切块。
- 通过 SMEM 标量预取、VMEM Scratch 显式分配、以及底层的异步 DMA API（`make_async_copy`），开发者可以在 TPU 上实现极其复杂、高度动态的控制流和内存管理。
- 这使得 TPU 能够胜任最前沿的大模型推理需求，而不仅仅是传统的静态图训练。

如果你能完全理解这个 Kernel 的源码，你已经具备了顶级的 TPU Kernel 开发能力！
