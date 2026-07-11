# 第 13 章：源码剖析 - Ragged Paged Attention

在前面的章节中，我们学习了 TPU 架构、BlockSpec、流水线、内存空间和标量预取。本章我们将把所有这些知识点融会贯通，剖析 JAX 官方代码库中最复杂、也最具代表性的生产级 Kernel 之一：**Ragged Paged Attention**。

这个 Kernel 旨在解决大语言模型（LLM）推理服务器（如 vLLM）中的核心痛点：混合 Prefill（预填充）和 Decode（解码）阶段的高效注意力计算，同时支持不连续的 Paged KV Cache。

*注：本章基于 JAX GitHub 仓库中的 `jax.experimental.pallas.ops.tpu.ragged_paged_attention.kernel` 源码进行分析。*

## 生产级推理的极端挑战

在真实的推理服务器中，情况远比标准的 FlashAttention 复杂：
1. **动态索引 (Paged Cache)**：KV Cache 不是连续的张量，而是打散存储在大小固定的物理内存页（Pages）中。我们需要通过 `page_indices` 查找物理地址。
2. **长度不一 (Ragged)**：批处理（Batching）的请求中，每个 Sequence 的 Query 长度和 KV 长度都不同。
3. **极高的内存带宽要求**：在 Decode 阶段，Query 长度通常为 1，这是一个极端的 Memory-bound 操作。必须使用双缓冲隐藏 KV Cache 的加载延迟。
4. **混合负载 (Chunked Prefill)**：同一个 Batch 中，可能有的请求在做 Prefill（长 Query），有的在做 Decode（单 Query），计算负载极度不均衡。

## 整体架构与 Megacore 负载均衡

```python
# 核心网格设计
grid = (num_heads_blks, num_q_blks)

compiler_params = pltpu.CompilerParams(
    dimension_semantics=("arbitrary", "arbitrary")
)
```

Kernel 的 Grid 维度是 `(头分组数量, Query块数量)`。
非常值得注意的是 `dimension_semantics=("arbitrary", "arbitrary")`。

**为什么不用 `parallel`？**
因为这是一个 Ragged 任务。有的 Query 块可能对应一个很长的 KV Cache，需要计算很久；有的 Query 块对应的 KV Cache 很短，瞬间就计算完了。如果强制使用 `parallel` 静态划分任务给 Megacore 的两个核心，必然导致一个核心早早空闲，另一个核心还在苦苦计算。

标记为 `arbitrary` 允许 TPU 编译器（Mosaic）和硬件调度器在运行时动态地进行**负载均衡**。先空闲下来的核心会自动抓取下一个 Grid 任务。

## 标量预取 (Scalar Prefetch) 的极致应用

由于每个 Sequence 的元数据（长度、页索引等）各不相同，这些数据必须被放入低延迟的 SMEM 中，供 Kernel 内部的控制流（`while_loop`）使用。

```python
scalar_prefetches = (
    kv_lens,         # 每个序列的真实的 KV 长度
    page_indices,    # 物理页映射表 [num_seqs, max_pages]
    cu_q_lens,       # Query 长度的前缀和 (用于定位 Q 在展平数组中的起始位置)
    seq_buf_idx_ref, # 跨 Grid 传递状态的计数器
    num_seqs,        # 总序列数
)

grid_spec = pltpu.PrefetchScalarGridSpec(
    num_scalar_prefetch=5, # 前 5 个参数全部进入 SMEM
    in_specs=[q_block_spec, pl.BlockSpec(memory_space=pl.ANY)], # 注意 KV Cache 的空间是 ANY/HBM
    ...
)
```

**关键点：** KV Cache `kv_pages` 被标记为 `memory_space=pl.ANY`（实际上驻留在 HBM），并且**没有给出具体的 BlockSpec 映射**。
为什么？因为它的读取完全是动态的。编译器在 Host 端根本不知道该搬运哪些页。搬运工作必须在 Kernel 内部由代码手动指挥。

## Scratch Buffers 与手动 DMA 信号量

为了在 Kernel 内部手动控制 KV Cache 的加载和流水线，代码分配了多个 VMEM 缓冲区和 DMA 信号量（Semaphores）：

```python
# 双缓冲的 KV Cache 容器
double_buf_scratch = pltpu.VMEM(
    (2, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, head_dim),
    kv_pages.dtype,
)
scratch_shapes = [
    double_buf_scratch,             # kv_bufs: 长度为 2 的双缓冲
    pltpu.SemaphoreType.DMA((2,)),  # 两个 DMA 信号量，用于同步双缓冲
    lm_scratch,                     # l_ref: FlashAttention 的 running sum
    lm_scratch,                     # m_ref: FlashAttention 的 running max
    acc_scratch,                    # acc_ref: 最终输出累加器
]
```

## 深入 Kernel：MultiPageAsyncCopyDescriptor

这是整个 Kernel 最硬核、最底层的地方。由于 KV Cache 是一页一页离散存储的，Kernel 内部定义了一个 `MultiPageAsyncCopyDescriptor` 类。它在循环内部，根据 SMEM 中的动态索引，手动发起从 HBM 到 VMEM 的异步拷贝。

```python
# 伪代码逻辑展示
def start_async_copy(seq_id, logical_page_start_idx, vmem_buf, sem):
    for i in range(num_kv_pages_per_blk):
        # 1. 从 SMEM 中的 page_indices 读取物理页号
        physical_page_idx = page_indices_ref[seq_id, logical_page_start_idx + i]
        
        # 2. 构造从 HBM 到 VMEM 的异步拷贝指令
        # 注意：kv_pages_hbm_ref.at[physical_page_idx] 是动态的 HBM 地址
        async_copy = pltpu.make_async_copy(
            kv_pages_hbm_ref.at[physical_page_idx],
            vmem_buf.at[i],
            sem # 绑定到信号量
        )
        async_copies.append(async_copy)
    
    # 触发 DMA 引擎开始搬运
    start_all_copies(async_copies)
```

## 嵌套的 while_loop 与手动软件流水线

由于每个 Sequence 的长度不同，Kernel 无法使用简单的 `for` 循环，必须使用 `jax.lax.while_loop`。

外层 `while_loop` 遍历当前 Grid 负责的 Query 块，内层 `while_loop` 遍历该 Query 对应的所有 KV 块。

因为我们放弃了 `emit_pipeline`（它不支持动态循环和离散页），我们必须**手动实现双缓冲流水线**：

```python
# 内层 while_loop 的核心流水线逻辑 (高度简化版)
def compute_with_kv_blk_in_cur_seq(kv_states):
    cur_buf_idx = kv_states.buf_idx
    next_buf_idx = 1 - cur_buf_idx # 切换缓冲区 (0 -> 1 -> 0)
    
    # 1. 发起下一个块的异步预取 (到 next_buf_idx)
    # 这会在后台运行，不阻塞计算
    @pl.when(has_next_kv_blk)
    def prefetch_next_kv_blk():
        start_async_copy(seq_id, next_logical_page, kv_bufs[next_buf_idx], semaphores[next_buf_idx])
        
    # 2. 等待当前块的 DMA 完成
    # 如果 DMA 还没搬完，计算单元会在这里挂起等待
    semaphores[cur_buf_idx].wait()
    
    # 3. 执行核心计算 (FlashAttention)
    # 使用 VMEM 中的 cur_buf_idx 数据
    flash_attention(q, kv_bufs[cur_buf_idx], l_ref, m_ref, acc_ref)
    
    # 4. 释放当前缓冲区的信号量，表示计算完毕，DMA 引擎可以覆盖它了
    semaphores[cur_buf_idx].signal()
    
    return next_state
```

## 总结

Ragged Paged Attention Kernel 是 Pallas 表达能力的巅峰展现。它证明了：
- Pallas 不仅仅能做静态的矩阵切块（像大多数深度学习编译器做的那样）。
- 通过 SMEM 标量预取、VMEM Scratch 显式分配、以及底层的异步 DMA API（`make_async_copy` 和信号量），开发者可以在 TPU 上实现极其复杂、高度动态的控制流和内存管理。
- 这种在高级 Python 语法下直接操纵底层硬件的能力，使得 TPU 能够胜任最前沿、最苛刻的大模型推理需求。

如果你能完全理解这个 Kernel 的源码结构和设计动机，你已经具备了顶级的 TPU Kernel 开发能力！祝你在面试中脱颖而出！
