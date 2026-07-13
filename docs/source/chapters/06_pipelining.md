# 第 6 章：软件流水线

## 为什么需要流水线

TPU 是顺序执行机器。不做流水线时，执行模式是：

```
DMA_load → 计算 → DMA_store → DMA_load → 计算 → DMA_store → ...
```

每个阶段都在等待前一个阶段完成。DMA 引擎和计算单元的利用率都只有约 33%。

流水线的目标是让 DMA 和计算**重叠执行**：

```
时间 →
DMA_load:  [块0] [块1] [块2] [块3] ...
计算:            [块0] [块1] [块2] ...
DMA_store:       [块0] [块1] [块2] ...
```

当计算单元处理块 i 时，DMA 引擎同时加载块 i+1 并存储块 i-1 的结果。理想情况下，总时间 = max(计算时间, 传输时间)。

## 与 GPU 的对比

| 维度 | GPU | TPU |
| :--- | :--- | :--- |
| 延迟隐藏机制 | 硬件 warp 调度器自动切换 | 软件显式编排 |
| 程序员负担 | 低（保证足够 occupancy 即可）| 高（需要手动或半手动安排流水线）|
| 灵活性 | 低（受限于硬件调度策略）| 高（可以精确控制每个阶段的时序）|
| 缓冲区空间 | 共享内存有限（~160KB），通常只能单缓冲 | VMEM 很大（16MB+），可以多缓冲 |

GPU 上的 "流水线" 通常指 software pipelining in registers（如 CUTLASS 的 multi-stage pipeline），需要在极其有限的共享内存中腾挪。TPU 的 VMEM 足够大，可以轻松容纳多个完整的 tile。

## 自动流水线

对于简单的 `pallas_call`，编译器会自动应用流水线优化。只要 Grid 遍历顺序是连续的，编译器会尝试预取下一个块。但在复杂算子中（MatMul、FlashAttention），需要显式控制流水线。

## emit_pipeline

`pltpu.emit_pipeline` 是 Pallas 提供的流水线构建器。它将"切分 + DMA + 双缓冲 + 计算"封装为高层 API：

```python
def matmul_kernel(a_vmem_ref, b_vmem_ref, acc_ref):
    # a_vmem_ref 和 b_vmem_ref 已经在 VMEM 中（由 emit_pipeline 管理）
    acc_ref[...] += jnp.dot(a_vmem_ref[...], b_vmem_ref[...])

def outer_kernel(a_hbm_ref, b_hbm_ref, c_hbm_ref, acc_ref):
    acc_ref[...] = jnp.zeros_like(acc_ref[...])

    pltpu.emit_pipeline(
        matmul_kernel,
        grid=(K // BK,),
        in_specs=[
            pl.BlockSpec((BM, BK), lambda k: (0, k), memory_space=pltpu.VMEM),
            pl.BlockSpec((BK, BN), lambda k: (k, 0), memory_space=pltpu.VMEM),
        ],
        out_specs=[
            pl.BlockSpec((BM, BN), lambda k: (0, 0), memory_space=pltpu.VMEM),
        ],
    )(a_hbm_ref, b_hbm_ref, acc_ref)

    c_hbm_ref[...] = acc_ref[...]
```

`emit_pipeline` 的关键参数：
- `grid`：流水线的迭代次数
- `in_specs` / `out_specs`：数据在 VMEM 中的切分方式（`memory_space=pltpu.VMEM`）
- 它会自动处理双缓冲、DMA 调度、信号量同步

## 嵌套流水线模式

高性能 MatMul 的标准模式是**两层流水线**：

1. **外层 `pallas_call`**：Grid 为 `(M//BM, N//BN)`，负责将 A 和 B 的大块从 HBM 搬到 VMEM
2. **内层 `emit_pipeline`**：在 VMEM 内部沿 K 维循环，将 K 维切块送入 MXU

为什么不把 K 维放在外层 Grid 中？因为那样每次 K 迭代都会触发累加器从 HBM 加载和写回（HBM Thrashing）。将 K 维放在内层流水线中，累加器始终驻留在 VMEM，只在 K 循环结束后写回一次。

## 流水线的数学分析

对于 N 次迭代的流水线：

- **无流水线**：总时间 = N × (T_load + T_compute + T_store)
- **有流水线**：总时间 ≈ T_prologue + N × max(T_load, T_compute, T_store) + T_epilogue

流水线效率取决于**最慢的阶段**：
- Memory-bound（T_load > T_compute）：流水线不能消除瓶颈，但隐藏了计算延迟
- Compute-bound（T_compute > T_load）：流水线完全隐藏了 DMA 延迟

## pl.loop 与手动流水线

当 `emit_pipeline` 的抽象不够灵活时（如 RPA v3 的不规则访问模式），需要用 `pl.loop` 手写流水线：

```python
@pl.loop(0, num_iters)
def _(i):
    cur_buf = i % 2
    nxt_buf = 1 - cur_buf

    # 等待当前数据就绪
    pltpu.make_async_copy(src.at[...], buf.at[cur_buf], sem.at[cur_buf]).wait()

    # 启动下一次预取
    @pl.when(i + 1 < num_iters)
    def _():
        pltpu.make_async_copy(src.at[...], buf.at[nxt_buf], sem.at[nxt_buf]).start()

    # 计算（与下一次 DMA 重叠）
    result = compute(buf.at[cur_buf][...])
    out_ref[...] = result
```

`pl.loop` 支持 `init_carry` 参数用于跨迭代传递状态：

```python
@pl.loop(0, N, init_carry=(running_max, running_sum))
def _(i, carry):
    m_prev, l_prev = carry
    # ... 更新 running max 和 sum ...
    return (m_new, l_new)
```

## 流水线设计的关键决策

**Block size 选择**：
- 太小：DMA 启动开销占比大，MXU 无法充分利用
- 太大：VMEM 容纳不下双缓冲
- 经验法则：选择使 VMEM 使用量在 50-70% 的 block size
- 可通过 `pltpu.CompilerParams(vmem_limit_bytes=...)` 限制 VMEM 使用量来调试

**流水线深度**：
- 双缓冲（depth=2）：最常见
- 三缓冲（depth=3）：当 DMA 延迟特别大时有用
- RPA v3 使用三重双缓冲（bkv、bq、bo 各有两个缓冲区）

**多级流水线**：
- 外层处理大块数据的 HBM ↔ VMEM 传输
- 内层处理 VMEM 内部的计算分块
- 在大矩阵乘法中很常见

## emit_pipeline_with_allocations

`pltpu.emit_pipeline_with_allocations` 是 `emit_pipeline` 的扩展版本，允许外部预分配 VMEM 缓冲区并绑定到流水线中。这在需要跨多个 `emit_pipeline` 调用共享缓冲区时有用。

## CompilerParams 中的流水线相关选项

```python
pltpu.CompilerParams(
    vmem_limit_bytes=8 * 1024 * 1024,  # 限制 VMEM 使用量（调试用）
    allow_input_fusion=True,            # 允许编译器融合输入的 DMA
    opt_level=3,                        # 优化级别
)
```

## Prologue / Epilogue 模式

手动流水线的标准结构：

```python
# Prologue：启动第一次 DMA
fetch_block(0, buf=0)

# Main loop：稳态执行
@pl.loop(0, N)
def _(i):
    wait_block(i, buf=i%2)
    @pl.when(i + 1 < N)
    def _():
        fetch_block(i+1, buf=(i+1)%2)
    compute(buf=i%2)
    store_result(i)

# Epilogue：等待最后的 DMA 完成（如果有输出 DMA）
wait_all_stores()
```

这个模式在 RPA v3 中被严格遵循：prologue 预取第一个 bq 和 bkv，main loop 处理所有序列，epilogue 等待所有输出 DMA 完成。
