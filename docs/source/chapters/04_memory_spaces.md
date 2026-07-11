# 第 4 章：内存空间与 DMA

## TPU 内存空间总览

| 空间 | 常量 | 容量 | 特点 |
| :--- | :--- | :--- | :--- |
| HBM | `pltpu.HBM` | 16-96GB | 主存。不可直接计算。DMA 访问。 |
| VMEM | `pltpu.VMEM` | 16MB+ | 向量内存。所有向量/矩阵计算的操作数必须在此。 |
| SMEM | `pltpu.SMEM` | 数百 KB | 标量内存。支持随机访问。用于索引、控制流数据。 |
| CMEM | `pltpu.CMEM` | 有限 | 跨核心共享内存。用于多核通信。 |
| VMEM_SHARED | `pltpu.VMEM_SHARED` | - | 共享 VMEM（多核场景）。 |
| SEMAPHORE | `pltpu.SEMAPHORE` | - | 信号量空间。 |
| HOST | `pl.HOST` | - | 主机内存。 |

## 自动 DMA vs 手动 DMA

Pallas 提供两种内存管理模式：

**自动模式**（通过 BlockSpec 的 block_shape 和 index_map）：
编译器自动生成 HBM ↔ VMEM 的 DMA 代码。适用于访问模式规则的场景。

**手动模式**（通过 `memory_space=pltpu.HBM` + `make_async_copy`）：
kernel 收到 HBM Ref，自己控制何时、搬运多少数据到 VMEM。适用于：
- 不规则访问模式（如 paged attention 的按页取数据）
- 需要精细控制流水线时序的场景
- 数据量动态变化的场景

## make_async_copy：手动 DMA

```python
# 创建异步拷贝描述符
copy = pltpu.make_async_copy(
    src_ref,    # 源 Ref（HBM 或 VMEM）
    dst_ref,    # 目标 Ref（VMEM 或 HBM）
    sem_ref,    # DMA 信号量
)

# 启动传输（非阻塞）
copy.start()

# ... 在此期间可以执行其他计算 ...

# 等待传输完成（阻塞）
copy.wait()
```

信号量是 TPU DMA 引擎的同步机制。每次 DMA 完成时，硬件会自动递增信号量。`wait()` 会阻塞直到信号量达到预期值。

一个常见的模式是发起多次 DMA 到同一个信号量，然后一次性 wait。这在 paged attention 中很常见——逐页发起 DMA，最后统一等待：

```python
# 发起多次 DMA，共用一个信号量
for page_idx in range(num_pages):
    pltpu.make_async_copy(
        cache_ref.at[pl.ds(page_indices[page_idx] * page_size, page_size)],
        vmem_buf_ref.at[pl.ds(page_idx * page_size, page_size)],
        sem_ref
    ).start()

# 一次性等待所有 DMA 完成
# 通过对同一个 sem 再做一次 start+wait（src=dst 的空拷贝）来实现 fence
pltpu.make_async_copy(vmem_buf_ref.at[pl.ds(0, 0)], vmem_buf_ref.at[pl.ds(0, 0)], sem_ref).wait()
```

## 信号量类型

```python
# DMA 信号量：用于 HBM ↔ VMEM 的数据传输同步
pltpu.SemaphoreType.DMA((num_semaphores,))

# 常规信号量：用于核心间同步
pltpu.SemaphoreType.REGULAR((num_semaphores,))

# 屏障信号量：用于全局屏障
pltpu.SemaphoreType.BARRIER((num_semaphores,))
```

信号量的底层操作：
```python
pl.semaphore_signal(sem_ref)  # 递增信号量
pl.semaphore_wait(sem_ref)    # 等待信号量递增
pl.semaphore_read(sem_ref)    # 读取当前值
```

## 双缓冲（Double Buffering）

双缓冲是 TPU kernel 中最基本的优化模式。核心思想：使用两块 VMEM 缓冲区交替工作——当计算单元处理缓冲区 A 中的数据时，DMA 引擎同时将下一批数据加载到缓冲区 B。

```python
def kernel(x_hbm_ref, o_hbm_ref, buf_ref, sem_ref):
    # buf_ref: shape (2, block_size, dim) — 两个缓冲区
    # sem_ref: shape (2,) — 两个信号量

    # Prologue：启动第一次 DMA
    pltpu.make_async_copy(
        x_hbm_ref.at[pl.ds(0, block_size)],
        buf_ref.at[0],
        sem_ref.at[0]
    ).start()

    @pl.loop(0, num_blocks)
    def _(i):
        cur = i % 2
        nxt = 1 - cur

        # 等待当前数据就绪
        pltpu.make_async_copy(
            x_hbm_ref.at[pl.ds(i * block_size, block_size)],
            buf_ref.at[cur],
            sem_ref.at[cur]
        ).wait()

        # 启动下一次 DMA
        @pl.when(i + 1 < num_blocks)
        def _():
            pltpu.make_async_copy(
                x_hbm_ref.at[pl.ds((i+1) * block_size, block_size)],
                buf_ref.at[nxt],
                sem_ref.at[nxt]
            ).start()

        # 计算（与下一次 DMA 重叠）
        result = some_computation(buf_ref.at[cur][...])
        # 写回结果...
```

RPA v3 kernel 使用了**三重双缓冲**：bkv（KV cache）、bq（queries）、bo（output）各有两个缓冲区，加上 4 对 DMA 信号量（`SemaphoreType.DMA((4, 2))`），实现了输入加载、计算、输出写回的完全重叠。

## Scratch Shapes

Scratch buffer 是 kernel 内部使用的临时内存，不对应任何输入或输出：

```python
scratch_shapes = [
    pltpu.VMEM((1024, 128), jnp.float32),     # VMEM 中的临时缓冲区
    pltpu.SMEM((16,), jnp.int32),             # SMEM 中的索引缓冲区
    pltpu.SemaphoreType.DMA((2,)),            # DMA 信号量
]
```

Scratch buffer 作为额外的 Ref 参数传入 kernel（在输出之后）：

```python
def kernel(x_ref, o_ref, scratch_vmem_ref, scratch_smem_ref, sem_ref):
    ...
```

Scratch buffer 的生命周期是**整个 Grid 执行期间**。Grid 从 `i=0` 推进到 `i=1` 时，scratch 中的数据会被保留。这是 TPU 顺序执行模型的直接优势——可以跨迭代累加，无需原子操作。

## SMEM 的用途

SMEM 附属于标量核心，支持 O(1) 随机访问。典型用途：

1. **动态索引**：page table、token 长度等需要按序列号查找的数据
2. **控制流变量**：循环计数器、条件标志、信号量索引追踪
3. **小型查找表**：如 RPA v3 中的 `kv_lens`、`page_indices`、`cu_q_lens`

通过 `PrefetchScalarGridSpec` 可以将 SMEM 数据作为 scalar prefetch 参数传入（详见第 10 章）。

## with_memory_space_constraint

`pltpu.with_memory_space_constraint` 用于显式指定一个数组应该驻留在哪个内存空间：

```python
# 确保输入驻留在 HBM（防止编译器将其移到 VMEM）
q_hbm = pltpu.with_memory_space_constraint(q, pltpu.HBM)
```

这在 TPU v7+ 上特别重要。RPA v3 kernel 中，`pallas_call` 的调用方会对输入做 `with_memory_space_constraint(q, pltpu.HBM)` 来确保大张量不被意外拷贝到 VMEM。

## run_scoped：动态临时内存

`pl.run_scoped` 允许在 kernel 内部动态分配临时内存：

```python
def kernel(x_ref, o_ref):
    def body(tmp_ref):
        tmp_ref[...] = x_ref[...] * 2
        o_ref[...] = tmp_ref[...]

    pl.run_scoped(body, pltpu.VMEM((128, 128), jnp.float32))
```

## sync_copy

`pltpu.sync_copy` 是 `make_async_copy().start()` + `.wait()` 的简写：

```python
pltpu.sync_copy(src_ref, dst_ref, sem_ref)
```

## 内存空间选择决策

1. 大张量、流式处理 → `pltpu.HBM`（手动 DMA 或 BlockSpec 自动管理）
2. 计算操作数 → `pltpu.VMEM`（必须）
3. 索引、长度、页表等标量数据 → `pltpu.SMEM`
4. 跨核心共享的小数据 → `pltpu.CMEM`
5. 同步原语 → `pltpu.SemaphoreType.*`
