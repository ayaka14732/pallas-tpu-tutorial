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

核心规则：**所有计算必须在 VMEM 中进行**。数据从 HBM 到 VMEM 的搬运由 DMA 引擎负责。

## 自动 DMA vs 手动 DMA

Pallas 提供两种内存管理模式：

**自动模式**（BlockSpec 管理）：编译器根据 `BlockSpec` 的 `block_shape` 和 `index_map` 自动生成 DMA 代码。kernel 函数收到的 Ref 已经在 VMEM 中了。

**手动模式**（`memory_space=pltpu.HBM` + `make_async_copy`）：kernel 收到 HBM Ref，自己控制何时、搬运多少数据到 VMEM scratch buffer。

## 示例 1：自动 DMA（BlockSpec 管理）

这是最简单的模式。你只需要声明 block_shape 和 index_map，编译器自动处理 DMA：

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def add_kernel(x_ref, y_ref, o_ref):
    # x_ref, y_ref 已经在 VMEM 中了（编译器自动 DMA）
    o_ref[...] = x_ref[...] + y_ref[...]

def auto_dma_add(x, y):
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[
            pl.BlockSpec((128,), lambda i: (i,)),
            pl.BlockSpec((128,), lambda i: (i,)),
        ],
        out_specs=pl.BlockSpec((128,), lambda i: (i,)),
        grid=(x.shape[0] // 128,),
    )(x, y)

# 使用
x = jnp.ones(1024, dtype=jnp.float32)
y = jnp.ones(1024, dtype=jnp.float32)
result = auto_dma_add(x, y)
# 编译器自动生成了 8 次 DMA（1024 / 128 = 8 个块）
```

在这种模式下，你**完全不需要关心 DMA**。编译器会在每个 grid 步骤开始前自动将对应的块从 HBM 搬到 VMEM，计算完后自动将输出从 VMEM 搬回 HBM。

## 示例 2：手动 DMA（HBM → VMEM）

当你需要手动控制数据搬运时，使用 `memory_space=pltpu.HBM` 让输入保留在 HBM 中，然后在 kernel 内部用 `make_async_copy` 搬运：

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BLOCK = 128

def manual_dma_kernel(
    x_hbm_ref,      # HBM 中的输入（不可直接计算）
    o_hbm_ref,      # HBM 中的输出
    buf_ref,         # VMEM scratch buffer（计算用）
    out_buf_ref,     # VMEM scratch buffer（输出用）
    load_sem_ref,    # DMA 信号量（加载）
    store_sem_ref,   # DMA 信号量（存储）
):
    # Step 1: 发起 HBM → VMEM 的异步拷贝
    load_copy = pltpu.make_async_copy(
        src_ref=x_hbm_ref.at[pl.ds(0, BLOCK)],  # HBM 中的前 128 个元素
        dst_ref=buf_ref,                          # VMEM scratch buffer
        sem=load_sem_ref,                         # 绑定信号量
    )
    load_copy.start()  # 非阻塞：DMA 引擎开始搬运

    # Step 2: 等待 DMA 完成
    load_copy.wait()   # 阻塞：直到数据到达 VMEM

    # Step 3: 在 VMEM 中计算
    out_buf_ref[...] = buf_ref[...] * 2.0

    # Step 4: 将结果从 VMEM 写回 HBM
    store_copy = pltpu.make_async_copy(
        src_ref=out_buf_ref,                      # VMEM 中的计算结果
        dst_ref=o_hbm_ref.at[pl.ds(0, BLOCK)],   # HBM 输出位置
        sem=store_sem_ref,
    )
    store_copy.start()
    store_copy.wait()

def manual_dma_example(x):
    return pl.pallas_call(
        manual_dma_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        # 关键：输入和输出都标记为 HBM
        in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        # Scratch buffers 和信号量
        scratch_shapes=[
            pltpu.VMEM((BLOCK,), jnp.float32),       # buf_ref
            pltpu.VMEM((BLOCK,), jnp.float32),       # out_buf_ref
            pltpu.SemaphoreType.DMA(()),             # load_sem_ref
            pltpu.SemaphoreType.DMA(()),             # store_sem_ref
        ],
        grid=(1,),
    )(x)

# 使用
x = jnp.arange(1024, dtype=jnp.float32)
result = manual_dma_example(x)
# result[:128] == x[:128] * 2.0
```

**关键点：**
- `pl.BlockSpec(memory_space=pltpu.HBM)` 告诉编译器"不要自动搬运这个参数，把 HBM 引用直接传给 kernel"
- `pltpu.VMEM((BLOCK,), jnp.float32)` 在 scratch_shapes 中分配 VMEM 缓冲区
- `pltpu.SemaphoreType.DMA(())` 分配一个 DMA 信号量
- `make_async_copy` 返回一个描述符对象，调用 `.start()` 启动传输，`.wait()` 等待完成

## 示例 3：处理整个数组（循环 + 手动 DMA）

上面的示例只处理了前 128 个元素。实际中需要循环处理整个数组：

```python
def manual_dma_full_kernel(
    x_hbm_ref,
    o_hbm_ref,
    buf_ref,
    out_buf_ref,
    load_sem_ref,
    store_sem_ref,
):
    num_blocks = x_hbm_ref.shape[0] // BLOCK

    @pl.loop(0, num_blocks)
    def _(i):
        # 加载第 i 个块
        load_copy = pltpu.make_async_copy(
            src_ref=x_hbm_ref.at[pl.ds(i * BLOCK, BLOCK)],
            dst_ref=buf_ref,
            sem=load_sem_ref,
        )
        load_copy.start()
        load_copy.wait()

        # 计算
        out_buf_ref[...] = buf_ref[...] * 2.0

        # 存储第 i 个块
        store_copy = pltpu.make_async_copy(
            src_ref=out_buf_ref,
            dst_ref=o_hbm_ref.at[pl.ds(i * BLOCK, BLOCK)],
            sem=store_sem_ref,
        )
        store_copy.start()
        store_copy.wait()
```

注意：这个版本**没有**重叠 DMA 和计算。每次都是 load → wait → compute → store → wait，效率不高。下一个示例展示如何用双缓冲优化。

## 示例 4：双缓冲（DMA 与计算重叠）

双缓冲的核心思想：使用两块 VMEM 缓冲区交替工作。当计算单元处理缓冲区 0 中的数据时，DMA 引擎同时将下一批数据加载到缓冲区 1。

```python
def double_buffer_kernel(
    x_hbm_ref,
    o_hbm_ref,
    buf_ref,          # shape: (2, BLOCK) — 两个缓冲区
    out_buf_ref,      # shape: (2, BLOCK)
    load_sem_ref,     # shape: (2,) — 两个信号量
    store_sem_ref,    # shape: (2,)
):
    num_blocks = x_hbm_ref.shape[0] // BLOCK

    # Prologue: 预取第一个块到 buf[0]
    pltpu.make_async_copy(
        src_ref=x_hbm_ref.at[pl.ds(0, BLOCK)],
        dst_ref=buf_ref.at[0],
        sem=load_sem_ref.at[0],
    ).start()

    @pl.loop(0, num_blocks)
    def _(i):
        cur = i % 2   # 当前使用的缓冲区
        nxt = 1 - cur  # 下一个缓冲区

        # 等待当前块的 DMA 完成
        pltpu.make_async_copy(
            src_ref=x_hbm_ref.at[pl.ds(i * BLOCK, BLOCK)],
            dst_ref=buf_ref.at[cur],
            sem=load_sem_ref.at[cur],
        ).wait()

        # 启动下一个块的 DMA（与计算重叠）
        @pl.when(i + 1 < num_blocks)
        def _():
            pltpu.make_async_copy(
                src_ref=x_hbm_ref.at[pl.ds((i + 1) * BLOCK, BLOCK)],
                dst_ref=buf_ref.at[nxt],
                sem=load_sem_ref.at[nxt],
            ).start()

        # 计算（此时 DMA 引擎在后台搬运下一个块）
        out_buf_ref.at[cur][...] = buf_ref.at[cur][...] * 2.0

        # 将结果写回 HBM
        pltpu.make_async_copy(
            src_ref=out_buf_ref.at[cur],
            dst_ref=o_hbm_ref.at[pl.ds(i * BLOCK, BLOCK)],
            sem=store_sem_ref.at[cur],
        ).start()

        # 等待上一次 store 完成（确保 buf 可以被覆盖）
        @pl.when(i > 0)
        def _():
            pltpu.make_async_copy(
                src_ref=out_buf_ref.at[nxt],
                dst_ref=o_hbm_ref.at[pl.ds((i - 1) * BLOCK, BLOCK)],
                sem=store_sem_ref.at[nxt],
            ).wait()

def double_buffer_example(x):
    num_blocks = x.shape[0] // BLOCK
    return pl.pallas_call(
        double_buffer_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        scratch_shapes=[
            pltpu.VMEM((2, BLOCK), jnp.float32),     # buf_ref (双缓冲)
            pltpu.VMEM((2, BLOCK), jnp.float32),     # out_buf_ref (双缓冲)
            pltpu.SemaphoreType.DMA((2,)),           # load_sem_ref
            pltpu.SemaphoreType.DMA((2,)),           # store_sem_ref
        ],
        grid=(1,),
    )(x)
```

**时间线：**
```
时间 →
DMA:    [load B0] [load B1] [load B2] [load B3] ...
计算:            [comp B0] [comp B1] [comp B2] ...
Store:           [stor B0] [stor B1] [stor B2] ...
```

DMA 和计算完全重叠后，总时间 ≈ max(DMA 时间, 计算时间) × N，而不是 (DMA + 计算) × N。

## 示例 5：Scratch Buffers

Scratch buffer 是 kernel 内部使用的临时内存，不对应任何输入或输出。通过 `scratch_shapes` 参数分配：

```python
def kernel_with_scratch(x_ref, o_ref, tmp_ref, acc_ref):
    # tmp_ref: VMEM scratch，用于中间计算
    # acc_ref: VMEM scratch，用于跨迭代累加
    tmp_ref[...] = x_ref[...] * x_ref[...]  # 平方
    acc_ref[...] = acc_ref[...] + tmp_ref[...]  # 累加

def example_with_scratch(x):
    return pl.pallas_call(
        kernel_with_scratch,
        out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
        in_specs=[pl.BlockSpec((128,), lambda i: (i,))],
        out_specs=pl.BlockSpec((128,), lambda _: (0,)),
        scratch_shapes=[
            pltpu.VMEM((128,), jnp.float32),  # tmp_ref
            pltpu.VMEM((128,), jnp.float32),  # acc_ref
        ],
        grid=(8,),
    )(x)
```

**Scratch buffer 的生命周期**：整个 Grid 执行期间持续存在。Grid 从 `i=0` 推进到 `i=1` 时，scratch 中的数据会被保留。这是 TPU 顺序执行模型的优势——可以跨迭代累加，无需原子操作。

## 示例 6：信号量操作

信号量是 DMA 引擎的同步机制。底层操作：

```python
def semaphore_example(x_hbm_ref, o_ref, buf_ref, sem_ref):
    # DMA 完成时，硬件自动对 sem 做 signal（+1）
    pltpu.make_async_copy(
        x_hbm_ref.at[pl.ds(0, 128)],
        buf_ref,
        sem_ref,
    ).start()

    # wait 会阻塞直到 sem >= 1，然后 sem -= 1
    pltpu.make_async_copy(
        x_hbm_ref.at[pl.ds(0, 128)],
        buf_ref,
        sem_ref,
    ).wait()

    # 也可以手动操作信号量（用于核间同步）
    # pl.semaphore_signal(sem_ref)  # sem += 1
    # pl.semaphore_wait(sem_ref)    # 等待 sem >= 1, 然后 sem -= 1
    # val = pl.semaphore_read(sem_ref)  # 读取当前值
```

## 示例 7：多次 DMA 共用一个信号量

在 paged attention 中，需要从不连续的物理页加载数据。可以对同一个信号量发起多次 DMA：

```python
def multi_page_load(
    cache_hbm_ref,   # [total_pages, page_size, dim]
    page_indices_ref, # SMEM: [num_pages_to_load]
    buf_ref,          # VMEM: [num_pages_to_load * page_size, dim]
    sem_ref,
):
    num_pages = 4  # 假设加载 4 个不连续的页

    # 对同一个信号量发起 4 次 DMA
    for p in range(num_pages):
        page_idx = page_indices_ref[p]  # 从 SMEM 读取物理页号
        pltpu.make_async_copy(
            src_ref=cache_hbm_ref.at[page_idx, :, :],
            dst_ref=buf_ref.at[pl.ds(p * PAGE_SIZE, PAGE_SIZE)],
            sem=sem_ref,
        ).start()

    # 等待所有 4 次 DMA 完成
    # 信号量会被递增 4 次，wait 需要等到值 >= 4
    # 实际上 Pallas 的 wait 会自动处理这个计数
    pltpu.make_async_copy(
        src_ref=cache_hbm_ref.at[0, :, :],  # dummy src（wait 不关心 src）
        dst_ref=buf_ref.at[pl.ds(0, PAGE_SIZE)],
        sem=sem_ref,
    ).wait()
```

## 示例 8：SMEM 用法

SMEM 用于存放标量数据（索引、长度等），支持 O(1) 随机访问：

```python
def kernel_with_smem(
    x_hbm_ref,
    o_hbm_ref,
    indices_smem_ref,  # SMEM scratch: 存放索引
    buf_ref,           # VMEM scratch
    sem_ref,
):
    # 从 SMEM 读取索引（O(1) 随机访问）
    target_block = indices_smem_ref[0]

    # 根据索引动态决定 DMA 目标
    pltpu.make_async_copy(
        src_ref=x_hbm_ref.at[pl.ds(target_block * BLOCK, BLOCK)],
        dst_ref=buf_ref,
        sem=sem_ref,
    ).start()
    # ...

def smem_example(x, indices):
    return pl.pallas_call(
        kernel_with_smem,
        out_shape=jax.ShapeDtypeStruct((BLOCK,), x.dtype),
        in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        scratch_shapes=[
            pltpu.SMEM((16,), jnp.int32),           # indices_smem_ref
            pltpu.VMEM((BLOCK,), jnp.float32),      # buf_ref
            pltpu.SemaphoreType.DMA(()),             # sem_ref
        ],
        grid=(1,),
    )(x)
```

注意：SMEM scratch 需要在 kernel 外部通过 `PrefetchScalarGridSpec` 的 scalar prefetch 来填充数据（见第 10 章），或者在 kernel 内部手动写入。

## 示例 9：run_scoped（动态临时内存）

`pl.run_scoped` 允许在 kernel 内部动态分配临时 Ref，用完自动释放：

```python
def kernel_with_scoped(x_ref, o_ref):
    def body(tmp_ref):
        # tmp_ref 只在这个作用域内有效
        tmp_ref[...] = x_ref[...] * 2.0
        o_ref[...] = tmp_ref[...] + 1.0

    # 分配一个临时 VMEM buffer，执行 body，然后释放
    pl.run_scoped(body, pltpu.VMEM((128,), jnp.float32))
```

与 scratch_shapes 的区别：
- `scratch_shapes`：在整个 kernel 生命周期内存在
- `run_scoped`：只在作用域内存在，可以在循环内部使用以减少 VMEM 占用

## 示例 10：sync_copy（同步 DMA）

`pltpu.sync_copy` 是 `make_async_copy().start()` + `.wait()` 的简写，用于不需要重叠的场景：

```python
def simple_copy_kernel(x_hbm_ref, o_ref, buf_ref, sem_ref):
    # 同步拷贝：阻塞直到完成
    pltpu.sync_copy(
        src_ref=x_hbm_ref.at[pl.ds(0, BLOCK)],
        dst_ref=buf_ref,
        sem=sem_ref,
    )
    # 此时 buf_ref 中已有数据
    o_ref[...] = buf_ref[...] * 2.0
```

## 示例 11：with_memory_space_constraint

在 `pallas_call` 外部，确保张量驻留在指定的内存空间：

```python
# 在调用 pallas_call 之前，确保 q 在 HBM 中
# 防止编译器将大张量意外拷贝到 VMEM
q_hbm = pltpu.with_memory_space_constraint(q, pltpu.HBM)

result = pl.pallas_call(kernel, ...)(q_hbm, k, v)
```

这在 TPU v7+ 上特别重要。RPA v3 kernel 中对输入做 `with_memory_space_constraint` 来确保大张量不被意外拷贝。

## 示例 12：pl.ds（动态切片）

`pl.ds(start, size)` 用于在 Ref 上做动态切片。`start` 可以是运行时变量：

```python
def dynamic_slice_kernel(x_hbm_ref, o_ref, buf_ref, sem_ref):
    # i 是运行时变量
    i = pl.program_id(0)

    # pl.ds(start, size): start 可以是动态的
    pltpu.make_async_copy(
        src_ref=x_hbm_ref.at[pl.ds(i * BLOCK, BLOCK)],  # 动态偏移
        dst_ref=buf_ref,
        sem=sem_ref,
    ).start()
    # ...
```

对比：
- `ref.at[0:128]`：静态切片，编译时确定
- `ref.at[pl.ds(i * 128, 128)]`：动态切片，运行时确定

## 信号量类型总结

```python
# DMA 信号量：用于 HBM ↔ VMEM 数据传输同步
pltpu.SemaphoreType.DMA(())         # 单个信号量
pltpu.SemaphoreType.DMA((2,))       # 2 个信号量（双缓冲用）
pltpu.SemaphoreType.DMA((4, 2))     # 4×2 信号量数组

# 常规信号量：用于核心间同步
pltpu.SemaphoreType.REGULAR(())

# 屏障信号量：用于全局屏障
pltpu.SemaphoreType.BARRIER(())
```

## 内存空间选择决策树

```
需要计算？
├── 是 → VMEM（必须）
└── 否 → 需要随机访问标量？
          ├── 是 → SMEM
          └── 否 → 数据量大？
                    ├── 是 → HBM（手动 DMA 管理）
                    └── 否 → 跨核共享？
                              ├── 是 → CMEM / VMEM_SHARED
                              └── 否 → VMEM scratch
```

## 与 GPU 的对比

| 概念 | GPU (CUDA) | TPU (Pallas) |
| :--- | :--- | :--- |
| 主存 | Global Memory | HBM |
| 片上高速内存 | Shared Memory (~160KB) | VMEM (16MB+) |
| 数据搬运 | 线程直接 load/store | DMA 引擎异步搬运 |
| 同步机制 | `__syncthreads()` | 信号量 (Semaphore) |
| 动态索引 | 线程可以访问任意地址 | 需要 SMEM + 手动 DMA |
| 缓冲管理 | 程序员手动管理 shared memory | BlockSpec 自动 或 scratch + DMA |

GPU 的优势：编程模型简单（线程直接访问全局内存）。
TPU 的优势：VMEM 容量大 100 倍，DMA 引擎可以完全隐藏内存延迟。
