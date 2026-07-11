# 第 4 章：内存空间与 DMA

## TPU 内存空间总览

| 空间 | 常量 | 容量（典型值） | 特点 |
| :--- | :--- | :--- | :--- |
| HBM | `pltpu.HBM` / `pl.ANY` | 32-192GB | 主存。不可直接计算。DMA 访问。 |
| VMEM (TensorCore) | `pltpu.VMEM` | 16-192MB / 核 | TensorCore 的向量内存。所有 TC 计算操作数必须在此。 |
| VMEM (SparseCore) | `pltpu.VMEM` | 256-512KB / subcore | SparseCore 每个 Vector Subcore 的本地向量内存。 |
| VMEM_SHARED | `pltpu.VMEM_SHARED` | 共享池 | SparseCore 所有 Vector Subcore + Scalar Subcore 共享的内存。 |
| SMEM | `pltpu.SMEM` | 16KB-1MB / 核 | 标量内存。支持随机访问。用于索引、控制流数据。 |
| SEMAPHORE | `pltpu.SemaphoreType.*` | - | 信号量空间。 |

核心规则：**所有计算必须在 VMEM 中进行**。数据从 HBM 到 VMEM 的搬运由 DMA 引擎负责。

## TensorCore VMEM vs SparseCore VMEM

TensorCore 和 SparseCore 各自拥有**物理上独立的 VMEM**，它们不共享：

| 属性 | TensorCore VMEM | SparseCore VMEM |
| :--- | :--- | :--- |
| 所属单元 | TensorCore（MXU + VPU） | 每个 Vector Subcore |
| 容量 | 16MB (v4) ~ 192MB (8i) | 256KB (v6e) ~ 512KB (v5p/v7) |
| 数量 | 每个 TensorCore 一块 | 每个 SparseCore 有 16 块（每个 subcore 一块）|
| 用途 | 密集计算（matmul、向量运算） | 稀疏/随机访问（gather、scatter、sort）|
| Pallas 常量 | `pltpu.VMEM` | `pltpu.VMEM`（相同常量，上下文决定）|

虽然 Pallas 中都用 `pltpu.VMEM` 表示，但编译器根据 kernel 运行在哪个 mesh（`TensorCoreMesh` vs `VectorSubcoreMesh`）来决定实际映射到哪块物理内存。

**VMEM_SHARED** 是 SparseCore 独有的概念：它是所有 Vector Subcore 和 Scalar Subcore 都能访问的共享内存区域。在其他文档中也被称为 "SPMEM"。它的主要用途是 Scalar Subcore 和 Vector Subcore 之间的数据交换。

## VMEM_SHARED 的用法

`VMEM_SHARED` 用于 SparseCore kernel 中 Scalar Subcore 和 Vector Subcore 之间的通信。典型模式：

1. Scalar Subcore 从 HBM 加载数据到 VMEM_SHARED
2. 通过信号量通知 Vector Subcore 数据已就绪
3. Vector Subcore 从 VMEM_SHARED 拷贝到自己的 VMEM 进行计算
4. 计算结果写回 HBM

### 信号量基础

由于 Scalar Subcore 和 Vector Subcore **并行执行**，它们之间需要信号量来同步：

```python
# 信号量是一个计数器
pl.semaphore_signal(sem, device_id=...)  # 将目标设备上的 sem 加 1
pl.semaphore_wait(sem, value=N)          # 阻塞直到本地 sem >= N，然后减 N
```

规则：
- 信号量在 kernel 结束时**必须为 0**，否则程序崩溃（over-signal）或挂起（over-wait）
- `device_id` 指定信号量所在的目标设备/subcore
- 对于跨 subcore 通信，需要指定 `device_id={"core": i, "subcore": j}`

### 完整示例：Scalar Subcore 加载 → Vector Subcore 计算

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

# 获取 SparseCore 硬件信息
sc_info = pltpu.get_tpu_info().sparse_core
num_cores = sc_info.num_cores
num_subcores = 1  # 简化：只用 1 个 vector subcore

# 创建 mesh
vector_mesh = plsc.VectorSubcoreMesh(
    core_axis_name="core",
    subcore_axis_name="subcore",
    num_cores=num_cores,
    num_subcores=num_subcores,
)
scalar_mesh = plsc.ScalarSubcoreMesh(
    axis_name="core",
    num_cores=num_cores,
)

x = jnp.arange(128, dtype=jnp.int32)

def vector_subcore_fn(x_hbm_ref, out_hbm_ref, shared_ref, tec_sem, scs_sem):
    """
    Vector Subcore 的工作：
    1. 等待 Scalar Subcore 通知数据已加载到 VMEM_SHARED
    2. 从 VMEM_SHARED 拷贝到本地 VMEM
    3. 计算
    4. 写回 HBM
    5. 通知 Scalar Subcore 工作完成
    """
    # 步骤 1：等待 Scalar Subcore 的信号（数据已就绪）
    # Scalar Subcore 会 signal 每个 vector subcore 一次
    pl.semaphore_wait(tec_sem, num_cores)  # 等待所有 scalar core 的信号

    # 步骤 2：从 VMEM_SHARED 拷贝到本地 VMEM
    local_ref = jax.empty_ref(jax.typeof(x), memory_space=pltpu.VMEM)
    pltpu.sync_copy(shared_ref, local_ref)

    # 步骤 3：在本地 VMEM 中计算（每个元素乘以 2）
    @pl.loop(0, x.size, step=sc_info.num_lanes)
    def _(i):
        s = pl.ds(i, sc_info.num_lanes)
        local_ref[s] = local_ref[s] * 2

    # 步骤 4：写回 HBM
    core_idx = jax.lax.axis_index("core")
    pltpu.sync_copy(local_ref, out_hbm_ref.at[core_idx])

    # 步骤 5：通知 Scalar Subcore 工作完成
    for i in range(num_cores):
        pl.semaphore_signal(scs_sem, device_id={"core": i})

def scalar_subcore_fn(x_hbm_ref, out_hbm_ref, shared_ref, tec_sem, scs_sem):
    """
    Scalar Subcore 的工作：
    1. 从 HBM 加载数据到 VMEM_SHARED
    2. 通知所有 Vector Subcore 数据已就绪
    3. 等待 Vector Subcore 完成计算
    """
    # 步骤 1：从 HBM 加载到 VMEM_SHARED
    pltpu.sync_copy(x_hbm_ref, shared_ref)

    # 步骤 2：通知所有 Vector Subcore（每个 core 的每个 subcore）
    for i in range(num_cores):
        for j in range(num_subcores):
            pl.semaphore_signal(tec_sem, device_id={"core": i, "subcore": j})

    # 步骤 3：等待 Vector Subcore 完成
    pl.semaphore_wait(scs_sem, num_cores * num_subcores)

# 调用 kernel
@jax.jit
def f(x):
    return pl.kernel(
        body=[vector_subcore_fn, scalar_subcore_fn],
        mesh=[vector_mesh, scalar_mesh],
        out_type=jax.ShapeDtypeStruct((num_cores, 128), x.dtype),
        scratch_types=[
            pltpu.VMEM_SHARED(x.shape, x.dtype),          # shared_ref
            pltpu.SemaphoreType.REGULAR(()) @ vector_mesh, # tec_sem（属于 Vector Subcore）
            pltpu.SemaphoreType.REGULAR(()) @ scalar_mesh, # scs_sem（属于 Scalar Subcore）
        ],
    )(x)

result = f(x)
# result[core_idx] == x * 2（每个 core 各自计算一份）
```

### 信号量分配的 `@ mesh` 语法

```python
pltpu.SemaphoreType.REGULAR(()) @ vector_mesh  # 信号量属于 Vector Subcore
pltpu.SemaphoreType.REGULAR(()) @ scalar_mesh  # 信号量属于 Scalar Subcore
```

`@ mesh` 表示信号量**物理分配在哪个 subcore 上**。这决定了：
- 谁可以 `wait` 这个信号量（只有拥有者可以 wait）
- 谁可以 `signal` 这个信号量（任何人都可以 signal 任何设备上的信号量）

### 执行时间线

```
Scalar Subcore:  [HBM→VMEM_SHARED] → signal(tec_sem) → wait(scs_sem) → done
                       ↓ 信号
Vector Subcore:  wait(tec_sem) → [VMEM_SHARED→VMEM] → [计算] → [VMEM→HBM] → signal(scs_sem)
```

两个 subcore 并行执行，通过信号量实现生产者-消费者模式。

### 简化版本：不需要 Scalar/Vector 协作时

如果不需要 Scalar Subcore 做额外工作（如索引计算），Vector Subcore 可以直接从 HBM 加载到 VMEM_SHARED 再到本地 VMEM，不需要信号量：

```python
def vector_subcore_fn_simple(x_hbm_ref, out_hbm_ref, shared_ref):
    # Vector Subcore 自己完成所有工作
    pltpu.sync_copy(x_hbm_ref, shared_ref)       # HBM → VMEM_SHARED
    local_ref = jax.empty_ref(jax.typeof(x), memory_space=pltpu.VMEM)
    pltpu.sync_copy(shared_ref, local_ref)        # VMEM_SHARED → VMEM

    @pl.loop(0, x.size, step=sc_info.num_lanes)
    def _(i):
        s = pl.ds(i, sc_info.num_lanes)
        local_ref[s] = local_ref[s] * 2

    pltpu.sync_copy(local_ref, out_hbm_ref)

def scalar_subcore_fn_simple(x_hbm_ref, out_hbm_ref, shared_ref):
    del x_hbm_ref, out_hbm_ref, shared_ref  # Scalar Subcore 不参与
    pass
```

这种模式下不需要信号量，但也没有利用 Scalar Subcore 的并行能力。

**关键点总结：**
- `pltpu.VMEM_SHARED(shape, dtype)` 分配一块所有 subcore 都能访问的共享内存
- 信号量通过 `@ mesh` 语法指定属于哪个 subcore 类型
- `pl.semaphore_signal(sem, device_id={...})` 中的 `device_id` 指定目标
- `pl.semaphore_wait(sem, value=N)` 阻塞直到信号量 >= N，然后减 N
- 信号量在 kernel 结束时必须为 0（所有 signal 和 wait 必须配对）

### DMA 路径总结

```
TensorCore 路径:
  HBM ←DMA→ TC VMEM ←load/store→ VREG (计算)

SparseCore 路径:
  HBM ←DMA→ VMEM_SHARED ←DMA→ SC VMEM (per-subcore) ←SIMD→ 计算
  HBM ←DMA→ SC VMEM (直接，跳过 VMEM_SHARED)
  HBM ←DMA→ SMEM (Scalar Subcore 标量访问)
```

在 SparseCore 上，`VMEM_SHARED` 充当了 HBM 和各 Vector Subcore 本地 VMEM 之间的中转站。但也可以直接从 HBM DMA 到 per-subcore VMEM（通过 `emit_pipeline` 的 `core_axis_name` 自动分配）。

## 自动 DMA vs 手动 DMA

Pallas 提供两种内存管理模式：

**自动模式**（BlockSpec 管理）：编译器根据 `BlockSpec` 的 `block_shape` 和 `index_map` 自动生成 DMA 代码。kernel 函数收到的 Ref 已经在 VMEM 中了。

**手动模式**（`memory_space=pltpu.HBM` + `make_async_copy` 或 `sync_copy`）：kernel 收到 HBM Ref，自己控制何时、搬运多少数据到 VMEM scratch buffer。

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
            pl.BlockSpec((8, 128), lambda i: (0, i)),
            pl.BlockSpec((8, 128), lambda i: (0, i)),
        ],
        out_specs=pl.BlockSpec((8, 128), lambda i: (0, i)),
        grid=(x.shape[1] // 128,),
    )(x, y)

# 使用
x = jnp.ones((8, 512), dtype=jnp.float32)
y = jnp.ones((8, 512), dtype=jnp.float32) * 2
result = auto_dma_add(x, y)
# result 的每个元素都是 3.0
# 编译器自动生成了 4 次 DMA（512 / 128 = 4 个块）
```

在这种模式下，你**完全不需要关心 DMA**。编译器会在每个 grid 步骤开始前自动将对应的块从 HBM 搬到 VMEM，计算完后自动将输出从 VMEM 搬回 HBM。

## 示例 2：sync_copy（最简单的手动 DMA）

`pltpu.sync_copy(src_ref, dst_ref)` 是最简单的手动 DMA 方式。它内部自动分配信号量，发起拷贝并等待完成：

```python
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def sync_copy_kernel(x_hbm_ref, o_hbm_ref):
    # 分配临时 VMEM 缓冲区
    @functools.partial(
        pl.run_scoped,
        buf=pltpu.VMEM((8, 128), jnp.float32),
    )
    def _(buf):
        # HBM → VMEM（同步，阻塞直到完成）
        pltpu.sync_copy(x_hbm_ref.at[:, pl.ds(0, 128)], buf)

        # 在 VMEM 中计算
        buf[...] = buf[...] * 2.0

        # VMEM → HBM（同步，阻塞直到完成）
        pltpu.sync_copy(buf, o_hbm_ref.at[:, pl.ds(0, 128)])

# 使用
x = jnp.arange(8 * 512, dtype=jnp.float32).reshape(8, 512)
result = pl.pallas_call(
    sync_copy_kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
    out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
)(x)
# result[:, :128] == x[:, :128] * 2.0
```

**关键点：**
- `pl.BlockSpec(memory_space=pltpu.HBM)` 告诉编译器"不要自动搬运，把 HBM 引用直接传给 kernel"
- `pltpu.sync_copy(src, dst)` 不需要手动管理信号量
- `pl.run_scoped` 在作用域内分配临时 VMEM 缓冲区，用完自动释放

## 示例 3：make_async_copy（异步 DMA）

当需要将 DMA 与计算重叠时，使用 `make_async_copy`。它需要显式的信号量：

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BLOCK_M, BLOCK_N = 8, 128

def async_copy_kernel(
    x_hbm_ref,      # HBM 中的输入
    o_hbm_ref,      # HBM 中的输出
    buf_ref,         # VMEM scratch buffer
    out_buf_ref,     # VMEM scratch buffer（输出用）
    load_sem_ref,    # DMA 信号量（加载）
    store_sem_ref,   # DMA 信号量（存储）
):
    # Step 1: 发起 HBM → VMEM 的异步拷贝
    load_copy = pltpu.make_async_copy(
        src_ref=x_hbm_ref.at[:, pl.ds(0, BLOCK_N)],  # HBM 中的第一个块
        dst_ref=buf_ref,                               # VMEM scratch
        sem=load_sem_ref,                              # 绑定信号量
    )
    load_copy.start()  # 非阻塞：DMA 引擎开始搬运

    # Step 2: 等待 DMA 完成
    load_copy.wait()   # 阻塞：直到数据到达 VMEM

    # Step 3: 在 VMEM 中计算
    out_buf_ref[...] = buf_ref[...] * 2.0

    # Step 4: 将结果从 VMEM 写回 HBM
    store_copy = pltpu.make_async_copy(
        src_ref=out_buf_ref,
        dst_ref=o_hbm_ref.at[:, pl.ds(0, BLOCK_N)],
        sem=store_sem_ref,
    )
    store_copy.start()
    store_copy.wait()

# 使用
x = jnp.arange(8 * 512, dtype=jnp.float32).reshape(8, 512)
result = pl.pallas_call(
    async_copy_kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
    out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
    scratch_shapes=[
        pltpu.VMEM((BLOCK_M, BLOCK_N), jnp.float32),  # buf_ref
        pltpu.VMEM((BLOCK_M, BLOCK_N), jnp.float32),  # out_buf_ref
        pltpu.SemaphoreType.DMA(()),                   # load_sem_ref
        pltpu.SemaphoreType.DMA(()),                   # store_sem_ref
    ],
    grid=(1,),
)(x)
# result[:, :128] == x[:, :128] * 2.0
```

**`sync_copy` vs `make_async_copy` 的选择：**
- `sync_copy`：简单场景，不需要重叠 DMA 和计算
- `make_async_copy`：需要重叠 DMA 和计算（双缓冲），或需要对多个不连续的页发起 DMA

## 示例 4：循环 + sync_copy（处理整个数组）

```python
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BLOCK_N = 128

def loop_sync_copy_kernel(x_hbm_ref, o_hbm_ref):
    num_blocks = x_hbm_ref.shape[1] // BLOCK_N

    @functools.partial(
        pl.run_scoped,
        buf=pltpu.VMEM((8, BLOCK_N), jnp.float32),
    )
    def _(buf):
        @pl.loop(0, num_blocks)
        def _(i):
            # 加载第 i 个块
            pltpu.sync_copy(x_hbm_ref.at[:, pl.ds(i * BLOCK_N, BLOCK_N)], buf)
            # 计算
            buf[...] = buf[...] * 2.0
            # 写回第 i 个块
            pltpu.sync_copy(buf, o_hbm_ref.at[:, pl.ds(i * BLOCK_N, BLOCK_N)])

# 使用
x = jnp.arange(8 * 512, dtype=jnp.float32).reshape(8, 512)
result = pl.pallas_call(
    loop_sync_copy_kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
    out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
)(x)
# result == x * 2.0
```

注意：这个版本**没有**重叠 DMA 和计算。每次都是 load → wait → compute → store → wait，效率不高。下一个示例展示如何用双缓冲优化。

## 示例 5：双缓冲（DMA 与计算重叠）

双缓冲的核心思想：使用两块 VMEM 缓冲区交替工作。当计算单元处理缓冲区 0 中的数据时，DMA 引擎同时将下一批数据加载到缓冲区 1。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BLOCK_M, BLOCK_N = 8, 128

def double_buffer_kernel(
    x_hbm_ref,
    o_hbm_ref,
    buf_ref,          # shape: (2, BLOCK_M, BLOCK_N) — 两个缓冲区
    out_buf_ref,      # shape: (2, BLOCK_M, BLOCK_N)
    load_sem_ref,     # shape: (2,) — 两个信号量
    store_sem_ref,    # shape: (2,)
):
    num_blocks = x_hbm_ref.shape[1] // BLOCK_N

    # Prologue: 预取第一个块到 buf[0]
    pltpu.make_async_copy(
        src_ref=x_hbm_ref.at[:, pl.ds(0, BLOCK_N)],
        dst_ref=buf_ref.at[0],
        sem=load_sem_ref.at[0],
    ).start()

    @pl.loop(0, num_blocks)
    def _(i):
        cur = i % 2   # 当前使用的缓冲区
        nxt = 1 - cur  # 下一个缓冲区

        # 等待当前块的 DMA 完成
        pltpu.make_async_copy(
            src_ref=x_hbm_ref.at[:, pl.ds(i * BLOCK_N, BLOCK_N)],
            dst_ref=buf_ref.at[cur],
            sem=load_sem_ref.at[cur],
        ).wait()

        # 启动下一个块的 DMA（与计算重叠）
        @pl.when(i + 1 < num_blocks)
        def _():
            pltpu.make_async_copy(
                src_ref=x_hbm_ref.at[:, pl.ds((i + 1) * BLOCK_N, BLOCK_N)],
                dst_ref=buf_ref.at[nxt],
                sem=load_sem_ref.at[nxt],
            ).start()

        # 计算（此时 DMA 引擎在后台搬运下一个块）
        out_buf_ref.at[cur][...] = buf_ref.at[cur][...] * 2.0

        # 将结果写回 HBM
        pltpu.make_async_copy(
            src_ref=out_buf_ref.at[cur],
            dst_ref=o_hbm_ref.at[:, pl.ds(i * BLOCK_N, BLOCK_N)],
            sem=store_sem_ref.at[cur],
        ).start()

        # 等待上一次 store 完成（确保 buf 可以被覆盖）
        @pl.when(i > 0)
        def _():
            pltpu.make_async_copy(
                src_ref=out_buf_ref.at[nxt],
                dst_ref=o_hbm_ref.at[:, pl.ds((i - 1) * BLOCK_N, BLOCK_N)],
                sem=store_sem_ref.at[nxt],
            ).wait()

# 使用
x = jnp.arange(8 * 512, dtype=jnp.float32).reshape(8, 512)
result = pl.pallas_call(
    double_buffer_kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
    out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
    scratch_shapes=[
        pltpu.VMEM((2, BLOCK_M, BLOCK_N), jnp.float32),  # buf_ref (双缓冲)
        pltpu.VMEM((2, BLOCK_M, BLOCK_N), jnp.float32),  # out_buf_ref (双缓冲)
        pltpu.SemaphoreType.DMA((2,)),                    # load_sem_ref
        pltpu.SemaphoreType.DMA((2,)),                    # store_sem_ref
    ],
    grid=(1,),
)(x)
# result == x * 2.0
```

**时间线：**
```
时间 →
DMA:    [load B0] [load B1] [load B2] [load B3]
计算:            [comp B0] [comp B1] [comp B2] [comp B3]
Store:           [stor B0] [stor B1] [stor B2] [stor B3]
```

DMA 和计算完全重叠后，总时间 ≈ max(DMA 时间, 计算时间) × N，而不是 (DMA + 计算) × N。

## 示例 6：Scratch Buffers（跨迭代累加）

Scratch buffer 是 kernel 内部使用的临时内存，不对应任何输入或输出。通过 `scratch_shapes` 参数分配。

Scratch buffer 的生命周期是**整个 Grid 执行期间**。Grid 从 `i=0` 推进到 `i=1` 时，scratch 中的数据会被保留。这是 TPU 顺序执行模型的优势——可以跨迭代累加，无需原子操作。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def sum_kernel(x_ref, o_ref, acc_ref):
    # acc_ref 是 scratch buffer，跨迭代保持状态
    acc_ref[...] = acc_ref[...] + x_ref[...]
    # 每次迭代都把当前累加结果写入输出
    o_ref[...] = acc_ref[...]

# 使用
x = jnp.arange(8 * 512, dtype=jnp.float32).reshape(8, 512)
result = pl.pallas_call(
    sum_kernel,
    out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    in_specs=[pl.BlockSpec((8, 128), lambda i: (0, i))],
    out_specs=pl.BlockSpec((8, 128), lambda _: (0, 0)),
    scratch_shapes=[
        pltpu.VMEM((8, 128), jnp.float32),  # acc_ref
    ],
    grid=(4,),  # 512 / 128 = 4 个块
)(x)
# result == x[:, 0:128] + x[:, 128:256] + x[:, 256:384] + x[:, 384:512]
```

## 示例 7：emit_pipeline（推荐的流水线方式）

对于规则的分块计算，`pltpu.emit_pipeline` 是推荐的方式。它自动处理双缓冲和 DMA 调度：

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def emit_pipeline_kernel(x_hbm_ref, o_hbm_ref):
    def body(x_ref, o_ref):
        # x_ref 已经在 VMEM 中了（emit_pipeline 自动管理）
        o_ref[...] = x_ref[...] * 2.0

    pltpu.emit_pipeline(
        body,
        grid=(4,),
        in_specs=[pl.BlockSpec((8, 128), lambda i: (0, i))],
        out_specs=[pl.BlockSpec((8, 128), lambda i: (0, i))],
    )(x_hbm_ref, o_hbm_ref)

# 使用
x = jnp.arange(8 * 512, dtype=jnp.float32).reshape(8, 512)
result = pl.pallas_call(
    emit_pipeline_kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
    out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
)(x)
# result == x * 2.0
```

`emit_pipeline` 与手动双缓冲的关系：
- `emit_pipeline` = 自动双缓冲 + 自动 DMA 调度
- 当访问模式规则时，用 `emit_pipeline`
- 当需要动态索引（如 paged attention）时，必须手动管理 DMA

## 示例 8：emit_pipeline 中混合 HBM 和自动缓冲的输入

`emit_pipeline` 支持部分输入自动缓冲、部分输入保留在 HBM 中：

```python
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def mixed_pipeline_kernel(x_hbm_ref, y_hbm_ref, o_hbm_ref):
    def body(x_ref, y_hbm_ref_inner, o_ref):
        # x_ref: 自动缓冲到 VMEM
        # y_hbm_ref_inner: 仍在 HBM，需要手动 sync_copy
        @functools.partial(
            pl.run_scoped,
            y_buf=pltpu.VMEM((8, 128), jnp.float32),
        )
        def _(y_buf):
            pltpu.sync_copy(y_hbm_ref_inner, y_buf)
            o_ref[...] = x_ref[...] + y_buf[...]

    pltpu.emit_pipeline(
        body,
        grid=(4,),
        in_specs=[
            pl.BlockSpec((8, 128), lambda i: (0, i)),                          # 自动缓冲
            pl.BlockSpec((8, 128), lambda i: (0, i), memory_space=pltpu.HBM),  # 保留在 HBM
        ],
        out_specs=[pl.BlockSpec((8, 128), lambda i: (0, i))],
    )(x_hbm_ref, y_hbm_ref, o_hbm_ref)

# 使用
x = jnp.ones((8, 512), dtype=jnp.float32)
y = jnp.ones((8, 512), dtype=jnp.float32) * 2
result = pl.pallas_call(
    mixed_pipeline_kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    in_specs=[
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(memory_space=pltpu.HBM),
    ],
    out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
)(x, y)
# result 的每个元素都是 3.0
```

## 示例 9：多页 DMA（Paged Attention 模式）

在 paged attention 中，需要从不连续的物理页加载数据。对同一个信号量发起多次 `make_async_copy`，然后逐个 wait：

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

PAGE_SIZE = 16
NUM_PAGES = 4

def paged_load_kernel(
    cache_hbm_ref,    # [total_pages, PAGE_SIZE, 128] in HBM
    o_hbm_ref,        # [NUM_PAGES * PAGE_SIZE, 128] in HBM
    page_indices_ref, # SMEM: [NUM_PAGES] — 物理页号
    buf_ref,          # VMEM: [NUM_PAGES, PAGE_SIZE, 128]
    sem_ref,          # DMA 信号量
):
    # 对同一个信号量发起多次 DMA（页物理不连续）
    copies = []
    for p in range(NUM_PAGES):
        page_idx = page_indices_ref[p]  # 从 SMEM 读取物理页号
        copy = pltpu.make_async_copy(
            src_ref=cache_hbm_ref.at[page_idx],       # HBM 中的物理页
            dst_ref=buf_ref.at[p],                     # VMEM 中的目标位置
            sem=sem_ref,
        )
        copies.append(copy)
        copy.start()

    # 等待所有 DMA 完成
    for copy in copies:
        copy.wait()

    # 计算（在 VMEM 中）
    # 将结果写回 HBM
    pltpu.sync_copy(buf_ref, o_hbm_ref)
```

这就是 Ragged Paged Attention kernel 中 `MultiPageAsyncCopyDescriptor` 的核心模式。

## 示例 10：pl.ds（动态切片）

`pl.ds(start, size)` 用于在 Ref 上做动态切片。`start` 可以是运行时变量：

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def dynamic_slice_kernel(x_hbm_ref, o_hbm_ref):
    i = pl.program_id(0)

    @functools.partial(
        pl.run_scoped,
        buf=pltpu.VMEM((8, 128), jnp.float32),
    )
    def _(buf):
        # pl.ds(start, size): start 可以是动态的（运行时确定）
        pltpu.sync_copy(x_hbm_ref.at[:, pl.ds(i * 128, 128)], buf)
        buf[...] = buf[...] + 1.0
        pltpu.sync_copy(buf, o_hbm_ref.at[:, pl.ds(i * 128, 128)])

# 使用
x = jnp.zeros((8, 512), dtype=jnp.float32)
result = pl.pallas_call(
    dynamic_slice_kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
    out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
    grid=(4,),
)(x)
# result 的每个元素都是 1.0
```

对比：
- `ref.at[:, 0:128]`：静态切片，编译时确定
- `ref.at[:, pl.ds(i * 128, 128)]`：动态切片，运行时确定

## 示例 11：run_scoped（动态临时内存）

`pl.run_scoped` 在 kernel 内部动态分配临时 Ref，用完自动释放：

```python
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def scoped_kernel(x_hbm_ref, o_hbm_ref):
    # 分配临时 VMEM 和 DMA 信号量
    @functools.partial(
        pl.run_scoped,
        x_vmem=pltpu.VMEM((8, 512), jnp.float32),
        o_vmem=pltpu.VMEM((8, 512), jnp.float32),
    )
    def _(x_vmem, o_vmem):
        # 整体拷贝 HBM → VMEM
        pltpu.sync_copy(x_hbm_ref, x_vmem)

        # 在 VMEM 中计算
        o_vmem[...] = x_vmem[...] * 3.0

        # 整体拷贝 VMEM → HBM
        pltpu.sync_copy(o_vmem, o_hbm_ref)

# 使用
x = jnp.ones((8, 512), dtype=jnp.float32)
result = pl.pallas_call(
    scoped_kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
    out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
)(x)
# result 的每个元素都是 3.0
```

与 scratch_shapes 的区别：
- `scratch_shapes`：在整个 kernel 生命周期内存在，适合跨迭代状态
- `run_scoped`：只在作用域内存在，适合循环内部的临时缓冲区（减少 VMEM 占用）

## 示例 12：with_memory_space_constraint

在 `pallas_call` 外部，确保张量驻留在指定的内存空间：

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# 在调用 pallas_call 之前，确保 q 在 HBM 中
# 防止编译器将大张量意外拷贝到 VMEM
q = jnp.ones((1024, 128), dtype=jnp.bfloat16)
q_hbm = pltpu.with_memory_space_constraint(q, pltpu.HBM)

# 然后传入 pallas_call
# result = pl.pallas_call(kernel, ...)(q_hbm, k, v)
```

这在 RPA v3 kernel 中用于确保大的 KV cache 张量不被意外拷贝。

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

信号量的工作方式：
- DMA 完成时，硬件自动对绑定的信号量做 signal（+1）
- `.wait()` 阻塞直到信号量 >= 1，然后 -1
- 对同一个信号量发起 N 次 DMA，信号量会被递增 N 次

## 内存空间选择决策树

```
TensorCore kernel:
  需要计算？
  ├── 是 → VMEM（必须）
  └── 否 → 需要标量随机访问？
            ├── 是 → SMEM
            └── 否 → 数据量大？
                      ├── 是 → HBM（手动 DMA 管理）
                      └── 否 → VMEM scratch

SparseCore kernel:
  需要跨 subcore 共享？
  ├── 是 → VMEM_SHARED
  └── 否 → 需要标量操作？
            ├── 是 → SMEM (Scalar Subcore)
            └── 否 → VMEM (per-subcore 本地)
```

## 与 GPU 的对比

| 概念 | GPU (CUDA) | TPU (Pallas) |
| :--- | :--- | :--- |
| 主存 | Global Memory | HBM |
| 片上高速内存 | Shared Memory (~160KB) | VMEM (16MB+) |
| 数据搬运 | 线程直接 load/store | DMA 引擎异步搬运 |
| 同步机制 | `__syncthreads()` | 信号量 (Semaphore) |
| 动态索引 | 线程可以访问任意地址 | 需要 SMEM + 手动 DMA |
| 缓冲管理 | 程序员手动管理 shared memory | BlockSpec 自动 / emit_pipeline / 手动 DMA |
| 流水线 | 硬件 warp 调度隐藏延迟 | 软件双缓冲显式重叠 |

GPU 的优势：编程模型简单（线程直接访问全局内存）。
TPU 的优势：VMEM 容量大 100 倍，DMA 引擎可以完全隐藏内存延迟；SparseCore 专门处理随机访问。
