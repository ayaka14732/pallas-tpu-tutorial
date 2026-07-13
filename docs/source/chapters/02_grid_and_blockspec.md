# 第 2 章：Grid 与 BlockSpec

本章讨论 Pallas TPU kernel 最核心的两个声明：

- `grid`：kernel body 会被执行多少次，每次执行的坐标是什么。
- `BlockSpec`：每次执行时，每个输入/输出 Ref 对应原数组的哪一块，以及这块数据应放在哪个内存空间。

## 循环模型

`pallas_call` 的 `grid` 参数定义了一个多维循环。对于 `grid=(M, N)`，kernel body 会被执行 `M * N` 次。每一次执行称为一个 **program**，kernel 内部可以用 `pl.program_id(axis)` 读到当前 program 在某个 grid 轴上的坐标。

```python
pl.pallas_call(
    kernel_fn,
    out_shape=...,
    grid=(M, N),
    in_specs=[block_spec_x],
    out_specs=block_spec_o,
)(x)
```

概念上等价于：

```python
for i in range(M):
    for j in range(N):
        # program_id(0) == i
        # program_id(1) == j
        x_block = x[block_spec_x.compute_slice(i, j)]
        o_block = o[block_spec_o.compute_slice(i, j)]
        kernel_fn(x_block_ref, o_block_ref)
```

默认 `grid=()` 表示只执行一次。

## BlockSpec 的结构

`BlockSpec` 回答的问题是：当前 program 应该拿到原数组的哪一块？

```python
pl.BlockSpec(
    block_shape=(BM, BN),
    index_map=lambda i, j: (i, j),
    memory_space=pltpu.VMEM,
)
```

它最重要的组成部分有四个：

| 参数 | 含义 |
| :--- | :--- |
| `block_shape` | kernel 每次看到的块形状 |
| `index_map` | grid 坐标 -> 原数组块坐标 |
| `memory_space` | Ref 驻留的内存空间 |
| `pipeline_mode` | 自动流水线缓冲策略，常见于 `pl.Buffered(...)` |

`in_specs` 和 `out_specs` 分别描述输入和输出。它们需要和输入/输出的 pytree 结构对应。

```python
in_specs = [
    pl.BlockSpec((BM, BK), lambda i, j, k: (i, k)),  # A
    pl.BlockSpec((BK, BN), lambda i, j, k: (k, j)),  # B
]
out_specs = pl.BlockSpec((BM, BN), lambda i, j, k: (i, j))
```

### block index，不是元素偏移

默认 `BlockSpec` 使用 blocked indexing。`index_map` 返回的是块索引，不是元素起始下标。

```python
spec = pl.BlockSpec(
    block_shape=(128, 256),
    index_map=lambda i, j: (i, j),
)
```

如果原数组形状是 `(1024, 512)`，某次 program 的 grid 坐标是 `(2, 1)`，那么：

```python
block_index = (2, 1)
element_slice = (slice(2 * 128, 3 * 128),
                 slice(1 * 256, 2 * 256))
# 即 [256:384, 256:512]
```

这条规则很重要。写 `lambda i: (i * 128,)` 通常是错的，因为那会跳到第 `i * 128` 个块，而不是第 `i * 128` 个元素。

### 默认值

`pl.BlockSpec()` 等价于“整个数组作为一个块，默认 index_map 返回全 0”：

```python
pl.BlockSpec()

# 概念上类似：
pl.BlockSpec(
    block_shape=None,  # 使用整个数组形状
    index_map=None,    # lambda *ids: (0, 0, ...)
)
```

这适合小张量或手动 DMA 的入口参数。对于大张量，不要轻易把整个数组搬到 VMEM。

## block shape 的选择

TPU 的 VMEM/VREG 按 tile 工作，block shape 不是任意形状都高效，也不是任意形状都被后端支持。

官方规则可以浓缩为：

- TPU 上 block rank 至少为 1。
- rank >= 2 时，最后两个维度通常应当分别是 `8` 和 `128` 的倍数，或者等于原数组对应维度。
- rank == 1 时，维度应当等于原数组维度，或者满足 TPU 后端支持的向量长度约束。
- 32-bit 类型的自然 tile 常按 `(8, 128)` 理解；低精度类型在 sublane 方向可容纳更多元素，但最后一维仍应围绕 128 lanes 设计。

实践里最稳妥的选择：

```python
# 2D elementwise / row block
block_shape = (8, 128)
block_shape = (16, 128)
block_shape = (128, 128)
block_shape = (256, 128)

# matmul
A_block = (BM, BK)
B_block = (BK, BN)
C_block = (BM, BN)
```

如果需要在 VMEM 中存放标量或很短的一维数组，通常应该改用 SMEM，或者把它扩展成 tile 友好的形状。VMEM 中的 `(1,)`、`(1, 1)` 往往会浪费整个 tile，性能和容量都不划算。

## 越界处理

当最后一个块越过数组边界时，Pallas 仍会把一个 `block_shape` 大小的 Ref 交给 kernel。越界部分的行为要分读和写来看：

- 输入读取：越界元素会被 padding，但 padding 值不保证；解释模式下浮点 padding 可能是 NaN，用来暴露错误访问。
- 输出写入：越界写入会被丢弃。
- 每个块至少要有一个元素在数组真实范围内。

因此，`grid=(pl.cdiv(n, block_size),)` 是常见写法，但不意味着 padding 的值就是 0。

对于纯 elementwise 写回，通常没有问题：

```python
def add_kernel(x_ref, y_ref, o_ref):
    # 最后一个块里，真实范围内的元素有效；越界写回会被丢弃
    o_ref[...] = x_ref[...] + y_ref[...]
```

对于 reduction 或 softmax 这类会把整个块参与计算的 kernel，必须显式 mask 掉越界元素。一个常见做法是传入真实长度，或者在 metadata 中保存边界。

```python
def sum_last_block_kernel(x_ref, n_ref, o_ref):
    pid = pl.program_id(0)
    offsets = pid * BLOCK + jnp.arange(BLOCK)
    mask = offsets < n_ref[0]
    vals = jnp.where(mask, x_ref[...], 0.0)
    o_ref[...] = jnp.sum(vals)
```

如果不 mask，最后一个块中的 padding 值可能污染结果。

## 常见 index_map 模式

### 1. 逐块遍历

最直接的模式：grid 坐标和数组块坐标一一对应。

```python
block = pl.BlockSpec((BM, BN), lambda i, j: (i, j))

grid = (
    pl.cdiv(M, BM),
    pl.cdiv(N, BN),
)
```

适合矩阵加法、逐元素变换、拷贝、简单 layout 转换。

### 2. 广播

某个输入不依赖部分 grid 维度，就会在这些维度上被广播复用。

```python
# x:    [M, N]
# bias: [N]
# grid=(M_blocks, N_blocks)
x_spec = pl.BlockSpec((BM, BN), lambda i, j: (i, j))
b_spec = pl.BlockSpec((BN,), lambda i, j: (j,))
o_spec = pl.BlockSpec((BM, BN), lambda i, j: (i, j))
```

`bias` 不依赖 `i`，所以同一个列块会被所有行块复用。

### 3. 归约/累加后缀

输出块不依赖最后一个 grid 维度，最后一维用于累加。

```python
grid = (
    pl.cdiv(M, BM),
    pl.cdiv(N, BN),
    pl.cdiv(K, BK),
)

a_spec = pl.BlockSpec((BM, BK), lambda i, j, k: (i, k))
b_spec = pl.BlockSpec((BK, BN), lambda i, j, k: (k, j))
c_spec = pl.BlockSpec((BM, BN), lambda i, j, k: (i, j))
```

kernel 中通常用 `pl.when(pl.program_id(2) == 0)` 初始化累加器：

```python
def matmul_kernel(a_ref, b_ref, c_ref):
    @pl.when(pl.program_id(2) == 0)
    def _():
        c_ref[...] = jnp.zeros_like(c_ref)

    c_ref[...] += a_ref[...] @ b_ref[...]
```

### 4. 重复读取同一输入块

输入可以忽略某些 grid 维度，从而复用同一块数据。

```python
# 每个 batch/head 都读取同一个表
table_spec = pl.BlockSpec((ROWS, D), lambda batch, head, row_block: (row_block, 0))
```

在 TPU 上，如果相邻 program 使用同一输入窗口，编译器有机会保留 VMEM 中的数据，减少重复 DMA。

### 5. 对角线或偏移访问

`index_map` 可以做简单整数表达式。

```python
# 读取 x 的第 i 个块，写到 o 的第 i+1 个块
x_spec = pl.BlockSpec((BLOCK,), lambda i: (i,))
o_spec = pl.BlockSpec((BLOCK,), lambda i: (i + 1,))
```

要保证 `i + 1` 不会让整个块完全越界；否则 block mapping 无效。

### 6. batch/head 布局

注意力类 kernel 常见输入形状是 `[batch, heads, seq, dim]` 或 `[batch, seq, heads, dim]`。`index_map` 是把逻辑工作分配到物理布局的地方。

```python
# q: [batch, heads, seq, dim]
# 每个 program 处理一个 (batch, head, q_block)
q_spec = pl.BlockSpec(
    (None, None, BQ, D),
    lambda b, h, q_blk: (b, h, q_blk, 0),
)
```

这里前两个维度使用 `None`/`Squeezed`，kernel 内部只看到 `(BQ, D)`，不需要携带大小为 1 的 batch/head 维度。

## Squeezed 维度

当 `block_shape` 中某个维度为 `None` 或 `pl.Squeezed()` 时，该维度的 block size 视为 1，但传入 kernel 的 Ref 会去掉这个维度。

```python
# 输入形状: [batch, seq_len, hidden]
# 每个 program 处理一个 batch，kernel 只看到 [seq_len, hidden]
batch_spec = pl.BlockSpec(
    block_shape=(None, seq_len, hidden),
    index_map=lambda b: (b, 0, 0),
)

def kernel(x_ref, o_ref):
    assert x_ref.shape == (seq_len, hidden)
    o_ref[...] = x_ref[...] * 2
```

等价写法：

```python
pl.BlockSpec(
    block_shape=(pl.Squeezed(), seq_len, hidden),
    index_map=lambda b: (b, 0, 0),
)
```

Squeezed 维度很适合 batch/head 这种“用于选择，不用于计算”的轴。它也能避免在最后两个维度里出现低效的 singleton tile。

## 其他索引模式

### pl.Element

`pl.Element(size)` 表示 `index_map` 返回元素起始下标，而不是块编号。

```python
spec = pl.BlockSpec(
    block_shape=(pl.Element(BM), pl.Element(BN)),
    index_map=lambda i, j: (i * BM, j * BN),
)
```

这和默认 blocked 模式得到的切片可能一样，但语义不同：

```python
# blocked：返回块编号
pl.BlockSpec((BM, BN), lambda i, j: (i, j))

# element：返回元素编号
pl.BlockSpec((pl.Element(BM), pl.Element(BN)), lambda i, j: (i * BM, j * BN))
```

`pl.Element` 主要用于需要元素级起点、padding 或特殊布局的场景。普通规则分块优先用默认 blocked 模式。

### pl.BoundedSlice

`pl.BoundedSlice(max_size)` 表示该维度的实际 slice 大小可以动态变化，但不会超过 `max_size`。这常用于 ragged 或动态长度块。

对应的 `index_map` 必须返回 `pl.ds(start, size)`，这里的 `start` 和 `size` 是元素单位，不是块单位。

```python
spec = pl.BlockSpec(
    block_shape=(pl.BoundedSlice(32), 128),
    index_map=lambda i, starts_ref, ends_ref: (
        pl.ds(starts_ref[i], ends_ref[i] - starts_ref[i]),
        0,
    ),
)
```

这个模式经常和 `PrefetchScalarGridSpec` 或 `emit_pipeline` 一起使用：metadata 放在 SMEM，`index_map` 根据 metadata 产生动态但有界的切片。

### pl.Indirect

`pl.Indirect(size)` 表示该维度由一组索引间接访问。输入上对应 gather，输出上对应 scatter。它主要出现在 SparseCore 或稀疏访问 kernel 中。

```python
# indices_ref[i] 给出这一轮要 gather 的若干行
x_spec = pl.BlockSpec(
    block_shape=(pl.Indirect(NUM_LANES), NUM_LANES),
    index_map=lambda i, indices_ref: (indices_ref.at[i], 0),
)
```

如果只是根据一个标量页号选择一个连续块，通常用 `PrefetchScalarGridSpec` 的动态 `index_map` 就够了；如果一次要按向量索引 gather/scatter，才考虑 `pl.Indirect`。

## no_block_spec 与整体传入

`pl.no_block_spec` 表示该输入/输出不参与自动分块。它会按整体 Ref 传入 kernel。

```python
in_specs = [
    pl.BlockSpec((128,), lambda i: (i,)),  # x 按块切分
    pl.no_block_spec,                      # metadata 整体传入
]
```

不过在 TPU 上，更常见的“整体传入”写法是显式指定内存空间：

```python
pl.BlockSpec(memory_space=pltpu.HBM)   # 整体 HBM Ref，手动 DMA
pl.BlockSpec(memory_space=pltpu.VMEM)  # 整体 VMEM Ref，小张量才适合
pl.BlockSpec(memory_space=pltpu.SMEM)  # 整体 SMEM Ref，适合标量/索引
```

当你写 `pl.BlockSpec(memory_space=pltpu.HBM)` 且不指定 `block_shape/index_map` 时，kernel 收到的是整个数组的 HBM Ref。它不会被自动搬进 VMEM，需要用 `pltpu.sync_copy` 或 `pltpu.make_async_copy` 手动搬运，详见第 4 章。

## GridSpec

`pallas_call` 有两种传 grid 和 specs 的方式：

```python
# 简写
pl.pallas_call(
    kernel,
    out_shape=...,
    grid=(M, N),
    in_specs=[...],
    out_specs=...,
)(...)
```

```python
# 显式 GridSpec
grid_spec = pl.GridSpec(
    grid=(M, N),
    in_specs=[...],
    out_specs=...,
    scratch_shapes=[...],
)

pl.pallas_call(
    kernel,
    out_shape=...,
    grid_spec=grid_spec,
)(...)
```

当只需要普通分块时，用简写即可。当需要 scalar prefetch、复杂 scratch 配置，或想把 grid/specs 作为一个对象传递时，用 `GridSpec` 更清楚。

### PrefetchScalarGridSpec

`pltpu.PrefetchScalarGridSpec` 是 TPU 特有扩展，用于让 `index_map` 依赖运行时 metadata。

```python
grid_spec = pltpu.PrefetchScalarGridSpec(
    num_scalar_prefetch=1,
    grid=(num_blocks,),
    in_specs=[
        pl.BlockSpec(
            (None, BLOCK, D),
            lambda i, page_table_ref: (page_table_ref[i], 0, 0),
        ),
    ],
    out_specs=pl.BlockSpec((None, BLOCK, D), lambda i, page_table_ref: (i, 0, 0)),
)
```

关键语义：

- `num_scalar_prefetch=1` 表示 `pallas_call` 的第一个参数会被预取到 SMEM。
- scalar prefetch 参数不需要在 `in_specs` 中写对应 spec。
- 后续普通输入/输出的 `index_map` 会额外收到这些 SMEM Ref。
- 适合 page table、ragged length、block-sparse 索引等动态访问。

Kernel 签名中 scalar prefetch 参数仍然排在最前面：

```python
def kernel(page_table_ref, x_ref, o_ref):
    # page_table_ref: SMEM
    # x_ref: 根据 page_table_ref[i] 自动搬到 VMEM 的数据块
    o_ref[...] = x_ref[...]
```

如果动态索引发生在 kernel 内部循环里，而不是每个 grid program 的入口处，通常需要改用手动 DMA。

## BlockSpec 的 memory_space

`memory_space` 控制 kernel 收到的 Ref 指向哪里。

```python
from jax.experimental.pallas import tpu as pltpu

pl.BlockSpec((BM, BN), lambda i, j: (i, j), memory_space=pltpu.VMEM)
pl.BlockSpec(memory_space=pltpu.HBM)
pl.BlockSpec(memory_space=pltpu.SMEM)
```

常见选择：

| memory_space | kernel 收到什么 | 典型用途 |
| :--- | :--- | :--- |
| 默认 / `pltpu.VMEM` | 已在 VMEM 的块 Ref | 自动 DMA、普通计算 |
| `pltpu.HBM` | HBM 中的整体或切片 Ref | 手动 DMA、复杂流水线 |
| `pltpu.SMEM` | 标量内存 Ref | metadata、索引、控制流 |

自动模式下，编译器根据 `BlockSpec` 在每个 program 前后生成 DMA：

```text
HBM input block -> VMEM input Ref -> kernel compute -> VMEM output Ref -> HBM output block
```

手动模式下，`BlockSpec(memory_space=pltpu.HBM)` 只把 HBM Ref 交给你：

```python
def kernel(x_hbm_ref, o_hbm_ref, buf_ref, sem_ref):
    copy = pltpu.make_async_copy(
        x_hbm_ref.at[:, pl.ds(0, 128)],
        buf_ref,
        sem_ref,
    )
    copy.start()
    copy.wait()
    ...
```

这正是第 4、5 章手动 DMA 和流水线的基础。

## pipeline_mode

`BlockSpec` 还可以指定 `pipeline_mode`，控制自动流水线中该 operand 使用多少缓冲。

```python
spec = pl.BlockSpec(
    (BM, BN),
    lambda i, j: (i, j),
    pipeline_mode=pl.Buffered(2),
)
```

常见用途：

- `pl.Buffered(1)`：强制单缓冲，减少 VMEM 占用。
- `pl.Buffered(2)`：双缓冲，尝试重叠 DMA 和计算。
- `pl.Buffered(buffer_count=2, use_lookahead=True)`：允许 lookahead prefetch。

通常不需要一开始就手动设置它。只有遇到 VMEM OOM、自动缓冲不符合预期，或在写高性能流水线时才调整。

## Scratch Shapes

Scratch buffer 是 kernel 内部临时内存，不对应任何输入或输出。

```python
scratch_shapes = [
    pltpu.VMEM((8, 128), jnp.float32),
    pltpu.SMEM((16,), jnp.int32),
    pltpu.SemaphoreType.DMA((2,)),
]
```

Scratch Ref 会作为额外参数传入 kernel，顺序在输入和输出 Ref 之后：

```python
def kernel(x_ref, y_ref, o_ref, tmp_ref, idx_ref, dma_sem_ref):
    ...
```

它和 `BlockSpec` 的关系是：

- `BlockSpec` 管理来自输入/输出数组的窗口。
- `scratch_shapes` 分配 kernel 自己使用的临时空间。
- Scratch 生命周期覆盖整个 `pallas_call` grid 执行，因此可以保存跨 program 的状态。

如果临时空间只在某个局部作用域使用，可以考虑第 4 章的 `pl.run_scoped`。

## 示例 1：1D 向量加法

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

BLOCK = 1024

def add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]

def vector_add(x, y):
    spec = pl.BlockSpec(
        block_shape=(BLOCK,),
        index_map=lambda i: (i,),
    )
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[spec, spec],
        out_specs=spec,
        grid=(pl.cdiv(x.shape[0], BLOCK),),
    )(x, y)
```

这里 `grid` 的第 `i` 个 program 处理 `[i * BLOCK : (i + 1) * BLOCK]`。最后一个块如果越界，越界写入会被丢弃。

## 示例 2：2D 矩阵加法

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BM, BN = 8, 128

def matadd_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]

def matrix_add(x, y):
    m, n = x.shape

    block = pl.BlockSpec(
        block_shape=(BM, BN),
        index_map=lambda i, j: (i, j),
    )

    return pl.pallas_call(
        matadd_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[block, block],
        out_specs=block,
        grid=(pl.cdiv(m, BM), pl.cdiv(n, BN)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL),
        ),
    )(x, y)
```

这个例子中每个输出块只写一次，两个 grid 维度都可并行。

## 示例 3：bias 广播

```python
BM, BN = 8, 128

def add_bias_kernel(x_ref, b_ref, o_ref):
    o_ref[...] = x_ref[...] + b_ref[None, :]

def add_bias(x, bias):
    m, n = x.shape

    x_spec = pl.BlockSpec((BM, BN), lambda i, j: (i, j))
    b_spec = pl.BlockSpec((BN,), lambda i, j: (j,))
    o_spec = pl.BlockSpec((BM, BN), lambda i, j: (i, j))

    return pl.pallas_call(
        add_bias_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[x_spec, b_spec],
        out_specs=o_spec,
        grid=(pl.cdiv(m, BM), pl.cdiv(n, BN)),
    )(x, bias)
```

`bias` 的 block 只依赖 `j`，不依赖 `i`。这就是 BlockSpec 里的广播，不需要在数组层面先把 bias 扩展成 `[M, N]`。

## 示例 4：矩阵乘法的 3D grid

```python
BM, BK, BN = 128, 128, 128

def matmul_kernel(a_ref, b_ref, c_ref):
    @pl.when(pl.program_id(2) == 0)
    def _():
        c_ref[...] = jnp.zeros_like(c_ref)

    c_ref[...] += a_ref[...] @ b_ref[...]

def matmul(a, b):
    m, k = a.shape
    _, n = b.shape

    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), a.dtype),
        in_specs=[
            pl.BlockSpec((BM, BK), lambda i, j, kk: (i, kk)),
            pl.BlockSpec((BK, BN), lambda i, j, kk: (kk, j)),
        ],
        out_specs=pl.BlockSpec((BM, BN), lambda i, j, kk: (i, j)),
        grid=(pl.cdiv(m, BM), pl.cdiv(n, BN), pl.cdiv(k, BK)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL, pltpu.ARBITRARY),
        ),
    )(a, b)
```

这段代码展示了三个关键点：

- `a_ref` 的列块随 `kk` 变化。
- `b_ref` 的行块随 `kk` 变化。
- `c_ref` 不随 `kk` 变化，所以同一输出块会被连续累加。

如果把 grid 写成 `(K_blocks, M_blocks, N_blocks)`，同一个输出块的写入就不再按字典序连续，这会破坏 TPU 上安全累加的结构。

## 示例 5：Squeezed batch 维度

```python
def scale_batch_kernel(x_ref, o_ref):
    # x_ref.shape == (SEQ, D)
    o_ref[...] = x_ref[...] * 0.5

def scale_by_batch(x):
    batch, seq, d = x.shape

    spec = pl.BlockSpec(
        block_shape=(None, seq, d),
        index_map=lambda b: (b, 0, 0),
    )

    return pl.pallas_call(
        scale_batch_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[spec],
        out_specs=spec,
        grid=(batch,),
    )(x)
```

这里 grid 只沿 batch 推进。kernel 内部完全不用关心 batch 维度，只处理一个 `[seq, d]` 的矩阵。

## 示例 6：动态页号访问

假设有一个 KV cache，形状是 `[num_pages, page_size, head_dim]`，每个 program 根据 page table 读取一个物理页：

```python
PAGE, D = 128, 128

def copy_page_kernel(page_table_ref, page_ref, o_ref):
    o_ref[...] = page_ref[...]

def gather_pages(kv_cache, page_table):
    num_pages = page_table.shape[0]

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(num_pages,),
        in_specs=[
            pl.BlockSpec(
                (None, PAGE, D),
                lambda i, page_table_ref: (page_table_ref[i], 0, 0),
            ),
        ],
        out_specs=pl.BlockSpec(
            (None, PAGE, D),
            lambda i, page_table_ref: (i, 0, 0),
        ),
    )

    return pl.pallas_call(
        copy_page_kernel,
        out_shape=jax.ShapeDtypeStruct((num_pages, PAGE, D), kv_cache.dtype),
        grid_spec=grid_spec,
    )(page_table, kv_cache)
```

这是 paged attention 的基础模式：page table 放在 SMEM，`index_map` 用它决定 HBM 中真正要搬运的页。

## API 示例

本节把本章涉及的 Pallas API 按“能复制运行”的方式重新过一遍。除特别标注 TPU/SparseCore 的例子外，代码都使用 `interpret=True`，因此在安装了 `jax` + `jaxlib` 的 CPU 环境也能运行。`interpret=True` 只用于调试语义，不代表 TPU 性能。

### pl.pallas_call：最小完整调用

`pl.pallas_call` 是入口。kernel 不返回值，而是写入输出 Ref。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...] * 2


x = jnp.arange(8, dtype=jnp.float32)
y = pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    interpret=True,
)(x)

assert jnp.all(y == x * 2)
print(y)
```

这里没有显式写 `grid/in_specs/out_specs`，默认就是单次调用、整块输入、整块输出。

### grid=int：一维 grid

`grid=4` 会被规范化为 `grid=(4,)`。每个 program 通过 `program_id(0)` 写一个位置。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def kernel(o_ref):
    i = pl.program_id(0)
    o_ref[i] = i


y = pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct((4,), jnp.int32),
    grid=4,
    interpret=True,
)()

assert jnp.all(y == jnp.array([0, 1, 2, 3], dtype=jnp.int32))
print(y)
```

### grid=tuple：二维 grid 与 program_id

多维 grid 对应嵌套循环。下面的输出把 `(i, j)` 编码成 `10*i + j`。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def kernel(o_ref):
    i = pl.program_id(0)
    j = pl.program_id(1)
    o_ref[i, j] = 10 * i + j


y = pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct((3, 4), jnp.int32),
    grid=(3, 4),
    interpret=True,
)()

expected = jnp.array(
    [[0, 1, 2, 3],
     [10, 11, 12, 13],
     [20, 21, 22, 23]],
    dtype=jnp.int32,
)
assert jnp.all(y == expected)
print(y)
```

### pl.num_programs：读取 grid 大小

`pl.num_programs(axis)` 返回某个 grid 轴的长度。它常用于最后一轮处理、边界判断或把工作均分到多个 program。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def kernel(o_ref):
    i = pl.program_id(0)
    n = pl.num_programs(0)
    o_ref[i] = n - 1 - i


y = pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct((5,), jnp.int32),
    grid=(5,),
    interpret=True,
)()

assert jnp.all(y == jnp.array([4, 3, 2, 1, 0], dtype=jnp.int32))
print(y)
```

### pltpu.CompilerParams.dimension_semantics：标注 grid 维度

`dimension_semantics` 告诉 TPU backend 哪些 grid 维度可以并行。下面的例子语义上是普通 2D 加法；在 Megacore TPU 上，两个 grid 维度都可并行。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


BM, BN = 2, 4


def kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]


x = jnp.arange(24, dtype=jnp.float32).reshape(6, 4)
y = jnp.ones_like(x)
spec = pl.BlockSpec((BM, BN), lambda i, j: (i, j))

out = pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    in_specs=[spec, spec],
    out_specs=spec,
    grid=(pl.cdiv(x.shape[0], BM), pl.cdiv(x.shape[1], BN)),
    compiler_params=pltpu.CompilerParams(
        dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL),
    ),
    interpret=True,
)(x, y)

assert jnp.all(out == x + y)
print(out)
```

如果某个维度会重复写同一个输出块，例如 matmul 的 K 维归约，就应标为 `pltpu.ARBITRARY`。

### pl.BlockSpec：显式分块

`BlockSpec((4,), lambda i: (i,))` 表示第 `i` 个 program 处理第 `i` 个长度为 4 的块。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


BLOCK = 4


def kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...] + 1


x = jnp.arange(10, dtype=jnp.float32)
spec = pl.BlockSpec((BLOCK,), lambda i: (i,))

y = pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    in_specs=[spec],
    out_specs=spec,
    grid=(pl.cdiv(x.shape[0], BLOCK),),
    interpret=True,
)(x)

assert jnp.all(y == x + 1)
print(y)
```

最后一个块的 Ref 仍然是长度 4，但越界写回会被丢弃。

### block_shape=None：整个数组作为一个块

`pl.BlockSpec()` 或 `pl.BlockSpec(block_shape=None, index_map=None)` 都表示整块输入。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...] - 5


x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
whole = pl.BlockSpec()

y = pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    in_specs=[whole],
    out_specs=whole,
    grid=(1,),
    interpret=True,
)(x)

assert jnp.all(y == x - 5)
print(y)
```

在 TPU 上，大数组不要随便整块搬进 VMEM；这个例子只是说明默认语义。

### pl.GridSpec：把 grid/spec/scratch 打包

`GridSpec` 和 `grid + in_specs + out_specs` 是同一件事的显式对象形式。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...] * x_ref[...]


x = jnp.arange(8, dtype=jnp.float32)
spec = pl.BlockSpec((4,), lambda i: (i,))
grid_spec = pl.GridSpec(
    grid=(2,),
    in_specs=[spec],
    out_specs=spec,
)

y = pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    grid_spec=grid_spec,
    interpret=True,
)(x)

assert jnp.all(y == x * x)
print(y)
```

使用 `grid_spec=` 时，不要再同时传 `grid=`、`in_specs=`、`out_specs=` 或 `scratch_shapes=`。

### pl.no_block_spec：某个输入整体传入

`pl.no_block_spec` 常用于小 metadata：数据块按 `BlockSpec` 切，metadata 整体传入。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def kernel(scale_ref, x_ref, o_ref):
    o_ref[...] = x_ref[...] * scale_ref[0]


x = jnp.arange(8, dtype=jnp.float32)
scale = jnp.array([3.0], dtype=jnp.float32)
block = pl.BlockSpec((4,), lambda i: (i,))

y = pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    in_specs=[pl.no_block_spec, block],
    out_specs=block,
    grid=(2,),
    interpret=True,
)(scale, x)

assert jnp.all(y == x * 3)
print(y)
```

如果 metadata 是标量控制流数据，在 TPU 上通常更推荐放到 SMEM 或 scalar prefetch。

### pl.Squeezed / None：选择维度但不暴露给 kernel

`None` 和 `pl.Squeezed()` 都表示该维度大小为 1，并从 Ref shape 中移除。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def kernel(x_ref, o_ref):
    # 输入原 shape 是 [batch, width]，但 kernel 只看到 [width]
    assert x_ref.shape == (4,)
    o_ref[...] = x_ref[...] + 10


x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
spec = pl.BlockSpec((None, 4), lambda b: (b, 0))

y = pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    in_specs=[spec],
    out_specs=spec,
    grid=(3,),
    interpret=True,
)(x)

assert jnp.all(y == x + 10)
print(y)
```

等价写法是 `pl.BlockSpec((pl.Squeezed(), 4), lambda b: (b, 0))`。

### pl.Element：index_map 返回元素下标

默认 blocked 模式中 `index_map` 返回块编号。`pl.Element(size)` 改成返回元素起点。这个 API 主要面向 TPU 后端；下面是完整写法。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...]


x = jnp.arange(10, dtype=jnp.float32)
in_spec = pl.BlockSpec(
    (pl.Element(4),),
    lambda i: (2 * i,),  # 元素起点：0, 2, 4
)
out_spec = pl.BlockSpec((None, 4), lambda i: (i, 0))

y = pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct((3, 4), x.dtype),
    in_specs=[in_spec],
    out_specs=out_spec,
    grid=(3,),
    interpret=True,
)(x)

expected = jnp.stack([x[0:4], x[2:6], x[4:8]])
assert jnp.all(y == expected)
print(y)
```

如果你只是规则地按块遍历，优先使用默认 blocked 模式。

### pl.ds / pl.dslice：Ref 上的动态切片

`pl.ds(start, size)` 和 `pl.dslice(start, size)` 是同一个动态切片构造器，常用于 Ref indexing。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


BLOCK = 4


def kernel(x_ref, o_ref):
    i = pl.program_id(0)
    s = pl.ds(i * BLOCK, BLOCK)
    o_ref[s] = x_ref[s] + 100


x = jnp.arange(12, dtype=jnp.float32)
y = pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    grid=(3,),
    interpret=True,
)(x)

assert jnp.all(y == x + 100)
print(y)
```

`pl.ds` 的 `start` 可以是运行时值；普通 Python `slice` 的边界必须更静态。

### pl.cdiv / align_to / next_power_of_2：形状工具

这几个工具是写 grid 和 tile 时的常用工具。

```python
from jax.experimental import pallas as pl


assert pl.cdiv(10, 4) == 3
assert pl.align_to(10, 4) == 12
assert pl.next_power_of_2(17) == 32

M, N = 1000, 513
BM, BN = 128, 128
grid = (pl.cdiv(M, BM), pl.cdiv(N, BN))
assert grid == (8, 5)
print(grid)
```

`pl.cdiv` 最常见的用途就是计算覆盖完整张量的 grid。

### pytree out_shape / out_specs：多个输出

输出可以是 pytree。kernel 参数顺序是：所有输入 Ref，然后按 pytree flatten 后的所有输出 Ref，再然后是 scratch Ref。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def kernel(x_ref, y_ref, sum_ref, diff_ref):
    sum_ref[...] = x_ref[...] + y_ref[...]
    diff_ref[...] = x_ref[...] - y_ref[...]


x = jnp.arange(8, dtype=jnp.float32)
y = jnp.ones((8,), dtype=jnp.float32)
spec = pl.BlockSpec((4,), lambda i: (i,))

sum_out, diff_out = pl.pallas_call(
    kernel,
    out_shape=(
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        jax.ShapeDtypeStruct(x.shape, x.dtype),
    ),
    in_specs=[spec, spec],
    out_specs=(spec, spec),
    grid=(2,),
    interpret=True,
)(x, y)

assert jnp.all(sum_out == x + y)
assert jnp.all(diff_out == x - y)
print(sum_out, diff_out)
```

如果 `out_shape` 是 dict/list/tuple，`out_specs` 也要用同样的 pytree 结构。

### pltpu.PrefetchScalarGridSpec：index_map 读取 SMEM metadata

`PrefetchScalarGridSpec` 让 `index_map` 除了 grid 坐标外，还能读取前几个 scalar prefetch 参数。下面的例子根据 `block_ids` 动态选择输入块。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


BLOCK = 4


def kernel(block_ids_ref, x_ref, o_ref):
    o_ref[...] = x_ref[...]


x = jnp.arange(16, dtype=jnp.float32)
block_ids = jnp.array([2, 0, 3], dtype=jnp.int32)

grid_spec = pltpu.PrefetchScalarGridSpec(
    num_scalar_prefetch=1,
    grid=(block_ids.shape[0],),
    in_specs=[
        pl.BlockSpec(
            (BLOCK,),
            lambda i, block_ids_ref: (block_ids_ref[i],),
        )
    ],
    out_specs=pl.BlockSpec(
        (BLOCK,),
        lambda i, block_ids_ref: (i,),
    ),
)

y = pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct((block_ids.shape[0] * BLOCK,), x.dtype),
    grid_spec=grid_spec,
    interpret=True,
)(block_ids, x)

expected = jnp.concatenate([x[8:12], x[0:4], x[12:16]])
assert jnp.all(y == expected)
print(y)
```

`num_scalar_prefetch=1` 表示 `block_ids` 不出现在 `in_specs` 中，但会作为 SMEM Ref 传给 kernel 和所有 `index_map`。

### input_output_aliases：输出复用输入 buffer

`input_output_aliases={input_index: output_index}` 用于声明某个输入和输出 alias。它主要是性能/内存优化，不改变数学语义。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...] + 1


x = jnp.arange(8, dtype=jnp.float32)
y = pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    input_output_aliases={0: 0},
    interpret=True,
)(x)

assert jnp.all(y == x + 1)
print(y)
```

不要把 alias 当作 Python 原地修改；JAX 调用语义仍然是函数式的。

### pl.Buffered / pipeline_mode：声明自动缓冲策略

`pipeline_mode` 是 `BlockSpec` 的参数，用于告诉自动流水线该 operand 如何缓冲。这个例子在语义上和普通分块相同，但显式要求双缓冲。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...] + 2


x = jnp.arange(8, dtype=jnp.float32)
spec = pl.BlockSpec(
    (4,),
    lambda i: (i,),
    pipeline_mode=pl.Buffered(buffer_count=2),
)

y = pl.pallas_call(
    kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    in_specs=[spec],
    out_specs=spec,
    grid=(2,),
    interpret=True,
)(x)

assert jnp.all(y == x + 2)
print(y)
```

真实 TPU 性能差异要看编译后的 DMA 调度；`interpret=True` 只验证语义。

### pl.BoundedSlice：动态但有界的块大小（TPU/emit_pipeline）

`pl.BoundedSlice(max_size)` 要求 `index_map` 返回 `pl.ds(start, size)`，且 `size <= max_size`。它最常见于 `pltpu.emit_pipeline`，下面是可在 TPU 环境运行的完整例子。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def dynamic_copy_kernel(x_hbm_ref, slices_hbm_ref, o_hbm_ref, slices_smem_ref):
    pltpu.sync_copy(slices_hbm_ref, slices_smem_ref)

    def body(x_ref, o_ref):
        o_ref[...] = x_ref[...]

    def index_map(i):
        start = slices_smem_ref[i, 0]
        end = slices_smem_ref[i, 1]
        return (pl.ds(start, end - start), 0)

    spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(8), 128),
        index_map=index_map,
    )

    pltpu.emit_pipeline(
        body,
        grid=(4,),
        in_specs=[spec],
        out_specs=spec,
    )(x_hbm_ref, o_hbm_ref)


x = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
slices = jnp.array([[0, 2], [2, 3], [3, 5], [5, 8]], dtype=jnp.int32)

out = pl.pallas_call(
    dynamic_copy_kernel,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    in_specs=[
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(memory_space=pltpu.HBM),
    ],
    out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
    scratch_shapes=[pltpu.SMEM(slices.shape, slices.dtype)],
)(x, slices)

assert jnp.all(out == x)
print(out)
```

这个例子需要 TPU backend，因为它使用了 `pltpu.sync_copy`、`pltpu.emit_pipeline` 和 SMEM。

### pl.Indirect：向量化间接索引（SparseCore）

`pl.Indirect(size)` 表示该维度由一组索引决定，输入侧是 gather，输出侧是 scatter。它主要用于 SparseCore/SC tiling。下面是一个完整形态的 SparseCore gather kernel；其中 `indices_ref.at[i]` 给出第 `i` 轮要 gather 的一组行。

```python
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


sc_info = pltpu.get_tpu_info().sparse_core
num_lanes = sc_info.num_lanes
num_steps = 4

mesh = plsc.VectorSubcoreMesh(
    core_axis_name="core",
    num_cores=sc_info.num_cores,
    subcore_axis_name="subcore",
    num_subcores=1,
)


@functools.partial(
    pl.kernel,
    mesh=mesh,
    out_type=jax.ShapeDtypeStruct((num_steps, num_lanes, num_lanes), jnp.int32),
    scratch_types={
        "indices_ref": pltpu.VMEM((num_steps, num_lanes), jnp.int32),
    },
)
def gather_kernel(x_hbm_ref, indices_hbm_ref, o_hbm_ref, *, indices_ref):
    pltpu.sync_copy(indices_hbm_ref, indices_ref)

    @functools.partial(
        pltpu.emit_pipeline,
        grid=(num_steps,),
        in_specs=[
            pl.BlockSpec(
                (pl.Indirect(num_lanes), num_lanes),
                lambda i: (indices_ref.at[i], 0),
            )
        ],
        out_specs=pl.BlockSpec(
            (pl.Squeezed(), num_lanes, num_lanes),
            lambda i: (i, 0, 0),
        ),
        tiling=pltpu.Tiling.SPARSE_CORE,
    )
    def pipeline(x_ref, o_ref):
        o_ref[...] = x_ref[...]

    pipeline(x_hbm_ref, o_hbm_ref)


x = jnp.arange(num_steps * num_lanes * num_lanes, dtype=jnp.int32).reshape(
    num_steps * num_lanes, num_lanes
)
indices = jnp.arange(num_steps * num_lanes, dtype=jnp.int32).reshape(
    num_steps, num_lanes
)
out = gather_kernel(x, indices)
assert out.shape == (num_steps, num_lanes, num_lanes)
print(out)
```

这个例子需要带 SparseCore 的 TPU。普通 TensorCore kernel 里如果只是“根据一个页号选连续页”，通常用 `PrefetchScalarGridSpec` 或手动 DMA，而不是 `pl.Indirect`。

## 设计检查清单

写 `grid` 和 `BlockSpec` 时，可以按下面顺序检查：

1. 输出块由哪些 grid 维度决定？
2. 哪些维度只是归约或累加？把它们放在 grid 后缀。
3. 每个输入块是否真的需要依赖所有 grid 维度？能忽略的维度就忽略，让编译器复用数据。
4. block_shape 是否 tile 友好，尤其是最后两个维度？
5. 最后一个块越界时，kernel 是否会把 padding 参与计算？如果会，必须 mask。
6. metadata 是规则静态索引、SMEM scalar prefetch，还是需要手动 DMA？
7. 哪些 grid 维度可并行？只把真正独立的维度标为 `PARALLEL`。

## 常见错误

**把 block index 当元素 offset：**

```python
# 错误：默认 blocked 模式下会跳过大量块
pl.BlockSpec((128,), lambda i: (i * 128,))

# 正确
pl.BlockSpec((128,), lambda i: (i,))
```

**归约维度放在 grid 前缀：**

```python
# 不推荐：k 在前，同一个输出块的写入不连续
grid = (K_blocks, M_blocks, N_blocks)

# 推荐
grid = (M_blocks, N_blocks, K_blocks)
```

**无 mask 地使用越界 padding 做 reduction：**

```python
# 有风险：最后一个块的 padding 值不保证为 0
o_ref[...] = jnp.sum(x_ref[...])
```

**把小标量放进 VMEM：**

```python
# 通常低效
scratch_shapes=[pltpu.VMEM((1,), jnp.int32)]

# 更适合
scratch_shapes=[pltpu.SMEM((1,), jnp.int32)]
```

**错误标注 parallel：**

```python
# 如果 k 维会累加到同一个输出块，不要标 parallel
dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL, pltpu.ARBITRARY)
```

## TPU 上的 grid 执行语义

在单 TensorCore TPU kernel 中，grid 通常按字典序顺序推进。这一点和 GPU 的“很多 block 并行抢占执行”不同。

这个顺序模型带来两个重要能力：

- 相邻 program 如果读取同一块输入，编译器可以复用已经在 VMEM 中的数据，避免重复 HBM -> VMEM 传输。
- 多个 program 可以连续写入同一个输出块，用于归约或累加，不需要原子操作。

但第二点有一个关键限制：写同一输出块的 program 必须在 grid 顺序中连续出现。实践中通常把“输出块坐标”放在 grid 的前缀维度，把“归约/累加维度”放在最后。

矩阵乘法就是典型例子：

```python
# grid = (M_blocks, N_blocks, K_blocks)
# i, j 决定输出 C 的块
# k 是归约维度，必须作为后缀维度
out_specs = pl.BlockSpec((BM, BN), lambda i, j, k: (i, j))
```

对于固定的 `(i, j)`，所有 `k` 会连续执行，因此同一个 `C[i, j]` 块可以作为累加器使用。

### `dimension_semantics` 的使用

如果 TPU 设备以 Megacore 形式暴露多个 TensorCore，可以通过 `pltpu.CompilerParams(dimension_semantics=...)` 告诉编译器哪些 grid 维度可并行分配到多个 core。

```python
from jax.experimental.pallas import tpu as pltpu

compiler_params = pltpu.CompilerParams(
    dimension_semantics=(
        pltpu.PARALLEL,   # i 维：不同输出行块独立
        pltpu.PARALLEL,   # j 维：不同输出列块独立
        pltpu.ARBITRARY,  # k 维：同一输出块的归约后缀
    )
)
```

经验规则：

- `PARALLEL`：不同 program 之间没有数据依赖，可以跨 core 并行。
- `ARBITRARY`：不要把该维度并行化；常用于归约、累加、跨迭代状态。
- 常见布局是若干个 `PARALLEL` 前缀维度，加若干个 `ARBITRARY` 后缀维度。

如果某个维度会导致多个 program 写同一个输出块，它通常不应该标成 `PARALLEL`。

## 与第 4 章的连接

`BlockSpec` 的默认模式会自动生成 HBM <-> VMEM 的 DMA，这适合规则分块访问。第 4 章会展开三种更细的内存管理方式：

- 自动 DMA：`BlockSpec((...), index_map=...)`
- 半自动流水线：`pltpu.emit_pipeline(...)`
- 手动 DMA：`BlockSpec(memory_space=pltpu.HBM)` + `sync_copy` / `make_async_copy`

理解本章后，可以把复杂 kernel 拆成两个层次：

```text
外层：grid / BlockSpec 决定每个 program 的数据窗口
内层：kernel body 决定窗口内如何计算、是否手动搬运、是否跨迭代累加
```

这是 Pallas TPU 编程最重要的概念之一。
