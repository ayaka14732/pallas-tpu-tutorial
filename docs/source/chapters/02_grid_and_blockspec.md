# 第 2 章：Grid、BlockSpec 与 Dimension Semantics

本章讨论 Pallas TPU kernel 中的 Grid、BlockSpec 与 Dimension Semantics：

- `grid`：kernel body 会被执行多少次，每次执行的坐标是什么。
- `BlockSpec`：每次执行时，每个输入/输出 Ref 对应原数组的哪一块，以及这块数据应放在哪个内存空间。
- Dimension Semantics：仅用于 TPU v4 和 v5p，标注某个维度是否可以并行。

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

TPU TensorCore 的 VMEM/VREG 按 tile 工作，block shape 不是任意形状都高效，也不是任意形状都被后端支持。

官方规则可以归纳为：

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

### pltpu.PrefetchScalarGridSpec：index_map 根据 SMEM 数据动态选择输入块

`PrefetchScalarGridSpec` 让 `index_map` 除了 grid 坐标外，还能读取 scalar prefetch 参数。下面的例子根据 `block_ids` 动态选择输入块。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

is_tpu_available = jax.devices()[0].platform == "tpu"

NUM_BLOCKS = 6
BLOCK_ROWS = 8
BLOCK_COLS = 128

def kernel(block_ids_ref: jax.Ref, x_ref: jax.Ref, o_ref: jax.Ref) -> None:
    block_ids_ref[1] = 3
    block_ids_ref[2] = 3
    block_ids_ref[3] = 3
    block_ids_ref[4] = 3
    block_ids_ref[5] = 3
    o_ref[...] = x_ref[...]

def fn(block_ids: jax.Array, x: jax.Array) -> jax.Array:
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(NUM_BLOCKS,),
        in_specs=[
            pl.BlockSpec(
                block_shape=(None, BLOCK_ROWS, BLOCK_COLS),
                index_map=lambda i, block_ids_ref: (block_ids_ref[i], 0, 0),
                pipeline_mode=pl.Buffered(1),  # use single buffering
            )
        ],
        out_specs=pl.BlockSpec(
            block_shape=(None, BLOCK_ROWS, BLOCK_COLS),
            index_map=lambda i, block_ids_ref: (i, 0, 0),
        ),
    )

    return pl.pallas_call(
        kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=False if is_tpu_available else pltpu.InterpretParams(),
        name="prefetch_fn",
    )(block_ids, x)

x = jnp.broadcast_to(jnp.arange(NUM_BLOCKS, dtype=jnp.int32).reshape(-1, 1, 1), (NUM_BLOCKS, BLOCK_ROWS, BLOCK_COLS))
# x[:, 0, 0] = [0, 1, 2, 3, 4, 5]

block_ids = jnp.array([2, 0, 0, 0, 0, 0], dtype=jnp.int32)
output = fn(block_ids, x)
print(output[:, 0, 0])  # expected [2 3 3 3 3 3] when `pipeline_mode=pl.Buffered(1)`
```

`num_scalar_prefetch=1` 表示 `block_ids` 不出现在 `in_specs` 中，但会作为 SMEM Ref 传给 kernel 和所有 `index_map`。

:::{warning}
在解释模式下，程序会给出错误的结果，详见 [jax-ml/jax#39179](https://github.com/jax-ml/jax/issues/39179)。
:::

## `dimension_semantics` 的使用

如果 TPU 设备以 Megacore 形式暴露多个 TensorCore，可以通过 `pltpu.CompilerParams(dimension_semantics=...)` 告诉编译器哪些 grid 维度可并行分配到多个 core。

可以取以下两个值：

- `pltpu.PARALLEL`：不同 program 之间没有数据依赖，可以跨 core 并行。
- `pltpu.ARBITRARY`：不要把该维度并行化；常用于归约、累加、跨迭代状态。
- 常见布局是若干个 `PARALLEL` 前缀维度，加若干个 `ARBITRARY` 后缀维度。

如果某个维度会导致多个 program 写同一个输出块，它通常不应该标成 `PARALLEL`。

例子如下：

```python
import jax
from jax.experimental.pallas import tpu as pltpu

if jax.devices()[0].platform != 'tpu' or not pltpu.get_tpu_info().is_megacore:
    print("Note: This device does not operate in Megacore mode.")

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

N = 32

def kernel(x_ref: jax.Ref, o_ref: jax.Ref) -> None:
    i = pl.program_id(0)

    @pl.when(i == 0)
    def _():
        o_ref[...] = jnp.zeros_like(o_ref)

    o_ref[...] += jnp.full_like(o_ref, 1.0) + x_ref[...]

def fn(x: jax.Array) -> jax.Array:
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), x.dtype),
        in_specs=[pl.BlockSpec(block_shape=(8, 128), index_map=lambda i: (i, 0))],
        out_specs=pl.BlockSpec(block_shape=(8, 128), index_map=lambda i: (0, 0)),
        grid=(N,),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(pltpu.PARALLEL,),  # <-- 错误！应该使用 `pltpu.ARBITRARY`
        ),
        # interpret=True,
    )(x)

x = jnp.zeros((N * 8, 128), dtype=jnp.float32)
result = fn(x)
print(result)
```

在 TPU v4 上：

输出矩阵全为 16，因为两个 core 并行了，写回的时候发生覆盖。

而如果使用 `pltpu.ARBITRARY`，或者解释模式，或者在没有 Megacore 的 TPU 上，输出矩阵全为 32。

只有 v4 和 v5p 有 Megacore；v5e, v6e, 7x, 8i 及更以后的都没有。

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

## API 示例

本节把本章涉及的 Pallas API 按“能复制运行”的方式重新过一遍。除特别标注 TPU/SparseCore 的例子外，代码都使用 `interpret=True`，因此在安装了 `jax` + `jaxlib` 的 CPU 环境也能运行。`interpret=True` 只用于调试语义，不代表 TPU 性能。

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

### pytree out_shape / out_specs：多个输出

输出可以是 pytree。kernel 参数顺序是：所有输入 Ref，然后按 pytree flatten 后的所有输出 Ref，再然后是 scratch Ref。

如果 `out_shape` 是 dict/list/tuple，`out_specs` 也要用同样的 pytree 结构。

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

但是 Megacore 不是。

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
