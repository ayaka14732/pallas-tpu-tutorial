# 第 2 章：Pallas Hello World

## 编程模型概览

Pallas kernel 由两部分组成：

1. **Kernel 函数**：在 TPU 向量核心上执行，操作的是 `Ref`（内存引用），而非普通 JAX 数组。Kernel 函数没有返回值，结果通过写入输出 `Ref` 来传递。
2. **`pallas_call` 调度**：在 host 端定义 grid、BlockSpec、scratch shapes 等，告诉编译器如何切分数据并调度 kernel。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
```

## Ref 语义

`Ref` 是 Pallas 的核心抽象。它代表一块内存区域的引用（类似 C 的指针），而非一个值。

```python
def add_kernel(x_ref, y_ref, z_ref):
    # x_ref[...] 从 VMEM 加载整个块到 VREG（向量寄存器）
    x = x_ref[...]
    y = y_ref[...]
    z = x + y
    # z_ref[...] = z 将 VREG 中的结果写回 VMEM
    z_ref[...] = z
```

`Ref` 支持的操作：

| 操作 | 语义 | 示例 |
| :--- | :--- | :--- |
| `ref[...]` | 加载整个块 | `x = x_ref[...]` |
| `ref[i]` | 加载第 i 个切片 | `row = x_ref[0]` |
| `ref[pl.ds(start, size)]` | 动态切片 | `x_ref[pl.ds(i*128, 128)]` |
| `ref[...] = val` | 写入整个块 | `z_ref[...] = result` |
| `ref.at[idx]` | 获取子引用（不触发加载）| `sub = ref.at[0, pl.ds(0, 64)]` |

`ref.at[...]` 与 `ref[...]` 的区别：`ref.at[...]` 返回的仍然是一个 `Ref`（子引用），不触发实际的内存加载。这在 DMA 操作中非常重要——你需要传递一个 `Ref` 给 `make_async_copy`，而不是一个已加载的值。

## pallas_call 的完整签名

```python
result = pl.pallas_call(
    kernel_fn,                    # Kernel 函数
    out_shape=...,                # 输出的 shape 和 dtype
    grid=...,                     # 网格大小（循环次数）
    in_specs=[...],               # 输入的 BlockSpec 列表
    out_specs=...,                # 输出的 BlockSpec
    grid_spec=...,                # 或者使用 PrefetchScalarGridSpec
    scratch_shapes=[...],         # Scratch buffer 分配
    compiler_params=...,          # 编译器参数
    input_output_aliases={...},   # 输入输出别名（in-place 更新）
    name="...",                   # Kernel 名称（用于 profiling）
)(inputs...)
```

## 向量加法：完整示例

```python
def add_kernel(x_ref, y_ref, z_ref):
    z_ref[...] = x_ref[...] + y_ref[...]

def vector_add(x: jax.Array, y: jax.Array) -> jax.Array:
    block_size = 1024  # TPU 上可以用大块，因为 VMEM 有 16MB+
    num_blocks = x.shape[0] // block_size

    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[
            pl.BlockSpec(block_shape=(block_size,), index_map=lambda i: (i,)),
            pl.BlockSpec(block_shape=(block_size,), index_map=lambda i: (i,)),
        ],
        out_specs=pl.BlockSpec(block_shape=(block_size,), index_map=lambda i: (i,)),
        grid=(num_blocks,),
    )(x, y)
```

## 执行流程

当 `vector_add(x, y)` 被调用时，Mosaic 编译器生成的代码等价于：

```
for i in range(num_blocks):
    # 1. DMA: HBM -> VMEM（编译器自动生成）
    x_vmem = DMA_load(x_hbm[i*1024 : (i+1)*1024])
    y_vmem = DMA_load(y_hbm[i*1024 : (i+1)*1024])

    # 2. 执行 kernel（VMEM -> VREG -> 计算 -> VREG -> VMEM）
    z_vmem = x_vmem + y_vmem

    # 3. DMA: VMEM -> HBM（编译器自动生成）
    DMA_store(z_vmem -> z_hbm[i*1024 : (i+1)*1024])
```

编译器还会自动插入**流水线**：当第 i 次迭代在计算时，第 i+1 次迭代的 DMA 加载已经在后台进行。这就是为什么即使是这个简单的 kernel，TPU 也能接近 HBM 带宽上限。

## 不使用 BlockSpec 的情况

当你不想让编译器自动切分数据时，可以使用 `pl.BlockSpec(memory_space=pltpu.HBM)` 或 `pl.BlockSpec(memory_space=pltpu.VMEM)`。此时整个数组作为一个 `Ref` 传入 kernel，由你自己管理数据搬运。

```python
# 整个数组驻留在 HBM，kernel 内部手动 DMA
in_specs = [pl.BlockSpec(memory_space=pltpu.HBM)]

# 整个数组预加载到 VMEM（仅当数据量小于 VMEM 容量时可用）
in_specs = [pl.BlockSpec(memory_space=pltpu.VMEM)]
```

这种模式在 Ragged Paged Attention 等复杂 kernel 中非常常见——grid 设为 `(1,)`，所有循环逻辑由 kernel 内部的 `pl.loop` 控制。

## Interpret 模式

Pallas 提供 interpret 模式，允许在 CPU 上模拟 kernel 执行，用于调试：

```python
# 方法 1：全局开启
pltpu.set_tpu_interpret_mode(True)

# 方法 2：通过 compiler_params
result = pl.pallas_call(
    kernel_fn,
    ...,
    interpret=True,  # 在 CPU 上模拟执行
)(inputs)
```

Interpret 模式下，所有 DMA 操作会被模拟为普通的内存拷贝，信号量操作会被模拟为计数器。这对于验证 kernel 逻辑的正确性非常有用，但不能反映真实的性能特征。

## pl.program_id 和 pl.num_programs

在 kernel 内部，可以通过 `pl.program_id(axis)` 获取当前 grid 索引，通过 `pl.num_programs(axis)` 获取 grid 大小：

```python
def kernel(x_ref, o_ref):
    i = pl.program_id(0)  # 当前在第几次迭代
    n = pl.num_programs(0)  # 总共多少次迭代
    # 可以用于条件执行
    @pl.when(i == 0)
    def _():
        # 只在第一次迭代执行的初始化逻辑
        ...
```

## debug_print

`pl.debug_print` 用于在 kernel 内部打印调试信息。它会在 TPU 上实际执行时输出到 host 的 stdout：

```python
def kernel(x_ref, o_ref):
    i = pl.program_id(0)
    val = x_ref[0, 0]
    pl.debug_print("iteration {}, first element = {}", i, val)
```

注意：`debug_print` 会引入同步点，影响性能。仅用于调试，生产代码中应移除。
