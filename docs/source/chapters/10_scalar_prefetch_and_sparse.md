# 第 10 章：标量预取与动态稀疏索引

到目前为止，我们编写的 Kernel（如矩阵乘法）都具有**静态的、规则的内存访问模式**。`BlockSpec` 的 `index_map` 仅仅依赖于网格索引 `(i, j)`，这意味着编译器在 Kernel 运行前就能确切知道每次需要从 HBM 搬运哪些数据。

但在现代大模型推理中，我们经常遇到**动态的、不规则的内存访问**。最典型的例子就是 Paged Attention（分页注意力）：我们需要根据一个动态的 `page_indices` 数组，从海量的 KV Cache 中挑出特定的页（Page）加载到 VMEM。

这种模式无法用普通的 `BlockSpec` 表达。Pallas 提供了 `PrefetchScalarGridSpec` 和**标量预取（Scalar Prefetch）**来解决这个问题。

## 为什么需要标量预取？

TPU 的标量核心（Scalar Core）拥有自己的内存（SMEM）。SMEM 的特点是延迟极低，支持 32-bit 的细粒度随机访问。

**标量预取的思想是**：
1. 将包含动态索引的数组（例如 `page_indices`）预先加载到 SMEM 中。
2. 扩展 `index_map` 函数的签名，使其不仅接收网格索引 `(i, j)`，还能接收 SMEM 中的这些动态索引数据。
3. `index_map` 根据读到的动态索引，决定去 HBM 的哪个位置取真正的数据块。

## PrefetchScalarGridSpec 的用法

要使用标量预取，我们需要用 `pltpu.PrefetchScalarGridSpec` 替换 `pallas_call` 默认的 `grid` 和 `in_specs`/`out_specs`。

### 1. 核心参数 `num_scalar_prefetch`

这是最重要的参数。它告诉 Pallas：传入 `pallas_call` 的**前 N 个参数**是标量预取参数，应该被加载到 SMEM 中。

```python
# 假设传入 pallas_call 的参数顺序是：
# kernel(indices_array, dense_data_array)
# 我们希望 indices_array 进入 SMEM，dense_data_array 进入 VMEM

grid_spec = pltpu.PrefetchScalarGridSpec(
    num_scalar_prefetch=1,  # 第 1 个参数是预取参数
    in_specs=[data_block_spec], # 只需为后续的普通参数提供 BlockSpec
    out_specs=...,
    grid=(...),
)
```

**注意：** 预取参数不需要 `BlockSpec`。它们会被完整地（或根据网格自动切分地）放入 SMEM。

### 2. index_map 的签名变化

当使用了标量预取后，所有普通参数的 `index_map` 都会接收到额外的参数。

```python
# 传统的 index_map
# lambda i, j: (i, j)

# 带有 1 个标量预取的 index_map
def data_index_map(i, j, indices_smem_ref):
    # indices_smem_ref 是驻留在 SMEM 中的引用
    # 我们可以从中读取动态索引！
    dynamic_idx = indices_smem_ref[i, j]
    
    # 返回真正的块索引
    return (dynamic_idx, 0)

data_block_spec = pl.BlockSpec(
    block_shape=(128, 128),
    index_map=data_index_map
)
```

### 3. Kernel 函数的签名变化

Kernel 函数的参数列表也会发生变化，预取参数会排在最前面：

```python
def my_sparse_kernel(indices_smem_ref, data_vmem_ref, out_vmem_ref):
    # indices_smem_ref: SMEM 引用
    # data_vmem_ref: VMEM 引用 (根据动态索引从 HBM 抓取过来的数据)
    # out_vmem_ref: VMEM 引用
    
    # 执行计算...
    out_vmem_ref[...] = data_vmem_ref[...] * 2.0
```

## 实战：动态块提取 (Dynamic Block Slicing)

让我们看一个完整的最小示例。我们有一个巨大的 1D 数据数组，和一个包含索引的数组。我们想根据索引提取数据块。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def dynamic_slice_kernel(indices_ref, data_ref, out_ref):
    # indices_ref 在 SMEM 中
    # data_ref 已经在 VMEM 中了，且正是我们需要的那个块！
    out_ref[...] = data_ref[...]

def dynamic_block_slice(data: jax.Array, indices: jax.Array, block_size: int = 128):
    num_blocks = indices.shape[0]
    
    # index_map 接收网格索引 i，以及预取的 indices_ref
    def data_idx_map(i, indices_ref):
        # 从 SMEM 读取标量索引
        idx = indices_ref[i]
        return (idx,)
        
    data_spec = pl.BlockSpec(
        block_shape=(block_size,),
        index_map=data_idx_map
    )
    
    out_spec = pl.BlockSpec(
        block_shape=(block_size,),
        index_map=lambda i, *_: (i,) # 输出是连续写入的
    )
    
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        in_specs=[data_spec],
        out_specs=out_spec,
        grid=(num_blocks,)
    )
    
    return pl.pallas_call(
        dynamic_slice_kernel,
        out_shape=jax.ShapeDtypeStruct((num_blocks * block_size,), data.dtype),
        grid_spec=grid_spec
    )(indices, data) # 注意：indices 作为第一个参数传入

# 测试
data = jnp.arange(1024, dtype=jnp.float32)
indices = jnp.array([5, 1, 7], dtype=jnp.int32) # 我们想要第 5, 1, 7 个块
# 结果将是这三个块的拼接
```

## 高级用法：Ragged Paged Attention 中的手动 DMA

在极其复杂的场景下（如 Ragged Paged Attention），连 `PrefetchScalarGridSpec` 自动处理的 DMA 都无法满足需求。因为每个 Sequence 的 Page 数量不同，且我们需要在 Kernel 内部使用 `lax.while_loop` 动态决定何时取下一个 Page。

在这种硬核场景下，开发者会：
1. 使用 `PrefetchScalarGridSpec` 将 `page_indices` 放入 SMEM。
2. 将 KV Cache 的 HBM 引用（`memory_space=pltpu.HBM`）直接传入 Kernel，**不分配 BlockSpec**。
3. 在 Kernel 内部，分配 VMEM Scratch Buffer 和 DMA 信号量（Semaphore）。
4. 在 Kernel 的 `while_loop` 中，手动调用 `pltpu.make_async_copy`，根据 SMEM 中的 `page_indices`，发起从 HBM 到 VMEM 的异步拷贝。

这种极致的手动控制，正是 Pallas 赋予开发者的终极能力。我们将在进阶部分的第 13 章详细剖析 Ragged Paged Attention 的源码。
