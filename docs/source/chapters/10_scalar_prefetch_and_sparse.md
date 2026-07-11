# 第 10 章：标量预取与动态稀疏索引

到目前为止，我们编写的 Kernel（如矩阵乘法、RMSNorm）都具有**静态的、规则的内存访问模式**。`BlockSpec` 的 `index_map` 仅仅依赖于网格索引 `(i, j)`。这意味着编译器（Mosaic）在 Kernel 运行前就能确切地生成一条静态的 DMA 指令流，按部就班地把数据搬进搬出。

但在现代大模型推理中，我们经常遇到**动态的、不规则的内存访问**。最典型的例子就是 Paged Attention（分页注意力）：我们需要根据一个动态的 `page_indices` 数组，从海量的 KV Cache 池中挑出特定的物理页（Page）加载到 VMEM。

这种模式无法用普通的 `BlockSpec` 表达。Pallas 提供了 `PrefetchScalarGridSpec` 和**标量预取（Scalar Prefetch）**来解决这个问题。

## GPU vs TPU：处理动态索引的区别

在 **CUDA (GPU)** 中，处理动态索引非常自然。因为 CUDA 线程可以直接发起对全局内存（Global Memory）的读取。
```cpp
// CUDA 中的动态读取非常直观
int page_idx = page_indices[blockIdx.x]; // 从全局内存读取索引
float* page_data = kv_cache[page_idx];   // 根据索引从全局内存读取数据
```

在 **TPU** 中，这就成了一个大问题。因为 TPU 的向量核心（执行 Kernel 的地方）**不能**直接访问 HBM。数据必须由 DMA 引擎提前搬运到 VMEM。但是，如果 DMA 引擎不知道 `page_idx` 是多少，它怎么预取数据呢？

**Pallas 的解决方案是：利用标量核心（Scalar Core）和标量内存（SMEM）。**

## 标量预取的思想

TPU 包含一个标量核心，它拥有自己极低延迟的 SMEM。标量核心可以快速读取 SMEM 中的数据，并动态地配置 DMA 描述符，告诉 DMA 引擎接下来去 HBM 的哪个物理地址抓取向量数据。

**标量预取的工作流：**
1. 在 Host 端配置 `PrefetchScalarGridSpec`。
2. 运行时首先将包含动态索引的数组（例如 `page_indices`）预先加载到 SMEM 中。
3. `index_map` 函数的签名被扩展。它现在不仅接收网格索引 `(i, j)`，还能接收驻留在 SMEM 中的这些动态索引的引用。
4. 在计算每个 Grid 步骤的 DMA 传输计划时，`index_map` 会读取 SMEM 中的动态索引，并返回真正的 HBM 数据块坐标。
5. DMA 引擎根据这个动态坐标，将数据从 HBM 搬运到 VMEM。

## PrefetchScalarGridSpec 的用法

要使用标量预取，我们需要用 `pltpu.PrefetchScalarGridSpec` 替换 `pallas_call` 默认的 `grid` 参数。

### 1. 核心参数 `num_scalar_prefetch`

这是最重要的参数。它告诉 Pallas：传入 `pallas_call` 的**前 N 个参数**是标量预取参数，应该被放置到 SMEM 中。

```python
# 假设传入 pallas_call 的参数顺序是：
# kernel(indices_array, dense_data_array)
# 我们希望 indices_array 进入 SMEM，dense_data_array 根据 indices_array 动态进入 VMEM

grid_spec = pltpu.PrefetchScalarGridSpec(
    num_scalar_prefetch=1,  # 声明第 1 个参数是预取参数
    in_specs=[data_block_spec], # 只需为后续的普通参数提供 BlockSpec
    out_specs=...,
    grid=(...),
)
```

**注意：** 预取参数本身不需要 `BlockSpec`。它们会被完整地（或根据网格第一维自动切分地）放入 SMEM。

### 2. index_map 的签名变化

当使用了标量预取后，所有普通参数的 `index_map` 都会接收到额外的参数。

```python
# 传统的 index_map
# lambda i, j: (i, j)

# 带有 1 个标量预取的 index_map
def data_index_map(i, j, indices_smem_ref):
    # indices_smem_ref 是驻留在 SMEM 中的引用
    # 我们可以从中读取标量值！(注意，只能读取标量，不能做向量运算)
    dynamic_idx = indices_smem_ref[i, j]
    
    # 返回真正的 HBM 块索引
    return (dynamic_idx, 0)

data_block_spec = pl.BlockSpec(
    block_shape=(128, 128),
    index_map=data_index_map
)
```

### 3. Kernel 函数的签名变化

Kernel 函数的参数列表也会发生变化，预取参数的引用会排在最前面：

```python
def my_sparse_kernel(indices_smem_ref, data_vmem_ref, out_vmem_ref):
    # indices_smem_ref: SMEM 引用
    # data_vmem_ref: VMEM 引用 (此时，DMA 引擎已经根据动态索引把它从 HBM 抓取过来了！)
    
    # 执行计算...
    out_vmem_ref[...] = data_vmem_ref[...] * 2.0
```

## 实战：动态块提取 (Dynamic Block Slicing)

让我们看一个完整的最小示例。我们有一个巨大的 1D 数据数组，和一个包含索引的数组。我们想根据索引，离散地提取数据块并拼接起来。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def dynamic_slice_kernel(indices_ref, data_ref, out_ref):
    # data_ref 已经是我们需要的目标块了
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

## 局限性与终极手段：手动 DMA

`PrefetchScalarGridSpec` 非常方便，但它有一个局限性：**它只能处理每个 Grid 步骤提取固定数量的块的情况。**

如果我们在处理 Ragged Paged Attention，每个请求（Sequence）的页数（Page count）都不一样。有些请求只有 1 页，有些请求有 100 页。这种动态循环次数的场景，连 `PrefetchScalarGridSpec` 的自动 DMA 都无法处理。

在这种硬核场景下，开发者必须：
1. 将 `page_indices` 和 `page_counts` 放入 SMEM。
2. 将 KV Cache 的 HBM 引用（`memory_space=pltpu.HBM`）直接传入 Kernel，**完全放弃 BlockSpec 自动搬运**。
3. 在 Kernel 内部，使用 `jax.lax.while_loop` 根据 SMEM 中的 `page_counts` 进行动态循环。
4. 在循环内部，手动调用 `pltpu.make_async_copy`，根据 SMEM 中的物理页号，构造底层的 DMA 描述符，并发起异步拷贝。

这种极致的手动控制，打破了 TPU 高级抽象的限制，赋予了开发者与硬件底层直接对话的能力。我们将在第 13 章的源码剖析中看到这种终极手段的实际应用。
