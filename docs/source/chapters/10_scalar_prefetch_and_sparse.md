# 第 10 章：标量预取与动态索引

## 问题背景

在标准的 `pallas_call` 中，`BlockSpec` 的 `index_map` 是一个纯函数——它只接受 grid 索引作为输入，返回固定的块坐标。数据访问模式在编译时就完全确定了。

但很多实际场景需要**数据依赖的索引**：
- KV Cache 的 paged attention：需要根据 page table 查找物理页
- 稀疏注意力：需要根据 mask 跳过某些块
- Ragged batching：不同序列长度不同，需要根据 metadata 确定边界

## TPU 的动态索引挑战

TPU 的向量核心不能直接访问 HBM。数据必须由 DMA 引擎提前搬运到 VMEM。但如果 DMA 引擎不知道目标地址（因为地址依赖于运行时数据），就无法预取。

解决方案：利用**标量核心（Scalar Core）**和**标量内存（SMEM）**。标量核心可以快速读取 SMEM 中的数据，并动态配置 DMA 描述符。

## PrefetchScalarGridSpec

`PrefetchScalarGridSpec` 允许 `index_map` 接受额外的标量参数：

```python
grid_spec = pltpu.PrefetchScalarGridSpec(
    num_scalar_prefetch=1,  # 前 1 个参数是 scalar prefetch
    grid=(num_blocks,),
    in_specs=[
        pl.BlockSpec(
            (BLOCK_SIZE, HEAD_DIM),
            lambda i, page_table_ref: (page_table_ref[i], 0)
        ),
    ],
    out_specs=pl.BlockSpec((BLOCK_SIZE,), lambda i, page_table_ref: (i,)),
)
```

关键点：
- `num_scalar_prefetch=N`：kernel 函数的前 N 个参数是 scalar prefetch 参数
- 这些参数在 `index_map` 中也可以使用
- 它们被预取到 SMEM（不占 VMEM）

## Kernel 函数签名

使用 scalar prefetch 时，kernel 函数签名变化：

```python
def kernel(
    page_table_ref,  # scalar prefetch 参数（SMEM 引用）
    data_ref,        # 普通输入（VMEM 引用，已由 DMA 根据动态索引搬运）
    out_ref,         # 输出（VMEM 引用）
):
    # page_table_ref 驻留在 SMEM
    # data_ref 已经是正确的数据块（DMA 根据 index_map 的动态结果搬运）
    out_ref[...] = data_ref[...]
```

## 与标准 BlockSpec 的对比

| 特性 | 标准 BlockSpec | PrefetchScalarGridSpec |
| :--- | :--- | :--- |
| index_map 输入 | 只有 grid 索引 | grid 索引 + scalar prefetch refs |
| 访问模式 | 编译时确定 | 运行时确定（数据依赖）|
| 适用场景 | 规则的分块访问 | 不规则/稀疏/间接索引 |
| 性能 | 最优 | 略有开销（额外的标量预取）|

## 完整示例：动态块提取

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def gather_kernel(indices_ref, data_ref, out_ref):
    out_ref[...] = data_ref[...]

def dynamic_gather(data: jax.Array, indices: jax.Array, block_size: int = 128):
    num_blocks = indices.shape[0]

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(num_blocks,),
        in_specs=[
            pl.BlockSpec(
                (block_size,),
                lambda i, idx_ref: (idx_ref[i],)  # 动态索引
            ),
        ],
        out_specs=pl.BlockSpec((block_size,), lambda i, idx_ref: (i,)),
    )

    return pl.pallas_call(
        gather_kernel,
        out_shape=jax.ShapeDtypeStruct((num_blocks * block_size,), data.dtype),
        grid_spec=grid_spec,
    )(indices, data)  # indices 作为第一个参数（scalar prefetch）
```

## 手动 DMA 与动态索引

当 `PrefetchScalarGridSpec` 的抽象不够灵活时（如循环内部需要根据运行时计算的偏移量发起 DMA），需要完全手动控制：

```python
def kernel(metadata_ref, k_hbm_ref, v_hbm_ref, out_ref,
           k_buf, v_buf, k_sem, v_sem):
    # metadata_ref: SMEM 引用
    seq_start = metadata_ref[0]
    seq_len = metadata_ref[1]
    num_blocks = seq_len // BLOCK_SIZE

    @pl.loop(0, num_blocks)
    def _(i):
        buf_idx = i % 2
        offset = seq_start + i * BLOCK_SIZE

        # 手动 DMA：从 HBM 的动态位置加载到 VMEM buffer
        pltpu.make_async_copy(
            k_hbm_ref.at[offset:offset+BLOCK_SIZE, :],
            k_buf.at[buf_idx],
            k_sem.at[buf_idx],
        ).start()

        # 等待上一次 DMA 完成
        prev_buf = (i - 1) % 2
        @pl.when(i > 0)
        def _():
            pltpu.make_async_copy(
                k_hbm_ref.at[0:BLOCK_SIZE, :],  # dummy, just for wait
                k_buf.at[prev_buf],
                k_sem.at[prev_buf],
            ).wait()

        # 计算...
```

## GridDimensionSemantics

`pltpu.GridDimensionSemantics` 控制 grid 维度的语义：

```python
pl.pallas_call(
    kernel,
    ...,
    grid=(batch_size, num_heads),
    dimension_semantics=(
        pltpu.GridDimensionSemantics.PARALLEL,
        pltpu.GridDimensionSemantics.ARBITRARY,
    ),
)
```

- `PARALLEL`：该维度的不同迭代之间没有数据依赖，可以被并行化到多核
- `ARBITRARY`：迭代顺序任意，但不保证并行

在 Megacore 配置中，`PARALLEL` 维度会被自动分配到不同的 TPU 核心。

## 与 GPU 的对比

GPU 上的动态索引是自然的——每个线程可以用任意计算出的地址访问全局内存。但代价是：
- 不连续访问导致 memory coalescing 失败
- 需要手动管理共享内存中的 gather/scatter
- cache miss 导致延迟不可预测

TPU 上的动态索引需要通过 `PrefetchScalarGridSpec` 或手动 DMA 显式声明。好处是：
- DMA 引擎处理所有数据搬运，不存在 coalescing 问题
- 编译器可以提前调度 DMA，隐藏延迟
- 数据到达 VMEM 后的访问是完全确定的

## 在 RPA v3 中的应用

RPA v3 kernel 大量使用 scalar prefetch 和手动 DMA：

1. **Scalar prefetch**：传入每个序列的 metadata（起始位置、长度、page indices）
2. **手动 DMA**：根据 page table 动态加载 KV cache 页到 VMEM
3. **双缓冲**：在循环中交替使用两个 VMEM buffer，实现 DMA 和计算重叠
4. **动态循环次数**：不同序列的 KV 长度不同，循环次数由 metadata 决定

这种模式是 TPU 上处理不规则数据访问的标准方法。
