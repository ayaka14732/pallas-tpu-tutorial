# 第 3 章：Grid 与 BlockSpec 深入解析

在上一章的 Hello World 中，我们接触了 `grid` 和 `BlockSpec`。这两个概念是 Pallas 编程模型的核心，它们决定了数据如何从 HBM 被切分成小块加载到 VMEM，以及 Kernel 是如何循环执行的。

本章我们将深入解析它们的语义，并学习如何处理更复杂的切分模式。理解这一章是编写任何非平凡 Kernel（如 MatMul 或 Attention）的前提。

## 循环的心智模型 (The Mental Model)

在 Pallas 中，你可以把 `pallas_call` 想象成一个巨大的、由硬件协助执行的 `for` 循环。

当你这样调用时：
```python
pl.pallas_call(
    kernel_fn,
    grid=(X, Y),
    in_specs=[in_spec],
    out_specs=out_spec
)(input_array)
```

它的行为在逻辑上等价于以下 Python 代码：

```python
for i in range(X):
    for j in range(Y):
        # 1. 根据 in_spec 计算输入切片位置，并触发 HBM -> VMEM 的 DMA 拷贝
        in_slice = in_spec.compute_slice(i, j)
        in_ref = input_array[in_slice]  # 此时 in_ref 指向 VMEM
        
        # 2. 根据 out_spec 计算输出切片位置
        out_slice = out_spec.compute_slice(i, j)
        out_ref = output_array[out_slice] # 此时 out_ref 指向 VMEM
        
        # 3. 执行 Kernel（此时数据在 VMEM 中，由 VPU/MXU 处理）
        kernel_fn(in_ref, out_ref)
        
        # 4. 触发 VMEM -> HBM 的 DMA 拷贝，将 out_ref 的内容写回 output_array
```

**与 GPU 的核心区别：**
在 CUDA 中，Grid 维度 `(X, Y)` 代表的是并发启动的线程块（Thread Blocks）。它们在不同的流多处理器（SM）上并行执行，执行顺序是**未定义的**。
但在 TPU Pallas 中，这个网格 `(X, Y)` 保证是**按字典序严格顺序执行**的（即先遍历 `i`，再遍历 `j`）。这种顺序性保证允许我们在不同的 Grid 迭代之间传递状态（通过 Scratch Buffer），这在 GPU 上通常需要昂贵的全局内存原子操作或 Kernel 拆分。

## BlockSpec 的组成

`BlockSpec` 由两部分组成：
1. `block_shape`：一个元组，定义了加载到 VMEM 中的数据块的形状。
2. `index_map`：一个函数，它接收网格索引（Grid indices），返回块索引（Block indices）。

### 块索引 (Block Indices) vs 元素切片 (Element Slices)

理解 `index_map` 的关键在于：它返回的是**块索引**，而不是实际的元素偏移量。Pallas 会自动将块索引乘以 `block_shape`，从而得到真实的元素切片。

例如，如果输入数组形状是 `(1024, 1024)`，`block_shape=(128, 128)`：
如果 `index_map(i, j)` 返回 `(2, 3)`，那么实际提取的切片是：
- 第 0 维：`2 * 128` 到 `3 * 128`，即 `256:384`
- 第 1 维：`3 * 128` 到 `4 * 128`，即 `384:512`

### 常见的 Index Map 模式

#### 1. 一一映射 (1:1 Mapping)
最常见的模式，网格索引直接对应块索引。适用于逐元素（Element-wise）操作。
```python
grid = (8, 8)
block_shape = (128, 128)
index_map = lambda i, j: (i, j)
```

#### 2. 广播 (Broadcasting)
如果某个输入在某个网格维度上不需要切分（即所有迭代都使用同一块数据），可以让 `index_map` 在该维度返回 0。
```python
# 假设我们要在矩阵每一行上加一个偏置向量
grid = (8, 8) # 遍历矩阵的行和列
# 矩阵的 index_map
matrix_map = lambda i, j: (i, j)
# 偏置向量的 index_map：它不随行索引 i 变化，始终使用整块向量
bias_map = lambda i, j: (0, j) 
```

#### 3. 降维/挤压 (Squeezing Dimensions)
有时我们希望加载的数据块维数比原数组少。例如，从 2D 矩阵中提取 1D 的行。
在 Pallas 中，可以通过在 `block_shape` 中使用 `None` 来实现。

```python
# input_array shape: (8, 128)
# 我们想每次提取一行，即 (128,) 的一维数组
block_shape = (None, 128)
index_map = lambda i: (i, 0)
```
此时传入 Kernel 的 `in_ref` 形状将是 `(128,)`，而不是 `(1, 128)`。这对于避免 TPU 上的"单元素维度"性能惩罚非常重要。

## 越界填充 (Out-of-Bounds Padding)

如果原数组的大小不能被 `block_shape` 整除怎么办？在 CUDA 中，我们需要在 Kernel 内部写大量的 `if (tid < N)` 边界检查代码。

Pallas 提供了一个非常优雅的特性：它会**自动处理越界读取**。当切片超出原数组边界时，Pallas 会自动在 VMEM 中填充默认值（通常是 0）。

例如，数组大小为 100，`block_shape=(128,)`。
当加载这块数据时，VMEM 中的 `ref` 大小依然是 128，其中前 100 个元素是真实数据，后 28 个元素被填充为 0。

**注意：** 越界写入（Out-of-bounds writes）目前会导致静默丢弃（Silently dropped），写入超出边界的部分不会影响原数组，但这是一种不推荐的做法，可能在未来版本中报错。

## TPU 限制与最佳实践

在 TPU 上使用 BlockSpec 时，必须牢记我们在第 1 章提到的硬件限制：

1. **Rank 限制**：在 TPU 上，块的秩（Rank，即维度数）必须至少为 1。不支持 0D（标量）的 BlockSpec。如果需要标量，应使用 `PrefetchScalarGridSpec`（见第 10 章）。
2. **Tile 对齐**：`block_shape` 的**最后两个维度**必须能被 8 和 128 整除，或者等于原数组对应维度的完整大小。
   - 合法：`(256, 128)`, `(8, 256)`
   - 非法：`(10, 100)`（除非原数组的最后两维就是 10 和 100，此时必须一次性加载整个维度）。

掌握了 Grid 和 BlockSpec，你就可以自由地将任何大型张量切片送入 TPU 的超快 VMEM 中。下一章，我们将详细探讨 TPU 的内存空间标注，这是实现复杂 Kernel（如 FlashAttention）不可或缺的工具。
