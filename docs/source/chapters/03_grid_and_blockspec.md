# 第 3 章：Grid 与 BlockSpec

## 循环模型

`pallas_call` 的 grid 参数定义了一个多维循环。对于 `grid=(M, N)`，kernel 会被调用 `M × N` 次，按字典序遍历所有 `(i, j)` 组合。每次调用时，`BlockSpec` 决定了哪些数据块被加载到 VMEM。

```python
pl.pallas_call(
    kernel_fn,
    grid=(M, N),
    in_specs=[block_spec_x],
    out_specs=block_spec_z,
)(x)
```

逻辑上等价于：

```python
for i in range(M):
    for j in range(N):
        x_block = x[block_spec_x.compute_slice(i, j)]  # DMA: HBM -> VMEM
        z_block = z[block_spec_z.compute_slice(i, j)]   # VMEM 中的输出区域
        kernel_fn(x_block_ref, z_block_ref)             # 执行
        # DMA: VMEM -> HBM（自动）
```

## BlockSpec 的两个组成部分

```python
pl.BlockSpec(
    block_shape=(BM, BN),       # 每次加载的块大小
    index_map=lambda i, j: (i, j)  # grid 索引 -> 块索引的映射
)
```

`index_map` 返回的是**块索引**（block index），不是元素偏移。实际的元素切片 = 块索引 × block_shape。

例如：数组形状 `(1024, 512)`，`block_shape=(128, 256)`，`index_map` 返回 `(2, 1)` 时，实际切片为 `[256:384, 256:512]`。

## 常见 index_map 模式

**逐块遍历**（最常见）：
```python
# 矩阵乘法：C[i,j] = A[i,:] @ B[:,j]
# A 的行随 i 变化，B 的列随 j 变化
in_specs_A = pl.BlockSpec((BM, BK), lambda i, j: (i, 0))  # A 的列索引固定为 0
in_specs_B = pl.BlockSpec((BK, BN), lambda i, j: (0, j))  # B 的行索引固定为 0
out_specs  = pl.BlockSpec((BM, BN), lambda i, j: (i, j))
```

**广播**（某个维度不随 grid 变化）：
```python
# bias 对所有行广播
bias_spec = pl.BlockSpec((BN,), lambda i, j: (j,))  # 只依赖 j，不依赖 i
```

**累加模式**（多次写入同一输出块）：
```python
# 矩阵乘法的 K 维累加：grid=(M//BM, N//BN, K//BK)
out_specs = pl.BlockSpec((BM, BN), lambda i, j, k: (i, j))  # 输出不依赖 k
# 每次 k 迭代都写入同一个输出块，实现累加
```

## Squeezed 维度

当 `block_shape` 中某个维度为 `None`（或使用 `pl.Squeezed()`），该维度会从传入 kernel 的 Ref 中被压缩掉。这用于处理"沿某个维度逐片处理"的场景。

```python
# 输入形状 (batch, seq_len, hidden)
# 想逐 batch 处理，kernel 内部只看到 (seq_len, hidden)
pl.BlockSpec(
    block_shape=(None, seq_len, hidden),  # None = 该维度被 squeeze
    index_map=lambda b: (b, 0, 0)
)
# kernel 收到的 ref 形状为 (seq_len, hidden)，而非 (1, seq_len, hidden)
```

等价写法：`pl.BlockSpec(block_shape=(pl.Squeezed(), seq_len, hidden), ...)`

## 越界处理（OOB Behavior）

当 block 的边界超出数组实际大小时，TPU 的行为是：
- **读取**：越界部分填充为 0
- **写入**：越界部分被丢弃

这意味着你不需要手动处理边界条件。即使数组大小不是 block_size 的整数倍，也可以安全地使用 `cdiv(size, block_size)` 作为 grid 大小。

```python
# 数组大小 1000，block_size 128
# grid = cdiv(1000, 128) = 8
# 最后一个块实际只有 1000 - 7*128 = 104 个有效元素
# 读取时，后 24 个位置自动填 0
# 写入时，后 24 个位置的写入被忽略
```

## 其他索引模式

除了标准的 `BlockSpec`，Pallas 还支持以下索引模式：

### pl.Element

逐元素访问。kernel 收到的 Ref 是标量（0D）。通常不用于 TPU（因为标量操作效率极低），但在某些控制流场景中有用。

### pl.BoundedSlice

安全的有界切片，编译器可以利用边界信息进行优化：

```python
pl.BlockSpec(
    block_shape=(pl.BoundedSlice(size=128, bound=array_size),),
    index_map=lambda i: (i,)
)
```

### pl.Indirect

间接索引。允许通过另一个数组来决定访问哪些块。这是实现 sparse kernel 的基础：

```python
# indices 数组包含要访问的块索引
pl.BlockSpec(
    block_shape=(128,),
    index_map=lambda i: (pl.Indirect(indices_ref, i),)
)
```

## no_block_spec 与整体传入

`pl.no_block_spec` 表示该输入/输出不参与自动切分，整个数组作为一个 Ref 传入 kernel。

```python
# x 按块切分，metadata 整体传入
in_specs = [
    pl.BlockSpec((128,), lambda i: (i,)),  # x 按块切分
    pl.no_block_spec,                       # metadata 整体传入
]
```

## BlockSpec 的 memory_space 参数

可以指定数据应该驻留在哪个内存空间：

```python
# 数据驻留在 HBM，kernel 内部手动管理 DMA
pl.BlockSpec(memory_space=pltpu.HBM)

# 数据预加载到 VMEM（自动 DMA）
pl.BlockSpec(memory_space=pltpu.VMEM)

# 数据放在 SMEM（用于标量/索引数据）
pl.BlockSpec(memory_space=pltpu.SMEM)
```

当使用 `memory_space=pltpu.HBM` 时，kernel 收到的 Ref 指向 HBM。此时你需要自己调用 `pltpu.make_async_copy` 来搬运数据。这是实现手动流水线的基础。

## Scratch Shapes

Scratch buffer 是 kernel 内部使用的临时内存，不对应任何输入或输出：

```python
scratch_shapes = [
    pltpu.VMEM((1024, 128), jnp.float32),     # VMEM 中的临时缓冲区
    pltpu.SMEM((16,), jnp.int32),             # SMEM 中的索引缓冲区
    pltpu.SemaphoreType.DMA((2,)),            # DMA 信号量
]
```

Scratch buffer 作为额外的 Ref 参数传入 kernel（在输入和输出之后）：

```python
def kernel(x_ref, y_ref, o_ref, scratch_vmem_ref, scratch_smem_ref, sem_ref):
    # scratch_vmem_ref 是 VMEM 中的临时空间
    # scratch_smem_ref 是 SMEM 中的临时空间
    # sem_ref 是 DMA 信号量
    ...
```

## 实践：2D 矩阵加法

```python
def matadd_kernel(x_ref, y_ref, z_ref):
    z_ref[...] = x_ref[...] + y_ref[...]

def matrix_add(x, y):
    M, N = x.shape
    BM, BN = 256, 256

    block_spec = pl.BlockSpec(
        block_shape=(BM, BN),
        index_map=lambda i, j: (i, j)
    )

    return pl.pallas_call(
        matadd_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[block_spec, block_spec],
        out_specs=block_spec,
        grid=(pl.cdiv(M, BM), pl.cdiv(N, BN)),
    )(x, y)
```
