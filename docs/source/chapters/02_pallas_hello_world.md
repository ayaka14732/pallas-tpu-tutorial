# 第 2 章：Pallas Hello World

在了解了 TPU 的基本架构后，我们现在来编写第一个 Pallas TPU Kernel。我们将实现一个最简单的操作：向量加法（Vector Add），即 `z = x + y`。

## JAX Pallas 的编程模型

在 Pallas 中，一个 Kernel 通常由两部分组成：

1. **Kernel 函数（Kernel Function）**：定义了在底层硬件（VMEM 和寄存器）上对**一个数据块（Block）**执行的具体操作。它操作的是引用（References，即 `jax.experimental.pallas.Ref`），而不是普通的值。
2. **主机端调用（Host-side Call）**：使用 `pallas_call` 将高层的 JAX 数组（存在 HBM 中）切分成块，传递给 Kernel 函数，并定义网格（Grid）和块规范（BlockSpec）。

## 编写 Kernel 函数

Kernel 函数的输入参数是底层内存（VMEM）的引用（Ref）。与普通的 JAX 函数不同，Kernel 函数**没有返回值**，结果是通过就地修改（In-place update）输出引用来实现的。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def add_kernel(x_ref, y_ref, z_ref):
    # x_ref, y_ref, z_ref 是指向 VMEM 的引用
    
    # 1. 从 VMEM 加载数据到寄存器
    x = x_ref[...]
    y = y_ref[...]
    
    # 2. 在向量寄存器上执行计算
    z = x + y
    
    # 3. 将结果写回 VMEM 的输出引用
    z_ref[...] = z
```

在这个简单的例子中，`x_ref[...]` 会触发从 VMEM 到向量寄存器的加载操作，`x + y` 在 TPU 的向量算术逻辑单元（VPU）上执行，最后 `z_ref[...]` 将寄存器中的结果存回 VMEM。

## 使用 pallas_call 调度 Kernel

定义了 Kernel 函数后，我们需要告诉 JAX 如何将大数组切分成小块，并在 TPU 上调度执行。这就是 `pallas_call` 的作用。

```python
def vector_add(x: jax.Array, y: jax.Array, block_size: int = 128) -> jax.Array:
    # 确保输入是一维数组且大小相同
    assert x.ndim == 1 and y.ndim == 1
    assert x.shape == y.shape
    seq_len = x.shape[0]
    
    # 计算需要多少个 Block（网格大小）
    num_blocks = seq_len // block_size
    
    # 定义 BlockSpec：说明每次 Kernel 调用处理多大的数据块
    # 这里我们定义每个输入和输出都处理 (block_size,) 大小的块
    block_spec = pl.BlockSpec(
        block_shape=(block_size,),
        index_map=lambda i: (i,)  # 将一维网格索引 i 映射到数据块的第 i 个切片
    )
    
    # 使用 pallas_call 包装 kernel
    kernel = pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[block_spec, block_spec],  # 对应 x_ref, y_ref
        out_specs=block_spec,               # 对应 z_ref
        grid=(num_blocks,)                  # 定义一维网格
    )
    
    # 执行 Kernel（此时数据在 HBM 中）
    return kernel(x, y)
```

## 发生了什么？

当你调用 `vector_add(x, y)` 时，TPU 编译器（Mosaic）和运行时在幕后为你做了大量工作：

1. **HBM 到 VMEM 的 DMA 传输**：对于网格中的每一个索引 `i`，系统会自动发起异步 DMA 拷贝，将 `x` 和 `y` 的第 `i` 个块（大小为 128）从主存（HBM）拷贝到超快内存（VMEM）。
2. **执行 Kernel**：一旦数据在 VMEM 中就绪，就会调用你编写的 `add_kernel`。此时 `x_ref` 和 `y_ref` 指向的正是刚才拷贝进 VMEM 的那 128 个元素。
3. **VMEM 到 HBM 的 DMA 传输**：`add_kernel` 执行完毕并将结果写入 `z_ref` 后，系统会自动将这块 VMEM 数据拷贝回 HBM 中的输出数组 `z` 的对应位置。

由于 TPU 是顺序执行机器，网格索引 `i=0, 1, 2...` 会依次执行。在后续章节中，我们将学习如何利用**流水线（Pipelining）**让第 `i` 步的计算与第 `i+1` 步的 DMA 拷贝重叠，从而实现极致性能。

## 完整代码与测试

你可以将以下代码保存并运行，验证你的第一个 Pallas Kernel：

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def add_kernel(x_ref, y_ref, z_ref):
    z_ref[...] = x_ref[...] + y_ref[...]

def vector_add(x, y):
    block_size = 128
    num_blocks = x.shape[0] // block_size
    
    block_spec = pl.BlockSpec(lambda i: (i,), (block_size,))
    
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[block_spec, block_spec],
        out_specs=block_spec,
        grid=(num_blocks,)
    )(x, y)

# 创建测试数据
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (1024,))
y = jax.random.normal(key, (1024,))

# 运行 Kernel
# 注意：第一次运行会触发 JIT 编译
z_pallas = jax.jit(vector_add)(x, y)
z_jax = x + y

# 验证结果
jnp.allclose(z_pallas, z_jax)
print("Success! Pallas kernel matches JAX output.")
```

在下一章中，我们将深入探讨 `Grid` 和 `BlockSpec`，这是 Pallas 中控制数据切分和循环逻辑的核心机制。
