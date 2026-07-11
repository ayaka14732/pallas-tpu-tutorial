# 第 2 章：Pallas Hello World

在了解了 TPU 的基本架构后，我们现在来编写第一个 Pallas TPU Kernel。我们将实现一个最简单的操作：向量加法（Vector Add），即 `z = x + y`。

虽然 JAX 的 `jax.jit` 已经能完美处理这种简单操作，但通过手写这个 Kernel，我们将建立起 Pallas 编程的核心心智模型：**Host 端调度与 Device 端执行的分离**。

## JAX Pallas 的编程模型

在 CUDA 编程中，我们写 `__global__` 函数，并在 Host 端用 `<<<grid, block>>>` 启动。在 Pallas 中，模式非常相似，一个 Kernel 通常由两部分组成：

1. **Kernel 函数（Device 侧）**：定义了在底层硬件（VMEM 和寄存器）上对**一个数据块（Block）**执行的具体操作。它操作的是引用（References，即 `jax.experimental.pallas.Ref`），而不是普通的值。
2. **主机端调用（Host 侧）**：使用 `pallas_call` 将高层的 JAX 数组（存在 HBM 中）切分成块，传递给 Kernel 函数，并定义网格（Grid）和块规范（BlockSpec）。

## 编写 Kernel 函数

Kernel 函数的输入参数是底层内存（通常是 VMEM）的引用（Ref）。与普通的 JAX 函数不同，Kernel 函数**没有返回值**，结果是通过就地修改（In-place update）输出引用来实现的。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def add_kernel(x_ref, y_ref, z_ref):
    """
    这是在 TPU 向量核心上执行的代码。
    此时，x_ref, y_ref 指向的数据已经被 DMA 引擎从 HBM 搬运到了 VMEM。
    """
    # 1. 从 VMEM 加载数据到 VREG (向量寄存器)
    # [...] 语法触发了内存加载操作
    x = x_ref[...]
    y = y_ref[...]
    
    # 2. 在向量算术逻辑单元 (VPU) 上执行计算
    # 这个操作会被编译为 TPU 的向量加法指令
    z = x + y
    
    # 3. 将结果写回 VMEM 的输出引用
    # 同样，[...] 语法触发了从 VREG 到 VMEM 的存储操作
    z_ref[...] = z
```

**关键点：** 在 Kernel 内部，你不能直接访问全局数组的长度或形状。Kernel 的视角被限制在当前传入的 `Ref` 的形状（在这个例子中，是一个 128 大小的块）。

## 使用 pallas_call 调度 Kernel

定义了 Kernel 函数后，我们需要告诉 JAX 如何将大数组切分成小块，并在 TPU 上调度执行。这就是 `pallas_call` 的作用。

```python
def vector_add(x: jax.Array, y: jax.Array, block_size: int = 128) -> jax.Array:
    # 确保输入是一维数组且大小相同
    assert x.ndim == 1 and y.ndim == 1
    assert x.shape == y.shape
    seq_len = x.shape[0]
    
    # 计算需要多少个 Block（网格大小）
    # 假设 seq_len 可以被 block_size 整除
    num_blocks = seq_len // block_size
    
    # 定义 BlockSpec：说明每次 Kernel 调用处理多大的数据块
    # 这里我们定义每个输入和输出都处理 (block_size,) 大小的块
    # index_map=lambda i: (i,) 意味着：网格的第 i 步，处理数据的第 i 个块
    block_spec = pl.BlockSpec(
        block_shape=(block_size,),
        index_map=lambda i: (i,)  
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

## 发生了什么？底层执行流程

当你调用 `vector_add(x, y)` 时，TPU 编译器（Mosaic）和运行时在幕后为你做了大量工作：

1. **HBM 到 VMEM 的 DMA 传输**：对于网格中的每一个索引 `i`（从 0 到 `num_blocks - 1`），系统会自动发起异步 DMA 拷贝，将 `x` 和 `y` 的第 `i` 个块（大小为 128）从主存（HBM）拷贝到超快内存（VMEM）。
2. **执行 Kernel**：一旦数据在 VMEM 中就绪，就会调用你编写的 `add_kernel`。此时 `x_ref` 和 `y_ref` 指向的正是刚才拷贝进 VMEM 的那 128 个元素。
3. **VMEM 到 HBM 的 DMA 传输**：`add_kernel` 执行完毕并将结果写入 `z_ref` 后，系统会自动将这块 VMEM 数据拷贝回 HBM 中的输出数组 `z` 的对应位置。

**与 GPU 的对比：** 在 CUDA 中，如果你不显式使用共享内存（Shared Memory），编译器可能会直接从全局内存（Global Memory）流式读取数据。但在 TPU 上，HBM -> VMEM -> VREG 的层级是严格执行的，`pallas_call` 自动为你生成了 HBM 和 VMEM 之间的 DMA 代码。

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
# 注意：第一次运行会触发 JIT 编译，Mosaic 编译器会将 Python 代码降级为 TPU 指令
z_pallas = jax.jit(vector_add)(x, y)
z_jax = x + y

# 验证结果
assert jnp.allclose(z_pallas, z_jax)
print("Success! Pallas kernel matches JAX output.")
```

在下一章中，我们将深入探讨 `Grid` 和 `BlockSpec`，这是 Pallas 中控制数据切分和循环逻辑的核心机制，也是实现复杂算子（如矩阵乘法和卷积）的基础。
