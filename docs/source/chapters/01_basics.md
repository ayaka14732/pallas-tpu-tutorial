# 第 1 章：Pallas 开发基础

## 开发环境配置

### 在本地开发

本教程讨论的是 TPU kernel 开发，但日常学习和调试不必每一步都依赖真实 TPU。TPU 资源通常有限，也不适合长期保持空闲实例等待试验。在学习本教程中不涉及 TPU 的部分时，可以先在本地开发，使用解释模式在 CPU 上运行；等到编写 Pallas TPU 相关代码时，再使用 TPU 运行。

在本机只需安装 CPU 版本的 JAX：

```bash
pip install -U jax
```

解释模式的具体用法会在本章后文介绍。

### 在 GCP 上开发

在 GCP 上开发 Pallas TPU 程序时，首先需要创建一台 Cloud TPU VM。

使用 uv 可以无需管理员权限，简便地下载最新版本的 Python 3.14。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

在工作目录使用 uv 创建 venv，安装 Python 3.14 和 JAX：

```bash
$HOME/.local/bin/uv venv --python 3.14 --seed venv
source venv/bin/activate
pip install -U "jax[tpu]"
```

安装完成后，可以用下面的代码确认当前进程能够访问 TPU，并查看 TPU 的相关信息：

```python
from jax.experimental.pallas import tpu as pltpu

print(pltpu.get_tpu_info())
```

:::{note}
uv 本身是 Python 的包管理器，但本教程仍然使用传统的 pip 包管理器，uv 仅用于安装最新版本的 Python。
:::

### 在 Colab 上使用

Colab 的运行时通常已经预装 JAX，但预装版本通常非常过时，不包含 Pallas TPU 后端所需的最新改动。因此，在 Colab 中需要先升级 JAX 版本，再重启运行时，让后续代码使用新安装的版本。

在第一个 cell 中执行以下代码以升级 JAX 版本：

```bash
!pip install -q -U "jax[tpu]"
```

安装完成后，在第二个 cell 中运行：

```python
import os
os.kill(os.getpid(), 9)
```

这行代码会主动结束当前 Python 进程。Colab 会自动重新启动运行时；重启后再继续执行后面的教程代码即可。

需要注意的是，Colab 目前只有 TPU v5e，不含 SparseCore，因此无法运行本教程中与 SparseCore 相关的代码。

## Pallas 编程模型

Pallas kernel 由两部分组成：

1. **Kernel 函数**：在 TPU 上执行，操作的是 `Ref`（内存引用），而非普通 JAX 数组。Kernel 函数没有返回值，结果通过写入输出 `Ref` 来传递。
2. **`pallas_call` 调度**：在 host 端定义 grid、BlockSpec、scratch shapes 等，告诉编译器如何切分数据并调度 kernel。

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

## 最简单的 Pallas kernel

### 向量逐元素相加的 Pallas 实现

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def add_kernel(x_ref: jax.Ref, y_ref: jax.Ref, z_ref: jax.Ref) -> None:
    z_ref[...] = x_ref[...] + y_ref[...]

def vector_add(x: jax.Array, y: jax.Array) -> jax.Array:
    block_size = 1024  # TPU 上可以用大块，因为 VMEM 有 16MB+
    num_blocks = pl.cdiv(x.shape[0], block_size)  # `pl.cdiv` 作除法并向上取整

    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[
            pl.BlockSpec(block_shape=(block_size,), index_map=lambda i: (i,)),
            pl.BlockSpec(block_shape=(block_size,), index_map=lambda i: (i,)),
        ],
        out_specs=pl.BlockSpec(block_shape=(block_size,), index_map=lambda i: (i,)),
        grid=(num_blocks,),
        interpret=True,  # 在 CPU 上模拟执行
    )(x, y)

x = jnp.full((1024,), 1.0, dtype=jnp.float32)
y = jnp.full((1024,), 2.0, dtype=jnp.float32)
result = vector_add(x, y)
print('result.shape:', result.shape)  # (1024,)
assert jnp.allclose(result, 3.0)
```

### 执行流程

当 `vector_add(x, y)` 被调用时，编译器生成的代码等价于：

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

编译器还会自动插入**流水线**：当第 _i_ 次迭代在计算时，第 _i_+1 次迭代的 DMA 加载已经在后台进行。这就是为什么即使是这个简单的 kernel，TPU 也能接近 HBM 带宽上限。

上面所说的 DMA，以及 HBM、VMEM、VREG 的相关内容将在第四章详细介绍。读者目前只需理解为：DMA 是指在 TPU 硬件的不同内存区域传输数据；HBM、VMEM、VREG 都是 TPU 硬件的内存区域，数据需要从 HBM 传递到 VMEM，再从 VMEM 传递到 VREG 才能进行计算。

## 解释模式

其中，代码中的 `interpret=True` 表示在 CPU 上模拟 kernel 执行。如果你在本地开发，没有真实的 TPU 环境，可以使用这种方法执行代码。解释模式有两种开启方法：

```python
# 方法 1：全局开启
pltpu.set_tpu_interpret_mode(True)

# 方法 2：在单个 pallas_call 中开启
result = pl.pallas_call(
    kernel_fn,
    ...,
    interpret=pltpu.InterpretParams(),  # 在 CPU 上模拟执行
)(inputs)
```

解释模式下，所有 DMA 操作会被模拟为普通的内存拷贝，信号量操作会被模拟为计数器。这对于验证 kernel 逻辑的正确性非常有用，但不能反映真实的性能特征。

:::{warning}
在旧的代码中，你可能会看到 `interpret=True` 的写法。那是旧的 CPU 解释模式，不能很好地模拟 TPU 硬件，应该避免使用。
:::

## 第二个 Pallas kernel

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

is_tpu_available = jax.devices()[0].platform == "tpu"

def kernel(x_ref: jax.Ref, o_ref: jax.Ref) -> None:
    # 获取当前 program 在各个 grid 维度上的索引
    col = pl.program_id(0)
    row = pl.program_id(1)

    # 获取 grid 各个维度的长度
    n = pl.num_programs(0)
    m = pl.num_programs(1)

    @pl.when((col != n - 1) & (row != m - 1))  # 过滤掉行号为 2 的行或列号为 2 的列
    def _():
        pl.debug_print("Executing grid ({}, {})", col, row)

    o_ref[...] = x_ref[...] * 2.0

def fn(x: jax.Array) -> jax.Array:
    N, M = x.shape
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec(block_shape=(8, 128), index_map=lambda i, j: (i, j))],
        out_specs=pl.BlockSpec(block_shape=(8, 128), index_map=lambda i, j: (i, j)),
        grid=(pl.cdiv(N, 8), pl.cdiv(M, 128)),
        interpret=False if is_tpu_available else pltpu.InterpretParams(),  # 在 CPU 上模拟执行
    )(x)

x = jnp.full((24, 384), 7.0, dtype=jnp.float32)
result = fn(x)
print('result.shape:', result.shape)
assert jnp.allclose(result, 14.0)
```

输出：

```
result.shape: (24, 384)
Executing grid (0, 0)
Executing grid (0, 1)
Executing grid (1, 0)
Executing grid (1, 1)
```

这个 kernel 有几个要点：

1. 在 kernel 内部，可以通过 `pl.program_id(axis)` 获取当前 grid 索引，通过 `pl.num_programs(axis)` 获取 grid 大小；
2. 可以使用 `pl.when` 在 kernel 内部进行条件执行，具体用法将在第 3 章讲解。
3. 可以使用 `pl.debug_print` 在 kernel 内部打印调试信息。但要注意，`debug_print` 会引入同步点，影响性能。仅用于调试，生产代码中应移除。

:::{warning}
此示例中的 `pl.debug_print` 在真实 TPU VM 上无法直接起作用，读者需要参考 [jax-ml/jax#25192 (comment)](https://github.com/jax-ml/jax/issues/25192#issuecomment-4971797607) 中的方法解决。
:::
