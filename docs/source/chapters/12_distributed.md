# 第 12 章：分布式 Kernel 与多核编程

TPU 的一大独特优势是其原生的高速互联网络。在 TPU Pod 中，多个 TPU 芯片通过 ICI（Inter-Chip Interconnect）直接相连，带宽极高、延迟极低。Pallas 允许我们在 Kernel 内部直接使用这些互联通信原语。

本章将介绍 Megacore（单芯片双核）编程和跨芯片分布式 Kernel。

## Megacore：单芯片双核

从 TPU v4 开始，每个 TPU 芯片通常包含 2 个 TensorCore。默认情况下，它们被抽象为一个设备。但通过 `dimension_semantics`，我们可以将 Grid 的某个轴并行化到两个核心上。

```python
pl.pallas_call(
    kernel_fn,
    grid=(8, 4),
    compiler_params=pltpu.CompilerParams(
        dimension_semantics=["parallel", "arbitrary"]
    ),
)
```

在这个例子中：
- 第一个维度（大小 8）被标记为 `parallel`，编译器会尝试将其分配到 2 个核心上（每个核心处理 4 个迭代）。
- 第二个维度（大小 4）被标记为 `arbitrary`，在每个核心内部顺序执行。

### 使用规则

- 只有 `parallel` 维度可以被分配到多个核心。
- `parallel` 维度必须出现在 `arbitrary` 维度之前。
- 如果 `parallel` 维度的大小不是核心数的倍数，负载可能不均衡。

## core_map：显式多核编程

对于需要核心间通信的场景（如 AllReduce），`dimension_semantics` 的隐式并行化不够灵活。Pallas 提供了 `pl.core_map` 来显式地在每个核心上运行不同的代码。

```python
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def per_core_kernel(x_ref, y_ref):
    # 这个 Kernel 在每个核心上独立执行
    # 可以通过 core_map 的上下文获取当前核心 ID
    y_ref[...] = x_ref[...] * 2.0

# 在 shard_map 内部使用 core_map
@jax.jit
def distributed_fn(x):
    def body(x_shard):
        return pl.core_map(
            per_core_kernel,
            out_shape=jax.ShapeDtypeStruct(x_shard.shape, x_shard.dtype),
            mesh=pltpu.create_tensorcore_mesh("core")
        )(x_shard)
    
    mesh = jax.sharding.Mesh(jax.devices(), "devices")
    return jax.shard_map.shard_map(
        body, mesh=mesh, in_specs=..., out_specs=...
    )(x)
```

## 跨芯片通信原语

在 Pallas Kernel 内部，你可以使用以下通信原语：

### 1. Remote DMA Copy

使用 `pltpu.make_async_remote_copy` 在芯片之间直接搬运数据：

```python
# 将本地 VMEM 中的数据发送到远程芯片的 VMEM
remote_copy = pltpu.make_async_remote_copy(
    src_ref=local_vmem_ref,
    dst_ref=remote_vmem_ref,
    send_sem=send_semaphore,
    recv_sem=recv_semaphore,
    device_id=target_device_id
)
remote_copy.start()
remote_copy.wait()
```

### 2. 集合通信 (Collective Operations)

Pallas 支持在 Kernel 内部调用集合通信操作：
- `lax.ppermute`：环形排列（Ring permutation）
- `lax.psum`：全归约求和（All-reduce sum）
- `lax.all_gather`：全收集（All-gather）

这些操作利用 TPU 的 ICI 网络，延迟远低于通过 HBM 中转。

## 实战：分布式矩阵乘法

在大规模模型并行中，矩阵乘法通常需要跨多个 TPU 芯片进行。一种常见的模式是：
1. 每个芯片持有权重矩阵的一部分（列并行或行并行）。
2. 输入通过 All-Gather 收集到每个芯片。
3. 每个芯片计算部分结果。
4. 通过 Reduce-Scatter 将结果归约回去。

在 Pallas 中，这些通信操作可以与计算**重叠**执行，进一步隐藏通信延迟。

## 与 JAX 分布式原语的关系

Pallas 的分布式 Kernel 通常嵌套在 JAX 的高级分布式原语中：
- `jax.shard_map`：将数据分片到多个设备。
- `jax.experimental.custom_partitioning`：自定义 XLA 的分区策略。

Pallas 的分布式 Kernel 提供了比 XLA 自动分区更细粒度的控制，适合需要极致性能的场景。
