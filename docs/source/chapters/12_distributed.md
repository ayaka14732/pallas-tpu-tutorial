# 第 12 章：分布式 Kernel 与多核编程

TPU 的一大独特优势是其原生的高速互联网络。在 TPU Pod 中，多个 TPU 芯片通过 **ICI（Inter-Chip Interconnect）** 直接相连，构建成一个巨大的 3D 环面（Torus）或多维拓扑结构。ICI 的带宽极高、延迟极低，使得跨芯片通信几乎和本地内存访问一样快。

在传统的 JAX 编程中，分布式逻辑由 XLA 编译器自动处理（例如通过 `jax.pmap` 或 `jax.sharding`）。但在 Pallas 中，我们被允许在 Kernel 内部**直接**使用这些互联通信原语，实现极致的性能调优。

本章将介绍 Megacore（单芯片双核）编程和跨芯片分布式 Kernel。

## Megacore：单芯片双核的抽象

从 TPU v4 开始，硬件架构发生了一个重要变化：每个 TPU 芯片实际上包含 **2 个独立的 TensorCore**。
默认情况下，JAX 和 XLA 会将这两个核心抽象为一个单一的逻辑设备（Megacore）。

但在 Pallas 层面，如果你使用普通的顺序执行 Grid，代码只会在其中一个核心上运行，导致一半的算力闲置！为了利用两个核心，我们需要使用 `dimension_semantics`。

### 使用 `dimension_semantics` 隐式并行化

我们可以将 Grid 的某个轴标记为并行，编译器会自动将其分配到两个核心上。

```python
pl.pallas_call(
    kernel_fn,
    grid=(8, 4),
    compiler_params=pltpu.CompilerParams(
        # 第 0 维 (大小 8) 在两个核心上并行执行
        # 第 1 维 (大小 4) 在每个核心内部顺序执行
        dimension_semantics=["parallel", "arbitrary"]
    ),
)
```

**使用规则与限制：**
- 只有标记为 `parallel` 的维度可以被分配到多个核心。
- `parallel` 维度必须出现在 `arbitrary`（或默认的 `unroll`）维度之前。
- **负载均衡**：如果 `parallel` 维度的大小不是核心数（通常是 2）的倍数，就会出现一个核心在计算，另一个核心在等待的情况。

## `core_map`：显式多核编程

对于需要核心间通信的复杂场景（如自定义的 AllReduce），隐式的 `dimension_semantics` 就不够用了。Pallas 提供了 `pl.core_map` 来显式地在每个核心上运行代码，这非常类似于 SPMD（单程序多数据）编程模型。

```python
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def per_core_kernel(x_ref, y_ref):
    # 这个 Kernel 在每个核心上独立执行
    # 可以通过 core_map 的上下文获取当前核心 ID，执行不同的逻辑
    core_id = pltpu.current_core_index()
    if core_id == 0:
        y_ref[...] = x_ref[...] * 2.0
    else:
        y_ref[...] = x_ref[...] + 1.0

# 在 shard_map 内部使用 core_map
@jax.jit
def distributed_fn(x):
    def body(x_shard):
        return pl.core_map(
            per_core_kernel,
            out_shape=jax.ShapeDtypeStruct(x_shard.shape, x_shard.dtype),
            mesh=pltpu.create_tensorcore_mesh("core")
        )(x_shard)
    
    # 结合 JAX 的设备分片
    mesh = jax.sharding.Mesh(jax.devices(), "devices")
    return jax.shard_map.shard_map(
        body, mesh=mesh, in_specs=..., out_specs=...
    )(x)
```

## 跨芯片通信原语：直接调用 ICI

在 Pallas Kernel 内部，你可以使用底层原语直接控制 ICI 网络。这在 GPU 编程中相当于在 CUDA Kernel 内部直接调用 NVLink/NCCL 原语（这通常是非常困难或不被推荐的，但在 TPU 上是常态）。

### 1. 集合通信 (Collective Operations)

Pallas 支持在 Kernel 内部直接调用集合通信操作：
- `lax.ppermute`：环形排列（Ring permutation），将数据发送到拓扑结构中的下一个节点。
- `lax.psum`：全归约求和（All-reduce sum）。
- `lax.all_gather`：全收集（All-gather）。

这些操作直接在向量寄存器和 ICI 网络之间发生，延迟远低于将数据写回 HBM 再由主机端发起通信。

### 2. 远程 DMA 拷贝 (Remote DMA Copy)

这是最底层的通信方式。使用 `pltpu.make_async_remote_copy`，你可以指示 DMA 引擎将本地 VMEM 中的数据**直接**推送到远程芯片的 VMEM 中！

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
# ... 可以在这里执行其他计算，隐藏网络延迟 ...
remote_copy.wait()
```

## 实战：通信与计算重叠的分布式矩阵乘法

在大规模模型并行（如张量并行 Tensor Parallelism）中，矩阵乘法通常需要跨多个 TPU 芯片进行。

标准的 Megatron-LM 模式是：
1. All-Gather 收集输入。
2. 本地计算 MatMul。
3. Reduce-Scatter 归约结果。

这种模式的通信和计算是串行的。在 Pallas 中，我们可以编写一个**通信与计算完全重叠（Overlapped）**的分布式 Kernel：

1. **流水线启动**：发起对相邻芯片的 `ppermute`（或远程 DMA），请求下一块权重。
2. **计算循环**：
   - 等待当前权重块的通信完成。
   - **同时**：发起下一块权重的通信请求。
   - **同时**：使用 MXU 计算当前权重块与本地输入的矩阵乘法。
3. 这样，ICI 网络的传输时间被完美地隐藏在 MXU 的计算时间之中。

这种将计算密集型算子（MatMul）与网络通信深度融合的定制 Kernel，是支撑谷歌训练万亿参数大模型的基础设施核心。
