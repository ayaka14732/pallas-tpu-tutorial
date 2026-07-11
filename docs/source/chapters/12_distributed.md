# 第 12 章：分布式 Kernel 与多核编程

## TPU 的多核架构

TPU 芯片通常包含多个 TensorCore（计算核心）：
- TPU v4：每个芯片 2 个 TensorCore
- TPU v5e：每个芯片 1 个 TensorCore
- TPU v5p：每个芯片 2 个 TensorCore

多个芯片通过 ICI（Inter-Chip Interconnect）高速互连，形成 3D Torus 拓扑。ICI 带宽极高（数 TB/s），使得跨芯片通信几乎和本地内存访问一样快。

## Megacore

当一个芯片有多个 TensorCore 时，默认情况下 kernel 只在一个核心上运行，另一半算力闲置。通过 `GridDimensionSemantics.PARALLEL` 标记可并行化的 grid 维度：

```python
pl.pallas_call(
    kernel,
    ...,
    grid=(batch_size, num_heads, seq_blocks),
    dimension_semantics=(
        pltpu.GridDimensionSemantics.PARALLEL,   # batch 可并行
        pltpu.GridDimensionSemantics.PARALLEL,   # head 可并行
        pltpu.GridDimensionSemantics.ARBITRARY,  # seq 顺序执行
    ),
)
```

规则：
- `PARALLEL` 维度的不同迭代之间**没有数据依赖**
- `PARALLEL` 维度必须出现在 `ARBITRARY` 维度之前
- 维度大小应是核心数的倍数（否则负载不均衡）

## core_map

`pltpu.core_map` 提供显式的多核控制（SPMD 模型）：

```python
def per_core_kernel(x_ref, y_ref):
    core_id = pltpu.current_core_index()
    # 根据 core_id 执行不同工作
    @pl.when(core_id == 0)
    def _():
        y_ref[...] = x_ref[...] * 2.0
    @pl.when(core_id == 1)
    def _():
        y_ref[...] = x_ref[...] + 1.0

result = pl.core_map(
    per_core_kernel,
    out_shape=jax.ShapeDtypeStruct(shape, dtype),
    mesh=pltpu.create_tensorcore_mesh("core"),
)(x)
```

## Remote DMA

TPU 核心之间可以通过 Remote DMA 直接传输数据，无需经过 HBM：

```python
remote_copy = pltpu.make_async_remote_copy(
    src_ref=local_vmem_ref,
    dst_ref=remote_vmem_ref,
    send_sem=send_semaphore,
    recv_sem=recv_semaphore,
    device_id=target_device_id,
)
remote_copy.start()
# ... 计算（与通信重叠）...
remote_copy.wait()
```

## ICI 集合通信

在 Pallas kernel 内部可以直接调用集合通信：

```python
# Ring permutation
result = jax.lax.ppermute(data, axis_name='devices', perm=[(i, (i+1) % n)])

# All-reduce sum
result = jax.lax.psum(data, axis_name='devices')

# All-gather
result = jax.lax.all_gather(data, axis_name='devices')
```

这些操作直接在 VMEM 和 ICI 之间发生，延迟远低于经过 HBM 的通信。

## 分布式 MatMul

大矩阵乘法可以跨多个核心/芯片并行：

**输出并行**（不需要通信）：
```python
# 每个核心计算输出的一部分
grid = (M // BM, N // BN)
dimension_semantics = (ARBITRARY, PARALLEL)  # N 维并行到多核
```

**归约并行**（需要 all-reduce）：
```python
# 每个核心计算部分 K 维的乘积
# 最后 all-reduce 求和
partial = local_matmul(a_shard, b_shard)
result = jax.lax.psum(partial, axis_name='devices')
```

## 通信-计算重叠

Megatron-LM 模式（通信和计算串行）：
```
All-Gather → MatMul → Reduce-Scatter
```

Pallas 优化模式（通信和计算重叠）：
```python
@pl.loop(0, num_shards)
def _(i):
    buf = i % 2

    # 等待当前数据到达
    wait_remote_copy(buf)

    # 启动下一次通信（与计算重叠）
    @pl.when(i + 1 < num_shards)
    def _():
        start_remote_copy(i + 1, buf=1-buf)

    # MXU 计算（隐藏通信延迟）
    acc[...] += matmul(local_input, weight_buf[buf])
```

## 与 GPU 多卡并行的对比

| 维度 | GPU (NCCL) | TPU (ICI) |
| :--- | :--- | :--- |
| 互连拓扑 | NVLink (点对点) 或 PCIe | ICI (3D Torus) |
| 带宽 | NVLink: ~900 GB/s | ICI: 数 TB/s (Pod 内) |
| 编程模型 | NCCL 集合通信（kernel 外部）| JAX 集合通信 + Remote DMA（kernel 内部）|
| 核内并行 | 多 SM 自动并行 | Megacore（需要显式标记）|
| 通信-计算重叠 | CUDA Stream + NCCL | 流水线 + 异步 DMA |

TPU 的关键优势：可以在 kernel 内部直接调用通信原语，实现细粒度的通信-计算重叠。GPU 上通常只能在 kernel 之间重叠。

## 在 RPA v3 中的应用

RPA v3 使用 Megacore 将不同的 batch/head 分配到不同核心：

```python
dimension_semantics=(
    pltpu.GridDimensionSemantics.PARALLEL,  # batch × head
)
```

每个核心独立处理一个 (batch, head) 组合，核心之间不需要通信。这是最简单也最高效的并行模式——当工作可以完全独立分割时，避免通信是最好的优化。

## 实践建议

1. **优先使用 PARALLEL 标记**：最简单，编译器自动处理
2. **避免核间通信**：设计算法使核心独立工作
3. **通信与计算重叠**：如果必须通信，使用双缓冲隐藏延迟
4. **注意 VMEM 分配**：多核共享 HBM 但各有独立 VMEM
5. **负载均衡**：确保 PARALLEL 维度大小是核心数的倍数
