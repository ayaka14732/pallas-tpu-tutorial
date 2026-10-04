# 主机内存

TensorCore 能直接搬运的内存不止芯片上的这几种。kernel 还可以读写主机（CPU 一侧）的 pinned host memory：把暂时不用的数据卸载到主机，腾出 HBM；或者在 kernel 运行期间与主机交换数据。本节看清这条通路在 TPU v4 上怎样工作、代价多大、公开接口有哪些限制，以及怎样用 tpuasm 绕过其中一个。

## 写法：主机内存中的输入与输出

本小节实验[源码](01_pallas_host_roundtrip.py)、[输出](01_pallas_host_roundtrip.txt)。

数组要放进 pinned host memory，需要在 sharding 中指定 `memory_kind='pinned_host'`：

```python
host = jax.NamedSharding(mesh, P(), memory_kind='pinned_host')
x = jax.device_put(x, host)
compiled = tpuasm_tools.compile(build(mesh, direct), x, mesh=mesh, out_shardings=host)
```

kernel 一侧有三处变化：

```python
@pl.kernel(
    out_type=(pl.MemoryRef(jax.core.ShapedArray(x.shape, x.dtype), pl.HOST), pltpu.HBM(x.shape, x.dtype)),
    scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.SemaphoreType.DMA),
    ...
)
def kernel(x_host, o_host, staging_hbm, vmem, sem):
    pltpu.async_copy(x_host, staging_hbm, sem).wait()
    pltpu.async_copy(staging_hbm, vmem, sem).wait()
    vmem[...] = vmem[...] * 2.0 + 1.0
    pltpu.async_copy(vmem, staging_hbm, sem).wait()
    pltpu.async_copy(staging_hbm, o_host, sem).wait()
...
result, _ = kernel(pltpu.with_memory_space_constraint(x, pl.HOST))
return pltpu.with_memory_space_constraint(result, pl.HOST)
```

- 输入用 `pltpu.with_memory_space_constraint(x, pl.HOST)` 要求留在主机内存，kernel 收到的是一个主机内存中的 Ref。
- 输出的类型写成 `pl.MemoryRef(jax.core.ShapedArray(shape, dtype), pl.HOST)`，并在 kernel 之外再加一次约束。
- 主机与 TC VMEM 之间要经过 HBM 中转。Mosaic 不允许在 `scratch_types` 中分配 HBM（与第 3 节的 CMEM 相同），所以中转 buffer 写成 kernel 的第二个输出，调用后丢弃。

结果正确，输出的 `memory_kind` 是 `pinned_host`。若让主机与 TC VMEM 直接 DMA，编译失败：`Unimplemented DMA from host to vmem`。

## 清单：一次主机 DMA 是一个请求

HBM 与 TC VMEM 之间仍是熟悉的 `dma.simple`，但主机与 HBM 之间的搬运完全不同：

```text
{ misc: vwait.eq [sflag:485], 0 }            # 等邮箱空闲
...                                          # 拼出主机地址 s16 与请求字 s25
{ misc: vsyncset.s32 [sflag:487], 8 }        # 长度：8 个 granule
{ misc: vsyncset.s32 [sflag:486], s16 }      # 主机地址
{ misc: vsyncset.s32 [sflag:485], s25 }      # 请求字，同时表示邮箱已占用
{ misc: vint 2 }                             # 中断主机
{ misc: vwait.ge [sflag:52], 8 }             # 等 DMA 信号量
```

TensorCore 没有自己发起这次 DMA。它把长度、主机地址和请求写进三个同步标志组成的邮箱，用 `vint 2` 中断主机，由主机上的 TPU runtime 完成搬运，再给 DMA 信号量加上 8。kernel 中的 `.wait()` 与其他 DMA 一样，只是等待的对象变成了主机。

## 代价

[tpu-v4-latency-numbers](../../../tpu-v4-latency-numbers/README.md) 测得，TensorCore 发起 HBM → pinned host，从 `vint` 之前到完成：

| 同时发出的请求数 | 周期数（K 为每个请求的 KiB 数） |
| ---: | --- |
| 1 | `131718 + 72.3K` |
| 4 | `520868 + 105.9K`（K ≤ 2048） |

一个请求的固定开销约 13 万个周期（约 125 µs），是 HBM → TC VMEM 的 270 多倍；每 KiB 约 72 个周期，约 15 GB/s。同时发出 4 个请求，固定开销约为 4 倍：这些请求没有像芯片内的 DMA 那样重叠。

所以主机内存适合少量、大块、可以提前发起的搬运，例如把整块不再使用的数据卸载出去，再在需要之前很久把它取回；不适合逐 tile 的流水。

## 运行时的页号：Mosaic 的限制与 tpuasm

本小节实验[源码](02_pallas_host_dynamic_page.py)、[输出](02_pallas_host_dynamic_page.txt)。

把主机内存中的输出组织成 4 页（`f32[4,8,128]`，每页 4 KiB），kernel 把结果写到第 p 页。页号为常数时一切正常；页号来自运行时的标量时（`pages_host.at[page_smem[0]]`），编译失败：

```text
INTERNAL: LLO_CHECK failure (.../llo_region_builder.cc:5132) multiplier_in_bytes % word_size == 0 (512 == 0) 512 4096
```

用 `pl.multiple_of` 声明对齐也不能绕过。第一章第 7 节用它向编译器保证一个运行时的起点是 tile 的倍数；这里把输出写成 `f32[32,128]`，每页是 8 行的窗口，起始行声明为 8 的倍数：

```python
window = pages_host.at[pl.ds(pl.multiple_of(page_smem[0] * 8, 8), 8)]
```

编译仍然失败，报的是同一个检查。检查的对象是地址表达式中页号的乘数：编译器把窗口的起点写成“行号 × 512 B”，而主机地址以 4096 B 为单位，512 不能被 4096 整除；`pl.multiple_of` 提供的“行号是 8 的倍数”没有被这条路径用来把乘数合并成 4096。

这是 [JAX issue #40200](https://github.com/jax-ml/jax/issues/40200) 记录的问题，在当前的 libtpu `0.0.49` 中仍然存在。它使得按运行时位置写主机内存的环形缓冲区无法直接写出。

本小节实验[源码](03_tpuasm_host_dynamic_page.py)、[输出](03_tpuasm_host_dynamic_page.txt)。

但上一小节的清单说明，主机地址只是写进邮箱的一个数。页号为常数 2 的载体中，主机地址这样计算：

```text
{ s0: sadd.s32 s25, 2, s2 }          # 主机输出的起点 + 2 页
```

主机地址以 4 KiB 的页为单位，常数 2 直接加在起点上。于是改写只需两处：在 kernel 读入页号的 `sfence` 之后插入一条 `{ s1: sld s14, [smem:0x3e] }`，把 SMEM 中的 `page[0]` 读进一个未使用的寄存器；再把 `sadd.s32 s25, 2, s2` 改为 `sadd.s32 s25, s14, s2`。改写后，页号为 0、1、2、3 时，数据都写进了对应的页。

这个改写依赖“主机地址以 4 KiB 为单位、页大小正好是 4 KiB”。页大小不同时，要先把页号乘以每页的单位数。

## persistent kernel

主机内存还带来一种完全不同的执行方式：一次启动的 kernel 不退出，在循环中反复读主机内存中的请求、处理、把结果写回主机，由主机不断投放新请求。这样省去了每次启动 kernel 的开销，主机和 TPU 之间形成一条长期运行的流水线。

[研究报告 22](../../../pallas-tpu-readings-dev/research_reports/22_pallas_tpu_v4_host_hbm_persistent_streaming.md) 在 TPU v4 上实现了这样的 kernel：主机与 kernel 通过主机内存中的请求、响应和完成标志通信。它需要的能力超出了本节：kernel 运行期间主机要能并发写入 pinned host memory 并让 kernel 看到，需要锁定版本的私有接口；运行时页号的问题则用静态展开每个槽位的分支绕过。这些都是“硬件可以、公开接口不提供”的典型例子。

另一条通路由主机主动发起：[tpu-v4-latency-numbers 第 9 节](../../../tpu-v4-latency-numbers/README.md)测得，经 Host Magic Queue 由主机发起的 HBM → 主机搬运，一个请求约 `80815 + 78.3K` 个周期，固定开销比 TensorCore 发起的少约四成，但同样需要私有的运行时接口。
