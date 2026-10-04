# DMA 的代价与软件流水

第一章第 3 节说过，DMA 的发起和等待是两个独立的事件：发起后 TensorCore 继续执行，DMA 引擎在后台搬运。本节先用实测数据给出各条 DMA 通路的代价，再利用发起与等待的分离，把数据搬运与计算、输入与输出重叠起来，组织成软件流水线，最后看 tile 的大小怎样决定流水线能否接近带宽上限。

## 各条 DMA 通路的代价

[tpu-v4-latency-numbers](../../../tpu-v4-latency-numbers/README.md) 在 TPU v4 上测量了一个 TensorCore 发起一条 DMA、从发起到等到完成的周期数，并拟合出“固定开销 + 每 KiB 开销”的模型（K 为 DMA 的 KiB 数，详见其 [results/02_local.md](../../../tpu-v4-latency-numbers/results/02_local.md)）：

| 方向 | 周期数 | 按 1.05 GHz 折算的带宽 |
| --- | --- | ---: |
| HBM → TC VMEM | `483.6 + 1.105K` | 约 970 GB/s |
| TC VMEM → HBM | `417.5 + 1.051K` | 约 1020 GB/s |
| HBM → Megacore Shared CMEM | `438.0 + 1.110K` | 约 970 GB/s |
| Megacore Shared CMEM → HBM | `378.3 + 1.001K` | 约 1070 GB/s |
| Megacore Shared CMEM → TC VMEM | `311 + 0.5K` | 约 2150 GB/s |
| TC VMEM → Megacore Shared CMEM | `309 + 1K` | 约 1080 GB/s |

> 暂且可以理解为：这些数字是在 kernel 中直接读取 TensorCore 的周期计数器得到的，TensorCore 的时钟约为 1.05 GHz。第三章第 3–4 节介绍这种计时方法，第 7 节介绍时钟频率是怎样标定的。

这张表给出两个设计原则：

- 固定开销很大。一条 4 KiB 的 HBM→TC VMEM DMA 要约 488 个周期，其中搬运本身只占 4 个；一条 256 KiB 的 DMA 约 766 个周期，固定开销仍占六成以上。小而多的 DMA 主要在付固定开销。
- 两个 TensorCore 共享 HBM 带宽。两个 TensorCore 同时从 HBM 读时，每个的每 KiB 开销从 1.105 升到约 2.156 个周期，合计带宽不变；而 Megacore Shared CMEM → TC VMEM 在两个 TensorCore 同时读时，每个仍是 0.5 个周期每 KiB，互不影响。

## 串行的写法与它的代价

本小节实验[源码](01_pallas_double_buffer.py)、[输出](01_pallas_double_buffer.txt)。

实验对 `f32[32768,128]`（16 MiB）计算 `y = 2x + 1`，每次处理一个 `f32[512,128]`（256 KiB）的 tile，共 64 个。最直接的写法每个 tile 依次发起输入 DMA、等待、计算、发起输出 DMA、等待：

```python
@pl.loop(0, tiles)
def _(tile: jax.Array) -> None:
    load(tile, 0).start()
    load(tile, 0).wait()
    o_vmem[0] = x_vmem[0] * 2.0 + 1.0
    store(tile, 0).start()
    store(tile, 0).wait()
```

这里用 `pltpu.make_async_copy(源, 目的, 信号量)` 代替第一章的 `async_copy`：它只创建 DMA 的描述，不立即发起，之后可以分别调用 `.start()` 和 `.wait()`。同一个描述可以由不同的位置重新创建，`load(tile, slot)` 和 `store(tile, slot)` 就是两个创建描述的小函数：

```python
def load(tile, slot):
    return pltpu.make_async_copy(x_hbm.at[pl.ds(tile * tile_rows, tile_rows)], x_vmem.at[slot], x_sems.at[slot])
```

TC VMEM 中的输入输出 buffer 都分配成两份（`pltpu.VMEM((2, tile_rows, 128), ...)`），信号量也是两个一组（`pltpu.SemaphoreType.DMA((2,))`），`.at[slot]` 取其中第 slot 份。串行的写法只用第 0 份。

串行时，每个 tile 依次付出输入 DMA（约 766 周期）、计算和输出 DMA（约 686 周期）。清单中一个 tile 的计算约 90 个 bundle，64 个 tile 合计约 `64 × (766 + 90 + 686) ≈ 99000` 个周期。实测每遍 102958 个周期，与估算接近。

> 暂且可以理解为：每遍的周期数是在设备上读周期计数器得到的。kernel 内部把整个流水线重复 4 遍和 8 遍，各读出 kernel 的周期数，相减除以 4，消去 kernel 中只做一次的部分。第三章第 3 节介绍这个计数器和读它的工具 `KernelClock`。

## 输入双缓冲

既然 DMA 在后台进行，就可以在计算当前 tile 时，提前发起下一个 tile 的输入 DMA，写进另一份 buffer：

```python
load(0, 0).start()                      # prologue：先发出第 0 个 tile

@pl.loop(0, tiles)
def _(tile: jax.Array) -> None:
    slot = tile % 2
    load(tile, slot).wait()             # 等当前 tile 到达

    @pl.when(tile + 1 < tiles)
    def _() -> None:
        load(tile + 1, 1 - slot).start()    # 立即发出下一个 tile，写进另一份 buffer

    o_vmem[0] = x_vmem[slot] * 2.0 + 1.0
    store(tile, 0).start()
    store(tile, 0).wait()
```

`slot` 在 0 和 1 之间交替，称为 ping-pong。这种调度有三段：循环之前发出第一个 DMA（prologue），循环中“等当前、发下一个、算当前”（稳态），最后一个 tile 不再发出新的 DMA。实测每遍 67660 个周期：下一个 tile 的输入与当前 tile 的计算、输出重叠了。

## 输入输出都双缓冲

输出也可以不等：计算结果写进 `o_vmem[slot]`，发起输出 DMA 后直接进入下一个 tile。代价是写 `o_vmem[slot]` 之前，必须确认这份 buffer 上一次的输出 DMA（两个 tile 之前）已经完成，否则会覆盖还没搬走的数据：

```python
    @pl.when(tile >= 2)
    def _() -> None:
        store(tile - 2, slot).wait()

    o_vmem[slot] = x_vmem[slot] * 2.0 + 1.0
    store(tile, slot).start()
...
# epilogue：循环结束后，排空最后两个 tile 的输出 DMA。
store(tiles - 2, tiles % 2).wait()
store(tiles - 1, (tiles - 1) % 2).wait()
```

一个正确的双缓冲必须同时满足四条：

1. 读一份输入 buffer 之前，它的输入 DMA 已经等到（不读未到达的数据）；
2. 下一个输入 DMA 只写另一份 buffer（不覆盖正在计算的数据）；
3. 写一份输出 buffer 之前，它上一次的输出 DMA 已经等到（不覆盖未搬走的数据）；
4. kernel 结束之前，所有输出 DMA 都已等到。

每份 buffer 用自己的信号量，是满足第 1、3 条的前提：第一章第 3 节已经看到，共用信号量时无法只等其中一次 DMA。三种写法的结果都与 `2x + 1` 逐元素一致；但数值正确只能说明这次没有出错，不能证明四条都满足，所以写的时候要逐条检查。

清单中的循环体与设计一致：开头一条 `vwait.ge [sflag:s17], 512` 等当前 tile（信号量编号随 slot 变化，所以是寄存器），随后带谓词 `@p4` 的 `dma.simple` 发出下一个 tile，带谓词 `@p5` 的 `vwait.ge` 等两个 tile 之前的输出，然后是 64 组 `vld`/`vmul`/`vadd`/`vst`，每组两条算术分别落在 `va0` 和 `va1`，同一个 bundle 中完成。

实测每遍 67425 个周期，与输入双缓冲几乎相同。

## 两条 DMA 重叠时省下的是什么

本小节实验[源码](03_pallas_dma_overlap.py)、[输出](03_pallas_dma_overlap.txt)。

输出也双缓冲之后几乎没有变快，串行改成输入双缓冲也只从约 10.3 万个周期降到 6.8 万个，而不是减半。原因要单独测：kernel 中去掉计算，每个 tile 只发起一条或两条 DMA 再等待，

```python
for group in copies:
    for dma in group:
        dma.start()
    for dma in group:
        dma.wait()
```

实验只改 `copies`：同一个 `group` 中的 DMA 一起发起、一起等待，不同的 `group` 先后进行。每个 tile 的周期数（同样是重复 8 遍与 4 遍的 kernel 周期数之差，再除以 tile 数）：

| 每个 tile 的 DMA | 256 KiB | 1 MiB |
| --- | ---: | ---: |
| 只读入 | 818 | 1681 |
| 只写出 | 675 | 1484 |
| 先读入再写出 | 1527 | 3191 |
| 读入与写出同时进行 | 1130 | 2836 |
| 两条读入同时进行 | 1115 | 2807 |

同时进行比先后进行只少了 397 和 355 个周期，大致是一次固定开销，而不是其中一条 DMA 的全部时间。1 MiB 时，读入与写出同时进行要 2836 个周期，接近两者单独的时间之和 3165；两条读入同时进行也一样。也就是说，一个 TensorCore 同时进行的两条 HBM DMA 共用 HBM 的带宽：重叠能省下固定开销，省不下搬运数据本身的时间。这与开头表中两个 TensorCore 同时读 HBM 的结论相同。

所以在这条流水线中，每个 tile 的时间大致是“一次固定开销 + 读入的数据时间 + 写出的数据时间”。256 KiB 时按上表约 1130 个周期，64 个 tile 是 72000 个周期，与流水线实测的 67425 个周期相近；输出再双缓冲，能重叠的只剩固定开销，而它已经被输入 DMA 盖住，所以没有收益。要更快，只能让固定开销占得更少，即增大 tile。

## 只改 tile 大小：固定开销决定了流水线的上限

输入输出都双缓冲，只改 tile 的大小：

| tile | 每个 DMA | 每遍 |
| --- | ---: | ---: |
| `f32[512,128]` | 256 KiB | 67929 个周期 |
| `f32[1024,128]` | 512 KiB | 53147 个周期 |
| `f32[2048,128]` | 1 MiB | 43872 个周期 |

上表的趋势可以用开头的代价表解释。这个流水线中，同一时刻只有一个输入 DMA 在进行：等到当前 tile 才发出下一个。所以每个 tile 至少要付一次完整的输入 DMA，包括约 480 个周期的固定开销；tile 越小，固定开销占的比例越大。1 MiB 的 tile 时，32 MiB 的读写在 43872 个周期内完成，每 KiB 约 1.34 个周期，约 800 GB/s，开始接近 HBM 的带宽（读写共用 HBM，每 KiB 约 1.05–1.1 个周期，约 1 TB/s）。

要进一步隐藏固定开销，可以让同一方向同时有多个 DMA 在进行：latency-numbers 的测量表明，同时发出 4 个 DMA 再一起等待时，固定开销只付一次（`470.6 + 4.76K`，K 为每个 DMA 的 KiB 数）。流水线的深度（同时进行几个 DMA）和 tile 的大小是两个独立的设计参数，前者受 TC VMEM 容量的限制，后者还影响计算部分的组织。



这里还有一个陷阱：如果不用 `pltpu.HBM(...)` 固定输出的内存空间，而在 XLA 的循环中反复调用 kernel 来计时，XLA 会把循环中传递的数组放进 Megacore Shared CMEM，kernel 的 DMA 也随之变成 CMEM 与 TC VMEM 之间的搬运，测到的就不是 HBM 的通路了。本节的 kernel 因此把输出固定为 `out_type=pltpu.HBM(x.shape, x.dtype)`，并把重复放在 kernel 内部。

## 对照：原生 XLA

本小节实验[源码](02_jax_transform.py)、[输出](02_jax_transform.txt)。

XLA 对同一运算生成的 fusion 也是一个流水线：两条 `dma.strided`（输入与输出）在循环中反复发出，长度放在寄存器中。每次循环处理 683 个 TC VREG，即约 2.7 MiB；整个数组还按 TensorCore 编号分给两个 TensorCore。XLA 选择了很大的 DMA，与上面 tile 越大越接近带宽的结论一致。

两者的时间怎样公平比较，是一个比看上去更难的问题：XLA 的版本用了两个 TensorCore，又可能把数据放进 Megacore Shared CMEM。第三章第 2 节专门讨论这个问题。
