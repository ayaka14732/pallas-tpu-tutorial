# Megacore Shared CMEM

TPU v4 的一颗芯片上，除了每个 TensorCore 私有的 TC VMEM，还有一块两个 TensorCore 共享的片上内存，本教程称为 Megacore Shared CMEM。本节回答三个问题：数据怎样进出它，向量单元怎样直接读它，两个 TensorCore 怎样通过它共享数据。

公开的 Pallas 接口在 TPU v4 上没有提供可用的 CMEM buffer：在 `scratch_types` 中申请 `pltpu.CMEM(shape, dtype)`，编译失败，`Scratch memref allocation only supported for vmem, smem and semaphore_mem`（实验[源码](03_pallas_cmem_scratch.py)、[输出](03_pallas_cmem_scratch.txt)）。本节的实验因此都用 tpuasm 改写已编译的 kernel 来完成：先让 Mosaic 编译一个只用 HBM 和 TC VMEM 的载体，再在机器清单中加入访问 CMEM 的指令。这正是前言所说的：硬件具备的能力，即使公开接口没有覆盖，也要有办法用上。

## 两条读取通路

本小节实验[源码](01_tpuasm_cmem_paths.py)、[输出](01_tpuasm_cmem_paths.txt)。

载体 kernel 有两个输入 x、y，但只读 x、计算 `2x`：

```python
def kernel(x_hbm: Ref, y_hbm: Ref, o_hbm: Ref, x_vmem: Ref, sem: Ref) -> None:
    pltpu.async_copy(x_hbm, x_vmem, sem).wait()
    x_vmem[...] = x_vmem[...] * 2.0
    pltpu.async_copy(x_vmem, o_hbm, sem).wait()
...
return kernel(pltpu.with_memory_space_constraint(x, pltpu.HBM), pltpu.with_memory_space_constraint(y, pltpu.HBM))
```

`pltpu.with_memory_space_constraint(x, pltpu.HBM)` 要求 XLA 把这个输入放在 HBM；输出同样用 `pltpu.HBM` 固定（第一章第 11 节）。这样 XLA 不会占用 CMEM，本 kernel 可以独占 CMEM，从地址 0 开始使用。

两种改写都先在输入 DMA 之前插入 4 个 bundle，把 y（地址在 `s1`）从 HBM 搬进 CMEM 地址 0：

```text
{ s0: simm.s32 s20, 0 }
{ s0: dma.simple [cmem:s20], [hbm:s1], length=8, dst_flag=[sflag:52] }
{ misc: vwait.ge [sflag:52], 8 }
{ misc: vsyncadd.s32 [sflag:52], -8 }
```

DMA 的端点写成 `[cmem:...]` 即可，长度和信号量的用法与第一章第 3 节完全相同。

第一种是 staging：把原来的输入 DMA 的源从 `[hbm:s0]` 改成 `[cmem:s20]`，数据经 CMEM → TC VMEM 进入原来的 `vld`。

第二种是直接读：保留 x 的输入 DMA，把读 TC VMEM 的 `vld` 换成读 CMEM 的 `cld`，并在乘法之前插入一条 `vpop`：

```text
{ cld: cld.8x128 crf, [cmem:0x0] }      # 原来的 vld：从 CMEM 地址 0 读一个 TC VREG 的数据，送入队列 crf
{ vr0: vpop.8x128 v0, crf }             # 新插入：从 crf 取回到 v0
{ va0: vmul.8x128.f32 v1, 2.0, v0 }     # 原来的乘法
```

两种改写的结果都是 `2y` 而不是 `2x`，说明数据确实来自 CMEM。`cld` 有自己的发射槽（第一章第 2 节的 `cld` 槽），它与 EUP、XLU、MXU 一样是“提交—取回”的形式：结果进入队列 `crf`，再由 `vpop` 取回。

## CMEM 有多大

本小节实验[源码](04_tpuasm_cmem_capacity.py)、[输出](04_tpuasm_cmem_capacity.txt)。

TC VMEM 的容量可以让编译器报出来（第一章第 3 节），CMEM 不能由 Pallas 分配，只能直接测。CMEM 的地址与 DMA 的长度一样以 512 B 的 granule 为单位。实验用第三章第 3 节的 `LccProbe` 执行手写片段：把一个 tile（记作 P）写到 CMEM 地址 0，把它的按位取反（记作 Q）写到地址 A，再把两处读回：

```python
+ dma('[cmem:s20]', '[vmem:s22]')     # P → CMEM 地址 0
+ dma('[cmem:s21]', '[vmem:s23]')     # Q → CMEM 地址 A
+ dma('[vmem:s24]', '[cmem:s20]')     # 读回地址 0
+ dma('[vmem:s26]', '[cmem:s21]')     # 读回地址 A
```

如果 A 在容量之内，地址 0 读回的仍是 P；如果 A 超出了容量并绕回到地址 0，第二次写入就会把 P 覆盖成 Q。实验只改 A：

```text
A = 0x1000（2.000 MiB 处）：地址 A 读回 Q True；地址 0 仍是 P True，变成了 Q False
A = 0x20000（64.000 MiB 处）：地址 A 读回 Q True；地址 0 仍是 P True，变成了 Q False
A = 0x3fff8（127.996 MiB 处）：地址 A 读回 Q True；地址 0 仍是 P True，变成了 Q False
A = 0x40000（128.000 MiB 处）：地址 A 读回 Q True；地址 0 仍是 P False，变成了 Q True
A = 0x7fff8（255.996 MiB 处）：地址 A 读回 Q True；地址 0 仍是 P True，变成了 Q False
A = 0x80000（256.000 MiB 处）：地址 A 读回 Q True；地址 0 仍是 P False，变成了 Q True
```

直到 `0x3fff8`（最后一个 tile）都互不干扰；`0x40000` 和 `0x80000` 写到了地址 0。所以 Megacore Shared CMEM 是 `0x40000` 个 granule，即 **128 MiB**，是一个 TensorCore 的 TC VMEM（16 MiB）的 8 倍，由两个 TensorCore 共用。超出容量的地址不会报错，也不会停机，而是按 `0x40000` 回绕：写错地址会悄悄覆盖别处的数据。`pltpu.get_tpu_info()` 报告的 `cmem_capacity_bytes=67000000` 是 JAX 中写死的估计值，与实测不符。

本节的 kernel 把输入输出都固定在 HBM，所以可以从地址 0 起随意使用 CMEM。XLA 自己也会把数组放进 CMEM（见本节最后），与 XLA 的其他部分共存时，要避开它已经占用的区域。

## 两条通路的开销

[tpu-v4-latency-numbers](../../../tpu-v4-latency-numbers/README.md) 测得（周期数，N 为读取的 TC VREG 个数，K 为 KiB 数）：

| 通路 | 周期数 |
| --- | --- |
| TC VMEM → TC VREG，连续 `vld` | `N + 13` |
| CMEM → TC VREG，`cld` 与 `vpop` 逐个串行 | `56N + 11` |
| CMEM → TC VREG，先发出 D 个 `cld` 再交替取回（D = 1、2、4） | `54⌈N/D⌉ + 2((N−1) mod D) + 13` |
| CMEM → TC VMEM，DMA | `311 + 0.5K` |
| HBM → TC VMEM，DMA | `483.6 + 1.105K` |

前四行是精确的公式，不是拟合：串行地读，每多一个 TC VREG 恰好多 56 个周期，而连续 `vld` 每多一个恰好多 1 个周期；CMEM → TC VMEM 的 DMA 每 2 KiB 恰好多 1 个周期，实测与公式最多差 1 个周期。只有最后一行经过 HBM 的 DMA 是对实测做的线性拟合，单次测量在拟合值上下波动。`cld` 之后要过 53 个周期才能 `vpop`，`cld` 和 `vpop` 各自每 2 个周期最多发射一次（第三章第 5 节）。先发出 D 个 `cld` 再交替取回，就把等待重叠起来：每 D 个 TC VREG 多 54 个周期，D = 4 时平均每个 13.5 个周期。

所以 CMEM 的定位是：比 HBM 近（DMA 固定开销 311 对 484 个周期，带宽约为 HBM 的两倍），比 TC VMEM 远。

- 只读一次的数据：用 `cld` 直接读，并尽量多发出几个 `cld` 再取回，省去一次 DMA 和 TC VMEM 空间。
- 要反复读的数据：先用 DMA 搬进 TC VMEM，再用 `vld`。

> 暂且可以理解为：“`cld` 之后 53 个周期才能 `vpop`、每 2 个周期最多发射一次”这类规则，来自在 kernel 中插入周期计数器的实测。第三章第 5 节把这些规则整理成发射模型，用于从清单估算执行时间。

## 两个 TensorCore 通过 CMEM 共享数据

本小节实验[源码](02_tpuasm_shared_between_cores.py)、[输出](02_tpuasm_shared_between_cores.txt)。

CMEM 的另一个价值在于它是两个 TensorCore 共享的：一份数据只需从 HBM 读一次，两个 TensorCore 都能用。实验以第 1 节的两 TensorCore kernel 为载体（各读 x 的一半、乘以 2），改写为：TensorCore 0 把整个 y 搬进 CMEM，两个 TensorCore 再各从 CMEM 读自己的一半。

```text
{ s0: smov s22, s1 }                                         # 保存 y 的地址（编译器随后会把 s1 另作他用）
{ s0: seq.s32 p10, s9, 0 ; s1: simm.s32 s21, 0 }             # p10 = (TensorCore 编号 == 0)
{ s0: @p10 dma.simple [cmem:s21], [hbm:s22], length=64, dst_flag=[sflag:52] }
{ misc: @p10 vwait.ge [sflag:52], 64 }
{ misc: @p10 vsyncadd.s32 [sflag:52], -64 }
...
{ s0: dma.simple [vmem:s24], [cmem:s1], length=32, dst_flag=[sflag:52] }   # 每个 TensorCore 读自己的一半
```

谓词 `@p10` 让只有 TensorCore 0 执行这次搬运。每个 TensorCore 的输入 DMA 改为从 CMEM 读，此时 `s1` 正是“TensorCore 编号 × 32”（第 1 节的清单），即它那一半在 CMEM 中的起点。

关键在于这次搬运放在哪里。实验比较两种位置，每种用 200 组不同的 y 运行：

| TensorCore 0 的搬运 | TensorCore 0 的一半出错 | TensorCore 1 的一半出错 |
| --- | ---: | ---: |
| 在入口汇合之前 | 0 次 | 0 次 |
| 在入口汇合之后 | 0 次 | 200 次 |

放在入口汇合之前时，TensorCore 0 先完成搬运，再给 TensorCore 1 发出汇合信号；TensorCore 1 等到这个信号才继续，读 CMEM 时数据已经就位。放在入口汇合之后，两个 TensorCore 各自继续执行，TensorCore 1 不会等 TensorCore 0 的搬运，读到的是 CMEM 中上一次运行留下的旧数据。TensorCore 0 自己的一半总是正确的，因为它在同一个 TensorCore 上先搬后读。

这个实验说明了共享内存的基本规则：一方写、另一方读，中间必须有同步来保证顺序。这里借用了编译器插入的入口汇合；第 4 节介绍怎样在 Pallas 中显式写出这种同步。

> 暂且可以理解为：TensorCore 0 的 `vsyncadd.remote` 在它的 DMA 已经等到之后才发出，所以 TensorCore 1 看到信号时，CMEM 中的数据已经完整。DMA 完成、信号发出、对方读取三者之间的顺序保证，第三章第 4 节讨论 `sfence` 时会再次遇到。

## 对照：原生 XLA 怎样使用 CMEM

XLA 会自动使用 CMEM。第一章第 11 节的 XLA baseline 中，输入先经 `copy-start`/`copy-done` 预取进 CMEM，再用 `cld` 读入；第 2 节中，XLA 把循环中传递的数组放进了 CMEM。XLA 的选择对单个 kernel 未必最优，但它说明 CMEM 在实际程序中被大量使用。手写 kernel 时，用 tpuasm 可以把同样的通路握在自己手里。
