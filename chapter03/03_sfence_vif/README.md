# sfence 与 VIF

第一章第 3 节用相互依赖的向量指令看到：紧跟其后的读数 R1 不变，隔一条 `sfence` 的 R2 才包含向量一侧的等待；并由此给出了两级发射的大意。那里只需要会用，这一节给出确切的规则：两级发射之间的队列有多大、`sfence` 到底等什么、读数本身怎样干扰被测的程序，以及它为什么不只影响计时，也影响正确性。第一章中编译器在某些等待之后插入的 `sfence`，在这里得到解释。

## 两级发射

TensorCore 的 bundle 按顺序经过两次发射：

```text
标量发射（每周期一个 bundle）──→ 标量指令在此执行，包括 srdreg 读 LCC、sld、spop
      │
      └─ 含向量或 DMA 指令的 bundle ──→ VIF ──→ 向量发射（按序，每周期最多一个）
```

- 标量发射按顺序进行，没有阻塞时每周期一个 bundle。标量槽中的指令在标量发射时执行。
- 含向量指令（包括 `misc` 槽的 `vwait`、`vsyncadd`）或 DMA 指令的 bundle，在标量发射之后进入一个先进先出的队列 VIF（vector issue FIFO），再按顺序向量发射。向量发射由硬件的计分板（scoreboard）放行：它记录每个 TC VREG、每个结果队列何时就绪，每个单元何时空闲；向量指令要等源操作数就绪、等对应的单元空闲，`vwait` 要等信号量满足条件。这些等待都发生在向量一侧，不挡住标量发射。
- VIF 的容量有限。积压到上限时，标量发射也停下来。

所以 LCC 读数记录的是读数所在 bundle 的标量发射时刻。向量一侧落后多少，读数看不到，除非 VIF 已满、反过来挡住了标量一侧。

## sfence 的语义

`sfence` 是标量槽 `s0` 中的指令。它所在的 bundle 也在向量一侧占一个位置；下一个 bundle 要等 VIF 中此前的全部项（包括 `sfence` 所在的这一项）都已向量发射、并在标量一侧释放之后，才能标量发射。一项在向量发射后 10 个周期才在标量一侧释放（第 4 节）。

VIF 为空时，`sfence` 所在 bundle 在下一个周期向量发射，再过 10 个周期释放，所以下一 bundle 晚 11 个周期：这就是第一章第 3 节中 R2 − R1 = 12 的来源。VIF 中有积压时，`sfence` 之后的读数要等积压全部发射完，于是包含了向量一侧落后的时间。

要注意 `sfence` 等的是“发射”，不是“完成”：

- 它不等 DMA 完成，也不等 MXU、EUP 的计算完成。只有当 VIF 中有一条 `vwait` 在等 DMA 的信号量时，`sfence` 才间接地等到了这次 DMA：`vwait` 在信号量满足之前不会向量发射。
- 它从下一个 bundle 起才起作用。与读数放在同一个 bundle 的 `sfence` 不能让这次读数包含此前的等待（[研究报告 42](../../../pallas-tpu-readings-dev/research_reports/42_scalar_cycle_boundaries.md)）。

## 实验：向量一侧慢于标量一侧

本小节实验[源码](01_tpuasm_vector_lag.py)、[输出](01_tpuasm_vector_lag.txt)。

片段的形式与第一章第 3 节相同（R0、工作、R1、`sfence`、R2），中间的工作是三种向量一侧较慢的序列。第一种就是第一章第 3 节的相互依赖的乘法，这里把 N 加大到 64。

**N 条相互依赖的向量乘法**，每条用上一条的结果：

```python
bundle('va0: vmul.8x128.f32 v11, v11, v10') * count
```

| N | 1 | 4 | 16 | 32 | 64 |
| --- | ---: | ---: | ---: | ---: | ---: |
| R1 − R0 | 2 | 5 | 17 | 36 | 100 |
| R2 − R0 | 14 | 19 | 43 | 75 | 139 |

R2 − R0 = 2N + 11：`vmul` 的结果要 2 个周期后才能被下一条使用，所以向量一侧每 2 个周期发射一条，比标量一侧慢一倍。N ≤ 16 时 R1 仍是 N + 1，看不到这种落后；N = 32、64 时 R1 也变大了，因为 VIF 积压到上限，挡住了标量发射。

**N 组从 Megacore Shared CMEM 读入 TC VREG**（第二章第 3 节的 `cld` 加 `vpop`）。`setup` 先把输入复制到 CMEM 地址 0：

```python
work = (bundle('cld: cld.8x128 crf, [cmem:0x0]') + bundle('vr0: vpop.8x128 v11, crf')) * count
```

| N | 1 | 4 | 16 | 32 | 64 |
| --- | ---: | ---: | ---: | ---: | ---: |
| R1 − R0 | 3 | 9 | 337 | 1201 | 2929 |
| R2 − R0 | 66、67 | 229 | 877 | 1741 | 3469 |

每组恰好 54 个周期：R2 − R0 = 54N + 13（N = 1 时偶尔少 1，为 66，见第 4 节）。N = 4 时 R1 只有 9，看起来这 8 个 bundle 只花了 8 个周期，实际上向量一侧要 200 多个周期才做完。

**一次 TC VMEM → CMEM 的 DMA 及其等待**：

```python
bundle(f's0: dma.simple [cmem:s23], [vmem:s23], length={granules}, dst_flag=[sflag:52]') + bundle(f'misc: vwait.ge [sflag:52], {granules}') + bundle(f'misc: vsyncadd.s32 [sflag:52], -{granules}')
```

| 大小 | 4 KiB | 32 KiB | 128 KiB |
| --- | ---: | ---: | ---: |
| R1 − R0 | 4 | 4 | 4 |
| R2 − R0 | 315 | 343 | 439 |

不加 `sfence` 的 R1 只有 4，与 DMA 的大小无关：`vwait` 在向量一侧等待，标量一侧直接走了过去。加了 `sfence` 的 R2 才包含 DMA 的时间。第二章第 2 节的 TC VMEM → CMEM 开销 `309 + 1K` 正是按“发起 → 等待 → 读数”测得的，等待与 `sfence` 放在同一个 bundle，4、32、128 KiB 时为 313、341、437。这里的 R2 − R0 恰好都多 2：`vwait` 之后还有 `vsyncadd` 和单独的 `sfence` 两个 bundle 要在向量一侧依次发射。

计时的规则由此确定：**测异步工作的完成，必须是“等待 → `sfence` → 下一个 bundle 读数”。** 只读 R1 会把 DMA 测成 4 个周期。反过来，`sfence` 会排空 VIF，使它之前和之后的工作不再重叠，所以计时本身会改变被测程序的行为：只在需要的边界上放 `sfence`。

## 实验：读数本身的干扰

本小节实验[源码](03_tpuasm_reading_density.py)、[输出](03_tpuasm_reading_density.txt)。

第一章第 3 节的 `KernelClock` 每次读数是 8 个 bundle，其中有一条 `sfence`。这里回答两个问题：读数差扣除 20 之后，是不是恰好等于中间那段代码的周期数；能不能在一段代码中间多读几次，把各段的周期数加起来当作整段的周期数。载体是三个只差一行的 kernel：把 `f32[128,128]` 搬进 TC VMEM，计算，搬回 HBM。

```python
OPERATIONS = {
    '2x + 1': lambda x: x * 2.0 + 1.0,
    'exp(x)': jnp.exp,
    'x 加上各行的和': lambda x: x + jnp.sum(x, axis=1, keepdims=True),
}
```

**两次读数之间隔 n 个 bundle。** 在同一个位置连着插入两次读数，中间放 n 个相同的 bundle：

```python
insert_bundles(serialized, {first: clock_read(0) + filler * count + clock_read(1)})
```

| n | 0 | 1 | 2 | 3 | 4 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 读数之差 | 20 | 20 | 20 | 21 | 22 | 26 | 34 |

空 bundle、`vnop`、`vadd` 三种 bundle 的结果相同。n ≥ 2 时读数之差是 18 + n，n = 0、1 时却都是 20。原因在读数自己的 8 个 bundle 里：前一次读数的最后一个 bundle 是恢复寄存器的 `sld s31`，后一次读数的第二个 bundle 是保存它的 `sst [smem:0x20101], s31`；`sld` 的结果 4 个周期后才能使用（第一章第 8 节），两次读数紧挨着时，这条 `sst` 要等 2 个周期。中间隔一个 bundle 时等 1 个周期，隔两个以上就不必等了。

所以“扣除 20”的约定下，n 个互不等待的 bundle 读作 n − 2。这是一个固定的偏移，不影响两个实现之间的比较，但把读数与模型或 bundle 数对照时要记得它。

**在计算段中间加入 k 次读数。** 计算段是输入 DMA 等到之后、发起输出 DMA 之前的 bundle。两端各读一次，中间再均匀地插入 k 次读数，每段的读数差各扣除 20，再相加：

```python
points = [first, *(first + (index + 1) * (last - first) // (extra + 1) for index in range(extra)), last]
segments = [difference - CLOCK_OVERHEAD for difference in readings(clock, clock.instrument(compiled, points), x, len(points))]
```

| kernel | 计算段的 bundle 数 | k = 0 | 1 | 2 | 4 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2x + 1` | 20 | 18 | 16 | 14 | 10 | 2 | 0 |
| `exp(x)` | 41 | 39 | 37 | 35 | 31 | 23 | 7 |
| x 加上各行的和 | 67 | 137 | 133 | 113 | 104 | 78 | 45 |

每个读数都是 4 次运行完全相同的值。k = 0 是这段计算连续执行的周期数。中间每多一次读数，合计就变小：

- `2x + 1` 与 `exp(x)` 每多一次读数少 2 个周期，正是上面那个偏移：每一段都少算 2 个周期。段数多到每段只有一两个 bundle 时，合计接近 0。
- “x 加上各行的和”少得更多：从 137 降到 45。这段计算要等 XLU 的跨通道归约，每次 79 个周期（第 4 节）。`sfence` 等的是此前的 bundle 全部发射，不等归约完成（本节开头）；读数的 8 个 bundle 和 `sfence` 的等待期间，已经提交的归约照常进行。于是读数自身的 20 个周期与归约的延迟重叠了，按 20 扣除时，把本来就要等的时间也扣掉了。

合计偏小只是这三个 kernel 的情形。标量一侧与向量一侧原本重叠的程序会偏向另一边：[研究报告 43](../../../pallas-tpu-readings-dev/research_reports/43_scalar_dma_overlap.md) 在一个循环的每次 DMA 等待处各加一条 `sfence`，整个程序慢了约 18.5%：标量一侧原本可以提前走到后面去，加了 `sfence` 之后每次都要停下来等。

由此得到两条规则：

- **一段区间只在两端各读一次。** 逐段读数之和不是连续执行的周期数，偏差的大小和方向取决于读数落在什么工作的中间。
- **要知道中间某一点的时刻，每次只移动一个读数。** 起点的读数不动，终点的读数换一个位置，重新运行；每次运行都只有两次读数，起点之后的程序一直连续执行到终点。第 4 节用这种办法逐个 bundle 地核对模型。

## 实验：去掉编译器插入的 sfence

本小节实验[源码](02_tpuasm_remove_sfence.py)、[输出](02_tpuasm_remove_sfence.txt)。

两级发射不只影响计时，也影响正确性。第一章第 8 节的 kernel 把两个标量 DMA 进 SMEM，再用 `sld` 读出：

```tpuasm
514: { s0: dma.simple [smem:s13], [hbm:s1], length=1, dst_flag=[sflag:52] }
515: { misc: vwait.ge [sflag:52], 1 }
516: { misc: vsyncadd.s32 [sflag:52], -1 }
517: { s0: sfence }
518: { s1: sld s16, [smem:0x3e] ; vld: vld.8x128 v0, [vmem:0x0] }
```

`vwait` 在向量一侧，`sld` 在标量一侧。若没有 517 的 `sfence`，`sld` 会在 `vwait` 等到 DMA 之前就标量发射，读到 SMEM 中的旧值。实验用 tpuasm 把 517 换成 `vnop`，其余不变：

```python
patched = tpuasm_tools.edit_bundles(serialized, {fences[0]: ('s0: sfence', 'misc: vnop')})
```

每次运行传入不同的标量，连续运行 20 次：

```text
原程序：20 次运行中，用本次标量 20 次，用上一次运行的标量 0 次
去掉 sfence：20 次运行中，用本次标量 0 次，用上一次运行的标量 19 次
```

去掉 `sfence` 后，每次读到的都是上一次运行留在 SMEM 中的标量（第一次运行读到的是原程序最后一次留下的值，不计入 19 次）。这就是第一章第 8 节中编译器“等待 SMEM 的 DMA 之后插入 `sfence`，等待向量数据时却不插入”的原因：读 TC VMEM 的 `vld` 也在向量一侧，排在 `vwait` 后面，自然不会早于 DMA 完成；读 SMEM 的 `sld` 在标量一侧，必须用 `sfence` 拦住。

## 顺序保证的一般规则

把上面的结论推广，就能判断两件事之间是否有顺序保证：

| 等待在 | 后续操作在 | 是否需要 `sfence` |
| --- | --- | --- |
| 向量一侧（`vwait`） | 向量一侧（`vld`、`vst`、`vsyncadd`、`vsyncadd.remote`） | 不需要：VIF 按顺序发射 |
| 向量一侧（`vwait`） | 标量一侧（`sld`、`sst`、`srdreg`、`spop`） | 需要 |

第二章第 3 节中，TensorCore 0 先等 DMA 把数据搬进 CMEM，再用 `vsyncadd.remote` 通知 TensorCore 1。两者都在向量一侧，`vsyncadd.remote` 排在 `vwait` 之后发射，所以 TensorCore 1 收到信号时数据已经完整，不需要 `sfence`。若通知改由标量一侧的操作完成，就必须在中间加 `sfence`。

[研究报告 47](../../../pallas-tpu-readings-dev/research_reports/47_vtrace_vif_timing.md) 讨论了同一问题在 `vtrace` 计时中的表现：`vtrace` 的记录走向量一侧，与 LCC 读数的边界不同，第 6 节会再遇到。
