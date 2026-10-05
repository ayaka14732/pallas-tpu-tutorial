# 综合：单芯片双 TC 大矩阵乘法

本节把前面的结论用在一个具体问题上：一颗芯片上的 `bf16[2048,2048] @ bf16[2048,2048] → f32[2048,2048]`。先由硬件参数估算这个问题的下限，再读 XLA 的做法，然后写出 Pallas kernel，按估算中的损失逐项改进，并按第三章的方法公平计时。

## 由硬件估算下限

**计算。** 一个 TensorCore 有 4 个 MXU（`pltpu.get_tpu_info()` 的 `num_mxus=4`），第三章第 5 节测得一个 MXU 每 8 个周期接收一个 8 行的 LHS TC VREG，与已装入 MXU 的 `128×128` RHS 块相乘（第一章第 10 节；下文把装入 MXU 的 RHS 块称为权重），即每周期 128 × 128 = 16384 次乘加。两个 TensorCore 共 8 个 MXU，每周期 131072 次乘加。这个问题有 2048³ ≈ 8.6 × 10⁹ 次乘加，至少要 **65536 个周期**。

**数据。** LHS、RHS 各 8 MiB，输出 16 MiB。若两个 TensorCore 各算一半的行，每个都要读全部 RHS，HBM 上共读写约 40 MiB；按第 2 节每 KiB 约 1 个周期计，约 4 万个周期，比计算少。所以这是一个计算受限的问题：只要搬运与计算重叠，时间就由 MXU 决定。不能重叠的只有两段：第一块数据到达之前，和最后一块结果写回之时。

## 原生 XLA

本小节实验[源码](01_jax_matmul.py)、[输出](01_jax_matmul.txt)。

```python
matmul = lambda lhs, rhs: jnp.dot(lhs, rhs, preferred_element_type=jnp.float32)
```

输入是 −2 到 2 的小整数，bf16 精确表示，f32 累加没有舍入，结果与 NumPy 逐元素相等。优化后的 HLO：

```text
%copy-start = (bf16[2048,2048]{...S(3)}, ...) copy-start(%lhs.1), cross_program_prefetch_index=0
%copy-done = bf16[2048,2048]{...S(3)} copy-done(%copy-start)
ROOT %fusion = f32[2048,2048]{1,0:T(8,128)} fusion(%copy-done, %rhs.1), kind=kOutput, calls=%fused_computation
```

XLA 先把整个 LHS 预取进 Megacore Shared CMEM（`S(3)`，第 3 节），fusion 再从 CMEM 读 LHS（清单中 512 条 `cld`）、用 `dma.strided` 读 RHS。两个 TensorCore 共用 CMEM 中的同一份 LHS，各算一半。

在程序和每条 HLO 指令的起止处读设备上的周期计数器，16 对新输入的中位数：

| | TensorCore 0 | TensorCore 1 |
| --- | ---: | ---: |
| `copy-done`（LHS 进入 CMEM） | 9138 | 9168 |
| `fusion` | 74832 | 74775 |
| 整个程序 | 105024 | 84122 |

> 暂且可以理解为：TensorCore 有一个每周期加 1 的计数器，`tpuasm_tools.KernelClock` 在已编译程序的指定位置插入读数，程序运行之后取回。第三章第 3 节详细介绍它；TensorCore 0 的整个程序还包含一段 kernel 之前的 runtime 代码，所以比较整个程序时用 TensorCore 1。

XLA 用了 84122 个周期，比下限多约 18600 个周期，其中约 9100 个是 LHS 进入 CMEM 之前完全不能计算的时间。

## Pallas：每个 TensorCore 一半的行

本小节实验[源码](02_pallas_matmul.py)、[输出](02_pallas_matmul.txt)。

按硬件参数安排数据：

- 两个 TensorCore 各算 1024 行（第 1 节）。每个只需要自己那一半 LHS，4 MiB，放得进 16 MiB 的 TC VMEM，不必经过 CMEM。
- RHS 按 256 列分成 8 块，每块 `bf16[2048,256]`（1 MiB），双缓冲：算第 j 块时读入第 j + 1 块（第 2 节）。
- 输出块 `f32[1024,256]`（1 MiB）也双缓冲：算完一块就发出写回，两块之后再等它完成。

TC VMEM 共用 4 + 2 + 2 = 8 MiB。主循环与第 2 节的流水线相同：

```python
@pl.loop(1, last)
def _(block: jax.Array) -> None:
    slot = block % 2
    load_rhs(block, slot).wait()

    @pl.when(block + 1 < BLOCKS)
    def _() -> None:
        load_rhs(block + 1, 1 - slot).start()

    # 输出 buffer slot 上一次用于 block - 2。
    @pl.when(block >= 2)
    def _() -> None:
        store(block - 2, slot).wait()

    compute(slot)
    store(block, slot).start()
```

RHS 块是一个宽数组的列窗口，起点在运行时才知道，按第一章第 7 节用 `pl.multiple_of` 标明对齐：

```python
rhs_hbm.at[:, pl.ds(pl.multiple_of(block * BLOCK, BLOCK), BLOCK)]
```

### 一次 jnp.dot 不能太大

最初的 `compute` 直接 `jnp.dot(lhs_vmem[...], rhs_vmem[slot])`，编译失败：

```text
RESOURCE_EXHAUSTED: ... Ran out of memory in memory space vmem. Used 16.08M of 16.00M vmem.
  1. Size: 8.08M
     XLA label: register allocator spill slots in HLO :: matmul.1
```

Mosaic 把一次 `jnp.dot` 的操作数整个当作 TC VREG 的数组处理，放不下的部分溢出到 TC VMEM；`[1024,2048]` 的 LHS 有 1024 个打包的 TC VREG，溢出区就要 8 MiB。所以 `compute` 每次只算 `chunk` 行：

```python
step = min(chunk, count)

@pl.loop(start, start + count, step=step)
def _(row: jax.Array) -> None:
    rows = pl.ds(pl.multiple_of(row, step), step)
    o_vmem[slot, rows] = jnp.dot(lhs_vmem[rows], rhs_vmem[slot], preferred_element_type=jnp.float32)
```

### 逐项改进

在这个结构上，实验依次改变一处，其余与上一个版本相同（源码中的 `VARIANTS`）：

1. **LHS 一次读入**（`panels=1`）：prologue 发出 4 MiB 的 LHS 和第 0 块 RHS，全部到达后才开始计算。
2. **LHS 分 4 段，同时发出**（`panels=4`）：第 0 块按段计算，每到一段 LHS（256 行，1 MiB）就算这 256 行：

   ```python
   for panel in range(panels):
       load_lhs(panel).wait()
       compute(0, panel)
   ```

3. **LHS 分 4 段，逐段发出**（`staged=True`）：prologue 只发出第 0 段，每等到一段才发出下一段：

   ```python
   for panel in range(1 if staged else panels):
       load_lhs(panel).start()
   ...
   for panel in range(panels):
       load_lhs(panel).wait()
       if staged and panel + 1 < panels:
           load_lhs(panel + 1).start()
       compute(0, panel)
   ```

4. **最后一块分段写回**（`split_last=True`）：最后一块也按段计算，每算完一段就发出这一段的写回，与后面几段的计算重叠：

   ```python
   for panel in range(panels):
       compute(block % 2, panel)
       store_panel(block, panel).start()
   ```

5. **每次 `jnp.dot` 512 行**（`chunk=512`）。

结果（设备上的周期数，16 对新输入的中位数；五个版本都与 NumPy 逐元素相等）：

| 版本 | kernel（TensorCore 0） | TensorCore 1 的整个程序 |
| --- | ---: | ---: |
| XLA | 74832（fusion）+ 9138（copy-done） | 84122 |
| 1. LHS 一次读入 | 84919 | 85038 |
| 2. LHS 分 4 段，同时发出 | 85012 | 85131 |
| 3. LHS 分 4 段，逐段发出 | 82613 | 82732 |
| 4. 最后一块分段写回 | 80626 | 80745 |
| 5. 每次 `jnp.dot` 512 行 | 78186 | 78305 |

读这张表：

- 最直接的版本 1 已经与 XLA 只差约 1000 个周期。它没有用 CMEM，两个 TensorCore 各自从 HBM 读 LHS 的一半，开头等待 4 MiB 的 LHS 加 1 MiB 的 RHS；XLA 开头等待 8 MiB 的 LHS 进入 CMEM。
- 版本 2 完全没有改善。原因在第 2 节：同时进行的 DMA 分享 HBM 的带宽。4 段 LHS 和第 0 块 RHS 一起发出，它们以大致相同的速度推进，第 0 段并不比整块更早到达。版本 3 只改了发出的时机，让第 0 段独占带宽先到达，其余各段在计算第 0 段时依次到达，快了约 2400 个周期。把 DMA 拆小只是前提，还要控制它们的先后。
- 版本 4 快了约 2000 个周期：最后一块 1 MiB 的写回（两个 TensorCore 同时写，约 2000 个周期）原本完全不能与计算重叠，现在只剩最后 256 行的 0.25 MiB。
- 版本 5 又快了约 2400 个周期。下一小节说明原因。

### 剩下的时间去了哪里

版本 4 比 65536 个周期的下限多约 15000 个周期。开头和结尾不能重叠的部分约 2000 个周期；其余来自 MXU 没有满负荷。每次 `jnp.dot` 算 256 行时，每个 `128×128` 的权重只用于 256 / 16 = 16 条打包的 `vmatmul`，之后就要换下一块权重：8 次 `vmatpush` 加一次 `vdwg`（第一章第 10 节）。清单中 `vmatpush` 与 `vmatmul` 的条数之比约为 1 : 2，换权重占用了 MXU 的大量发射机会。版本 5 把每次 `jnp.dot` 的行数加倍，每块权重用于 32 条 `vmatmul`，降到 78305 个周期，比 XLA 快 7%。

版本 5 的代价是代码量：Mosaic 把每次 `jnp.dot` 完全展开成指令，kernel 段由版本 4 的 17764 个 bundle 增加到 19238 个。行数再加大，溢出区还会超过 TC VMEM 的容量（前一小节）。

## 小结

这个问题中，硬件给出的结论直接决定了设计：

- 计算受限，所以首要目标是让 MXU 不停；数据搬运只要与计算重叠即可，大小按 TC VMEM 的容量选。
- 每个 TensorCore 一半的 LHS 放得进 TC VMEM，就不需要 CMEM。XLA 的 CMEM 预取解决的是另一种安排下的问题，代价是开头约 9100 个周期的等待。
- 流水线的头和尾是不能重叠的部分，把它们切小可以直接省下时间。
- 把 DMA 拆小之后，还要控制它们的先后：同时发出的 DMA 分享带宽，谁也不会先到。
- `jnp.dot` 的大小同时受溢出区容量、程序大小和权重复用率的限制：太大放不下，太小 MXU 要频繁换权重。

要进一步接近下限，需要直接控制 MXU 的权重装入与 `vmatmul` 的顺序（第一章第 10 节的 MXU FIFO 接口与 tpuasm 修正），让一块权重在更多行上复用。
