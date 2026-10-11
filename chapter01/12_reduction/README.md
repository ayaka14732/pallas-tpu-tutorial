# 归约

归约把一个轴上的元素合并成一个：求和、求最大值。在 TC VREG 中，沿通道和沿子通道是两种完全不同的操作，前者由 XLU 完成，后者由子通道循环移位与逐元素运算拼成。本节逐一看这两个方向，再看跨多个 tile 的归约，以及归约结果怎样从向量单元进入标量单元。

## 写法：输出的 shape 由运算推出

本小节实验[源码](01_pallas_reduction.py)、[输出](01_pallas_reduction.txt)；kernel 的构造在 [reduction_common.py](reduction_common.py) 中。

相对于第 1 节的最小 kernel，计算换成一个归约函数 `f`，输出的 shape 用 `jax.eval_shape` 从 `f` 推出：

```python
out = jax.eval_shape(f, jax.ShapeDtypeStruct(x.shape, x.dtype))
@pl.kernel(
    out_type=pltpu.HBM(out.shape, out.dtype),
    ...
)
def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, o_vmem: Ref, sem: Ref) -> None:
    pltpu.async_copy(x_hbm, x_vmem, sem).wait()
    o_vmem[...] = f(x_vmem[...])
    pltpu.async_copy(o_vmem, o_hbm, sem).wait()
```

这里的 `out_type` 不是 `jax.ShapeDtypeStruct`，而是 `pltpu.HBM(shape, dtype)`：它明确要求输出放在 HBM。第 1 节说过，kernel 输入输出的默认内存空间是 `ANY`，由 XLA 决定放在哪里。对于 `f32[8,1]` 这样又窄又小的输出，XLA 会把它放进 Megacore Shared CMEM，kernel 中的 DMA 目的地随之变成 `[cmem:...]`（实验[源码](03_pallas_output_memory_space.py)、[输出](03_pallas_output_memory_space.txt)）。

> 暂且可以理解为：Megacore Shared CMEM 是芯片上一块两个 TensorCore 共享的片上内存，容量 128 MiB，比 TC VMEM 大，比 HBM 近。第二章第 3 节详细介绍它。

本章只研究单个 TensorCore 与 HBM、TC VMEM 之间的操作，所以这里固定为 HBM。另外，`f32[8,1]` 在 HBM 中的 layout 与 kernel 写出的不同，XLA 会在 kernel 之后追加一段重排；本节的统计只看 kernel 本身（`kernel_listing(compiled, pallas_only=True)`）。

## 沿通道：XLU 的跨通道归约

`jnp.sum(x, axis=1, keepdims=True)` 把 `f32[8,128]` 的每一行加成一个数：

```tpuasm
{ vx0: vadd.xlane.0.8x128.f32 trf0, v0 }
{ vr0: vpop.8x128 v3, trf0 }
{ vst: vst.msk.8x128 [vmem:0x8], vm0, v3 }
```

一条 `vadd.xlane` 把一个 TC VREG 送入 XLU，XLU 对每个子通道的 128 个通道求和，结果经 `trf0` 取回。随后的掩码 store 只写每行的第 0 列（掩码 `vm0` 由 `vlaneseq`、`vand`、`veq` 生成，表示 lane == 0）。`max(x, axis=1)` 相同，指令换成 `vmax.xlane`。

bf16 输入（`bf16[16,128]`，先 `astype(jnp.float32)`）先 unpack 成两个 f32 TC VREG，各做一次 `vadd.xlane`。

### 结果在每个通道中

本小节实验[源码](04_tpuasm_xlane_layout.py)、[输出](04_tpuasm_xlane_layout.txt)。

上面的掩码 store 只写第 0 列，那么 `vpop` 取回的 TC VREG 中，各行的和放在哪里？用 tpuasm 直接执行一条 `vadd.xlane`，把取回的整个 TC VREG 写回主机（第 3 节的 `LccProbe` 载体，`run_tiles` 取回片段写入 TC VMEM 的 tile）：

```python
body = bundle('vx0: vadd.xlane.0.8x128.f32 trf0, v10') + GAP + bundle('vr0: vpop.8x128 v11, trf0') + GAP + bundle('vst: vst.8x128 [vmem:0x8], v11') + GAP
```

输入第 s 行第 l 列是 `(128s + l) mod 7`，再在第 37 列加上 10s，使各行的和与最大值都不同：

```text
## vadd.xlane
  每行的期望值：[379.0, 393.0, 407.0, 414.0, 421.0, 435.0, 449.0, 449.0]
  取回的 TC VREG 第 0、1、127 列：[379.0, 393.0, ...]、[379.0, 393.0, ...]、[379.0, 393.0, ...]
  每行 128 个 lane 都等于该行的结果：True
```

`vmax.xlane` 相同。XLU 把每一行的结果广播到这一行的全部 128 个通道。所以 `keepdims=True` 的 `f32[8,1]` 只是取第 0 列；如果下一步要用这个结果对同一行的每个元素做运算（例如减去最大值、除以和），它已经在每个通道中，不需要再广播。

### 周期数：79 个周期的往返，每 8 个周期一次提交

本小节实验[源码](06_tpuasm_reduction_latency.py)、[输出](06_tpuasm_reduction_latency.txt)。

跨通道归约与第 9 节的重排用的是同一个 XLU，测法也相同：

| 片段 | R2 − R0 | 结论 |
| --- | ---: | --- |
| `vadd.xlane` 后隔 d − 1 个 `vnop` 再 `vpop`，d = 1、40、78、79、80 | 92、92、92、93、94 | 从提交到可以取回是 79 个周期 |
| `vmax.xlane`、`vmin.xlane`，同样的片段 | 相同 | 三种归约的延迟相同 |
| 一个队列上连续 k 次 `vadd.xlane` 再取回 k 次，k = 1、2、4、8 | 92、100、116、148 | 每多一次多 8 个周期 |
| 每个 bundle 向 `trf0`、`trf1` 各提交一次，共 2k 次，再成对取回，k = 1、2、4、8 | 92、100、116、148 | 两个队列互不妨碍：同样的周期数完成两倍的归约 |

归约的往返是 79 个周期，比重排的 69 个长 10 个周期；发射间隔同样是 8 个周期。一个 TC VREG 的归约，时间几乎全在这 79 个周期里；许多 TC VREG 一起归约时，两个队列各每 8 个周期接受一条，平均每 4 个周期完成一个 TC VREG。

## 沿子通道：循环移位与逐元素运算

`jnp.sum(x, axis=0, keepdims=True)` 把 8 行加成一行。XLU 没有跨 sublane 的归约，编译器用第 9 节介绍的 `vrot.slane.down` 拼出一棵二叉树：

```tpuasm
4 × vrot.slane.down，vadd     # 与循环移动 4 行的自己相加
2 × vrot.slane.down，vadd     # 再与移动 2 行的相加
1 × vrot.slane.down，vadd     # 再与移动 1 行的相加
```

共 7 条 `vrot.slane.down`、3 条 `vadd.8x128.f32`，每个子通道都得到 8 行之和。第 9 节已经看到，`vrot.slane.down` 每次只移动一个子通道，没有位移量操作数；所以树虽然只有 3 层，移位却要 4 + 2 + 1 = 7 条。

`vrot.slane.down` 与 `vadd` 一样在向量 ALU 槽（`va0`、`va1`）中执行，不经过 XLU，也不需要取回。它的结果 2 个周期后可用：上面的时序实验中，N 条相互依赖的 `vrot.slane.down` 的 R2 − R0 是 2N + 11（N = 1、4、7、16 时为 14、19、25、43），与 f32 的加法相同（第 7 节）。树的每一层都依赖上一层，7 条移位与 3 条加法串成一条链，一个 TC VREG 约 20 个周期，是沿通道的 XLU 归约的四分之一。

`jnp.max(x, axis=0)` 却没有用树：清单中是 7 条 `vrot.slane.down` 和 7 条 `vmax.8x128.f32`，每移动一行就取一次最大值。移位数相同，逐元素运算多了 4 条，而且 7 次 `vmax` 前后依赖，串成一条长链。

可以自己写出树形的版本：

```python
def f(x):
    for shift in shifts:
        x = jnp.maximum(x, pltpu.roll(x, shift, axis=0))
    return x[0:1, :]
```

| `shifts` | `vmax` | `vrot.slane.down` |
| --- | ---: | ---: |
| 编译器生成 | 7 | 7 |
| `(4, 2, 1)` | 3 | 17 |
| `(4, 6, 7)` | 3 | 7 |

位移取 `(4, 2, 1)` 时，`vmax` 少了，移位却多到 17 条。原因是 `pltpu.roll(x, k, axis=0)` 被降低成 `(8 - k) % 8` 条 `vrot.slane.down`（第 9 节中位移 3 是 5 条，位移 7 是 1 条），位移 2 和 1 分别要 6 条和 7 条。对求最大值来说，向上或向下移动并不影响结果，于是改用 `(4, 6, 7)`，得到 3 条 `vmax` 加 7 条移位，与编译器为求和生成的树一样。这个例子说明，即使写的是同一个算法，也要对照清单，确认每个高层操作落成了几条指令。

## 跨多个 tile：先逐元素，再 sublane

`f32[64,128]` 沿 axis=0 求和：8 个 tile 先用 7 条 `vadd` 逐元素加成一个 TC VREG，再在这个 TC VREG 内做上面的 sublane 树（3 条 `vadd`、7 条移位），共 10 条 `vadd`。跨 tile 的归约就是普通的逐元素运算，开销随 tile 数线性增长；只有最后一个 TC VREG 内部的归约需要移位。

## 归约到标量：从向量单元到标量单元

`jnp.sum(x, keepdims=True)` 把 `f32[8,128]` 加成一个数。先 `vadd.xlane` 沿通道求和，再用 sublane 树合并 8 行，之后出现了两条新指令：

```tpuasm
{ vst: vpush v2sf, v11 }      # 把 TC VREG 送进向量到标量的队列 v2sf
{ s0: spop s0, v2sf }         # 标量单元从 v2sf 取出一个数
{ va0: vmov.8x128 v15, s0 }   # 再广播回向量，写入输出
```

第 8 节说过，标量进入向量单元只需一条 `vmov`；反方向要经过队列 `v2sf`：向量单元用 `vpush` 送入，标量单元用 `spop` 取出。`spop` 取出的是这个 TC VREG 第 0 个子通道、第 0 个通道的元素（[源码](05_tpuasm_vpush_element.py)、[输出](05_tpuasm_vpush_element.txt)：对一个随机的 TC VREG 执行 `vpush`、`spop`，取出的值只与输入的 (0, 0) 位置相等）；前面的归约已经把总和放进了每个位置，所以取哪一个都一样。这条通路让向量的计算结果可以决定标量单元的行为，例如作为循环次数或分支条件。这里结果最终又要写回 TC VMEM，编译器仍然把它送进了标量单元，再广播回来。

这条通路的延迟也在时序实验中：`vpush` 之后隔 d − 1 个 `vnop` 再 `spop`，读紧跟其后的 R1。

| d | 1 | 12 | 40 | 42 | 44 | 48 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 − R0 | 45 | 45 | 45 | 45 | 46 | 50 |

`spop` 是标量指令，它等待时挡住的是标量发射，所以这一次连不带 `sfence` 的 R1 也跟着变：d ≤ 42 时 R1 − R0 都是 45。**`spop` 最早在 `vpush` 之后 43 个周期执行，期间标量一侧停止发射。** 把向量的计算结果变成循环次数或分支条件，每次都要付这 43 个周期。

## 对照：原生 XLA

本小节实验[源码](02_jax_reduction.py)、[输出](02_jax_reduction.txt)。

XLA 的五个归约都先把输入预取到 Megacore Shared CMEM，再用 `cld` 读入 TC VREG，这是 XLA 的默认选择。计算部分：

| 归约 | XLA | Pallas |
| --- | --- | --- |
| `sum(x, axis=1)` | 1 次 `vxpose` 转置、24 条 `vadd`、17 条 `vpop`、7 条移位 | 1 条 `vadd.xlane` |
| `sum(x, axis=0)` | 8 条 `vadd`、7 条移位 | 3 条 `vadd`、7 条移位 |
| `max(x, axis=0)` | 8 条 `vmax`、7 条移位 | 7 条 `vmax`、7 条移位 |
| `sum(x)` | 1 条 `vadd.xlane`、8 条 `vadd`、7 条移位 | 1 条 `vadd.xlane`、3 条 `vadd`、7 条移位 |

XLA 的 sublane 归约用了 8 条逐元素运算，沿通道的求和则先转置、再沿子通道归约，没有用 `vadd.xlane`。归约是最常见的操作之一，读清单、选对硬件通路，就能比 XLA 的通用写法少很多指令。少多少周期，见下一小节。

## 周期数

本小节实验[源码](07_jax_pallas_cycles.py)、[输出](07_jax_pallas_cycles.txt)。

把各种归约放进第 3 节的 TC VMEM 边界，原生 XLA 与 Pallas kernel 各测一次，两边的结果逐元素相同：

| 归约 | 输入 | XLA | Pallas |
| --- | --- | ---: | ---: |
| `sum(x, axis=1)` | `f32[8,128]` | 1045（1044–1049） | 100 |
| `max(x, axis=1)` | `f32[8,128]` | 1041 | 100 |
| `sum(x, axis=1)` | `f32[64,128]` | 1100（1100–1101） | 125 |
| `sum(x, axis=1)` | `f32[1024,128]` | 1944 | 605 |
| `sum(x, axis=1)` | `f32[8,1024]` | 1059（1059–1060） | 115 |
| `sum(x, axis=0)` | `f32[8,128]` | 55 | 51 |
| `max(x, axis=0)` | `f32[8,128]` | 55 | 49 |
| `sum(x, axis=0)` | `f32[64,128]` | 65 | 62 |
| `sum(x, axis=0)` | `f32[1024,128]` | 303 | 295 |
| `sum(x)` | `f32[8,128]` | 128 | 187 |

**沿通道。** Pallas 的 100 个周期就是一次 XLU 往返：79 个周期的等待，加上读入、取回、写出和有等待时外壳的 19 个周期。8 个 TC VREG 是 125，128 个是 605：从 8 个到 128 个多 120 个 TC VREG、480 个周期，每个 4 个周期，正是两个队列满负荷时的吞吐。`f32[8,1024]` 每行跨 8 个 TC VREG，先用 7 条 `vadd` 逐元素合并，再做一次 `vadd.xlane`，只比一个 TC VREG 多 15 个周期。

XLA 在这个方向上慢一个数量级。它不用 `vadd.xlane`，而是把输入转置之后沿子通道归约；区间里还有 TC VMEM → CMEM、CMEM → TC VMEM 的 DMA（它把中间结果放进了 CMEM），光是两次 DMA 的固定开销就有几百个周期。所以沿通道的归约是手写 kernel 最容易赢的地方之一：`f32[8,128]` 上 100 对 1045。

**沿子通道。** 两边都不经过 XLU，相差不多。Pallas 的树少几条 `vadd`，少 3 到 8 个周期。

**归约到标量。** 这一行 Pallas 比 XLA 慢 59 个周期：Pallas 经过 `vpush`/`spop` 把结果送到标量单元再广播回来，多付了上文那 43 个周期；XLA 的结果一直留在向量单元。只需要把总和写回内存时，可以改写成两步都保留维度的归约 `jnp.sum(jnp.sum(x, axis=1, keepdims=True), axis=0, keepdims=True)`：清单中不再有 `vpush`，周期数是 120，比 XLA 的 128 还少。

### 只改写法：把一条累加链拆成两条

`sum(x, axis=0)` 在 `f32[1024,128]` 上是 295 个周期，128 个 tile 平均每个约 2 个周期，而第 3 节的 `x * 2` 每个 tile 只要 1 个周期。原因在依赖关系：编译器把 128 个 tile 依次加到同一个累加器上，每条 `vadd.8x128.f32` 都用上一条的结果，而 f32 加法的结果 2 个周期后才可用（第 7 节）。127 条加法串成一条链，每条 2 个周期。

加法的次数不能少，但依赖可以改：把 tile 轮流加到两个累加器上，最后把两个累加器相加。两条链互不依赖，可以交替发射：

```python
def chained_sum(chains: int):
    def f(x):
        tiles = [x[start:start + 8] for start in range(0, x.shape[0], 8)]
        sums = tiles[:chains]
        for index, tile in enumerate(tiles[chains:]):
            sums[index % chains] = sums[index % chains] + tile
        while len(sums) > 1:
            sums = [a + b for a, b in zip(sums[0::2], sums[1::2])]
        return jnp.sum(sums[0], axis=0, keepdims=True)
    return f
```

| 写法 | `vadd.8x128.f32` | `vld` | 周期数 |
| --- | ---: | ---: | ---: |
| XLA | 135 | 130 | 303 |
| Pallas，`jnp.sum`（一条链） | 130 | 128 | 295 |
| Pallas，2 条链 | 130 | 128 | 180 |
| Pallas，4 条链 | 130 | 128 | 182 |

指令一条没有少，周期数从 295 降到 180，比 XLA 少四成。两条链之后每个周期发射一条加法；再增加到 4 条链没有用，因为 `vadd.8x128.f32` 只能在 `va1` 槽发射（第 7 节），每个周期最多一条，瓶颈从结果的延迟换成了发射槽。实验的输入是整数值的 f32，求和是精确的，三种写法的结果逐位相同；一般的浮点输入下，改变相加的顺序会改变舍入，结果在末位上可能不同，这是把链拆开时要接受的。

这个例子是本章“数周期”的一个缩影：先从周期数看出每个 tile 2 个周期不对劲，再从清单找到依赖链，然后只改依赖关系，周期数的变化正好等于延迟的变化。

