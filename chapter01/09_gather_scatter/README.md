# gather 与 scatter

前面各节的数据都待在原来的位置：第 s 行第 l 列的输入，计算后仍写回第 s 行第 l 列。本节研究数据位置的改变，按范围从小到大分三层：在一个 TC VREG 内重排（lane 或 sublane 之间），按行从 HBM 取数据，以及把数据写到由索引指定的位置（scatter）。

## TC VREG 内的重排：lane gather

本小节实验[源码](01_pallas_lane_permute.py)、[输出](01_pallas_lane_permute.txt)。

本节的 kernel 有两个输入：数据 `x = f32[8,128]` 和索引 `i = int32[8,128]`。相对于第 1 节的最小 kernel，多一个输入和一个 TC VMEM buffer，计算换成一个函数 `f(x, i)`：

```python
x_vmem[...] = f(x_vmem[...], i_vmem[...])
```

第一个 `f` 是沿通道的 gather，每个输出位置从同一行的任意一列取值：

```python
f = lambda x, i: jnp.take_along_axis(x, i, axis=1)   # y[s,l] = x[s, i[s,l]]
```

清单中的计算指令：

```tpuasm
{ va0: vlt.8x128.s32 vm0, v0, 0 ;  va1: vadd.8x128.s32 v1, 128, v0 }
{ va0: vsel.8x128 v2, vm0, v1, v0 }          # 负索引加 128
{ vx0: vsetperm.2.all.u8 pcr0, v2 }          # 用索引设置重排模式
{ vx0: vperm.0.8x128 trf0, v3 }              # 按模式重排数据
{ vr0: vpop.8x128 v4, trf0 }                 # 取回结果
```

重排由跨通道单元 XLU（cross-lane unit）完成。它与第 7 节的 EUP 一样是“提交—取回”的形式：`vsetperm` 把索引装进重排控制寄存器 `pcr0`，`vperm` 把数据送入 XLU，结果进入队列 `trf0`，再由 `vpop` 取回。一次 gather 处理一整个 TC VREG：8 行各自按自己的 128 个索引重排。

前三条指令处理负索引：`take_along_axis` 允许 `-1` 表示最后一列，编译器先把负数加 128。实验的索引都非负，但编译器无法知道，这几条指令照样存在。

### 周期数：XLU 的往返延迟与发射间隔

本小节实验[源码](06_tpuasm_xlu_permute_latency.py)、[输出](06_tpuasm_xlu_permute_latency.txt)。

XLU 是本章遇到的第二个提交—取回的单元，它的两个时间参数用第 7 节测 EUP 的办法测：提交与取回之间插入 d − 1 个 `vnop`，看 R2 − R0 从哪个 d 开始增加；再连续提交 k 次，看每多一次多几个周期。重排的模式事先用 `vsetperm` 装好，不计入区间：

```python
'vperm（通道重排）': ('vx0: vperm.0.8x128 trf0, v10', bundle('vx0: vsetperm.2.all.u8 pcr0, v10') + bundle('misc: vnop') * 16),
...
show_latency(probe, f'XLU {name} → vpop trf0', instruction, 'vr0: vpop.8x128 v11, trf0', setup, distances=(1, 8, 40, 64, 68, 69, 70, 72))
```

| d | 1 | 8 | 40 | 64 | 68 | 69 | 70 | 72 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `vperm` → `vpop`，R2 − R0 | 82 | 82 | 82 | 82 | 82 | 83 | 84 | 86 |

d ≤ 68 时 R2 − R0 都是 82：`vpop` 在等结果，中间放多少个 `vnop` 都一样。82 减去固定的 13 个周期，**从提交到可以取回是 69 个周期**。下一小节的通道循环移位 `vrot` 读数完全相同，也是 69。

| 连续提交 k 次，再取回 k 次 | k = 1 | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: |
| `vperm`，R2 − R0 | 82 | 90 | 106 | 138 |

每多一次多 8 个周期：**同一个 XLU 每 8 个周期接受一次提交**（XLU 每个周期处理 TC VREG 的一行）。这两个数说明了 XLU 的性质：一次重排要等 69 个周期，这段时间里向量 ALU 能发射上百条指令；但 69 个周期的等待中可以继续提交，8 个 TC VREG 一起重排只要 138 − 13 = 125 个周期，而不是 8 × 69。

所以用 XLU 的代码有两种极限：结果前后依赖时，时间是往返的次数乘 69；互不依赖时，时间是提交的条数乘 8。第 13、15 节会分别遇到这两种情形。

## sublane gather：Pallas 不支持，但可以合成

第二个 `f` 把索引轴改为 sublane：

```python
f = lambda x, i: jnp.take_along_axis(x, i, axis=0)   # y[s,l] = x[i[s,l], l]
```

编译失败：

```text
Sublane gather not supported by this TPU generation
```

TPU v4 的 XLU 只能沿通道方向按索引重排。但沿子通道方向有另一条指令：`vrot.slane.down` 把 8 个子通道循环移动一位，第 s 个子通道的内容移到第 s − 1 个，第 0 个绕到第 7 个。有了“整体移动”和“逐元素选择”，就能合成任意的 sublane gather：第 k 轮把 x 这样移动 k 次，这时第 s 行第 l 列放的是 `x[(s+k)%8, l]`；凡是 `i[s,l] == (s+k)%8` 的位置，就选这一轮的值。

```python
def sublane_gather_by_rotation(x, indices):
    sublane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 0)
    result = jnp.zeros_like(x)
    rotated = x
    for k in range(8):
        result = jnp.where(indices == (sublane + k) % 8, rotated, result)
        rotated = pltpu.roll(rotated, 7, axis=0)
    return result
```

`jax.lax.broadcasted_iota(dtype, shape, axis)` 生成沿 axis 递增的序号，这里得到每个位置的行号。`pltpu.roll(x, shift, axis)` 是循环移位，语义与 `np.roll` 相同：`out[s] = x[(s - shift) % 8]`，`shift=7` 时 `out[s] = x[(s + 1) % 8]`，正是一条 `vrot.slane.down`。

这个函数数值正确，共 40 条计算指令，其中 7 条 `vrot.slane.down`、8 条 `vsel`，其余是比较和行号计算，而 lane gather 只有 3 条 XLU 指令。但指令多不等于慢。放进第 3 节的 TC VMEM 边界（本节最后一小节），这 40 条指令是 62 个周期，lane gather 是 101 个周期：40 条指令都在向量 ALU 中，每个周期可以发射两条，最长的依赖链是 7 条各 2 个周期的移位；lane gather 的 3 条指令则要等 XLU 的 69 个周期。只有一个 TC VREG 时，合成的写法反而快。TC VREG 多了之后情况反过来：合成的写法每个 TC VREG 都要占用向量 ALU 二十来个周期，XLU 每 8 个周期就能接受一个。

它说明了本教程反复出现的两件事：编译器拒绝，不代表硬件做不到；数指令条数，不如数周期。下一小节会看到，硬件其实还有一条更直接的通路。

## sublane gather 的硬件通路：经过 TC VMEM 的索引读写

本小节实验[源码](05_tpuasm_iar.py)、[输出](05_tpuasm_iar.txt)。

XLU 只能沿通道方向重排，但沿子通道方向按索引取数并不需要 XLU。TC VMEM 的地址以行（子通道）为单位，一条普通的 `vld` 从地址 A 起读 8 行，第 s 个子通道读的是第 A + s 行。TPU v4 另有两个索引寄存器 `iar0`、`iar1`，带索引的读写指令让每个元素在行号上再加一个自己的偏移，通道号不变：

```tpuasm
{ vst: vsetiar.raw iar0, v13 }             # 把 v13 的 8×128 个整数装进 iar0
{ vld: vld.iar0.8x128 v14, [vmem:0x80] }   # v14[s,l] = 第 0x80 + s + v13[s,l] 行的第 l 列
```

这两条指令都不在 Pallas 能生成的范围内，本小节用第 3 节的 `LccProbe` 直接执行。先在 TC VMEM 中写一张表，第 r 行第 l 列的值是 128r + l，从读回的值就能看出每个元素来自哪一行、哪一列。`vsetiar` 有三种形式，区别只在偏移怎样取：

```text
  vsetiar.raw：通道号不变：True；行号等于 s + i[s,l]：True
  vsetiar.lane：通道号不变：True；行号等于 s + i[0,l]：True
  vsetiar.sublane：通道号不变：True；行号等于 s + 16 × i[s,l]：True
```

`.raw` 每个元素用自己的索引；`.lane` 整列都用第 0 个子通道的索引；`.sublane` 把索引乘以 16 行。三种形式读到的都是同一列的数据，所以这是沿子通道（以及跨 tile）方向的 gather。

有了它，`y[s,l] = x[i[s,l], l]` 只需要把 x 写进 TC VMEM、算出偏移、读一次。第 s 个子通道要读第 i 行，偏移是 i − s；为了不出现负数，把 x 连续写两遍，偏移取 (i − s) mod 8，读到的是第 i 行或第 i + 8 行，内容相同：

```tpuasm
{ vst: vst.8x128 [vmem:0x80], v10 }        # x 写两遍：第 0–7 行与第 8–15 行
{ vst: vst.8x128 [vmem:0x88], v10 }
{ va0: vsub.8x128.s32 v12, v11, v3 }       # v11 = 索引 i，v3 = 子通道号 s
{ va0: vand.8x128.u32 v13, 0x7, v12 }      # (i − s) mod 8
{ vst: vsetiar.raw iar0, v13 }
{ vld: vld.iar0.8x128 v14, [vmem:0x80] }
```

结果与 `np.take_along_axis(x, i, axis=0)` 相同。6 条指令，合成写法是 40 条。

写的方向也有对应的指令。`vst.iar` 把第 s 个子通道、第 l 个通道的元素写到第 A + s + 偏移行的第 l 列，这是沿子通道方向的 scatter。`y[i[s,l], l] = x[s,l]`（每列的 i 是 0–7 的一个排列）用同样的偏移 (i − s) mod 8 写进预先填好的两个 tile，再把两个 tile 合起来，结果与 `np.put_along_axis` 相同：

```tpuasm
{ vst: vsetiar.raw iar0, v13 }
{ vst: vst.iar.8x128 [vmem:0x80], iar0, v10 }
```

scatter 必须回答“多个元素写同一个位置”怎么办。硬件的回答是不允许：一条 `vst.iar` 中有两个元素写到同一行同一列时，TensorCore 停机，主机得到 `RuntimeUnexpectedCoreHalt`（[研究报告 62](../../../pallas-tpu-readings-dev/research_reports/62_pallas_top_k_correctness_and_speed.md) 的配套实验）。索引必须保证同一列的 8 个子通道写到互不相同的行。同一组实验还表明 `vst.iar` 的偏移可以是负数，写到地址之前的行。

这些指令与普通的读写一样快：

```text
  vld，紧接着使用：15
  vld.iar0，紧接着使用：15
  vsetiar.raw、vld.iar0，紧接着使用：19
  vst 之后从同一个 tile vld.iar0，紧接着使用：20
  16 条 vld：29
  16 条 vld.iar0：29
  16 条 vst：29
  16 条 vst.iar：29
```

表中是 R2 − R0，其中 13 个周期是读数与排空的固定开销。`vld.iar` 与 `vld` 的结果都在下一个周期可用，连续发射都是每周期一条；`vst.iar` 与 `vst` 相同；多一条 `vsetiar` 多 4 个周期。上面整段 sublane gather 连同一条使用结果的指令是 24，扣除固定开销为 11 个周期，而 lane gather 的 `vperm` 要等 XLU 的 69 个周期。

所以在 TPU v4 上，沿子通道方向按索引重排比沿通道方向便宜，只是 Pallas 目前不生成这些指令，需要用 tpuasm 写。

## 固定位移的循环移位

第三、四个 `f` 是固定位移的循环移位：

```python
f = lambda x, i: pltpu.roll(x, 5, axis=1)   # lane 方向移动 5 位
f = lambda x, i: pltpu.roll(x, 3, axis=0)   # sublane 方向移动 3 位
```

通道方向的移位是一次 XLU 操作，位移量放在标量寄存器中：

```tpuasm
{ vx0: vrot.0.8x128 trf0, v0, s16 }
{ vr0: vpop.8x128 v1, trf0 }
```

子通道方向则是 5 条 `vrot.slane.down.8x128.u32`。这条指令没有位移量操作数，每次只能把内容移到编号小 1 的 sublane，也没有反方向的指令；`roll(x, 3, axis=0)` 要把内容移到编号大 3 的 sublane，等于朝反方向移 5 次，于是要 5 条。上面合成 sublane gather 时，每轮的 `roll(rotated, 7, axis=0)` 只需一条。

两个方向的开销因此相反：通道方向是一条指令，但要等 XLU 的 69 个周期（上文）；子通道方向是 5 条指令，每条的结果 2 个周期后可用（第 7 节），共 10 个周期。

## 按行 gather：每行一次 DMA

本小节实验[源码](02_pallas_row_gather_dma.py)、[输出](02_pallas_row_gather_dma.txt)。

当要取的是整行，而且数据还在 HBM 中时，重排就不必经过 TC VREG：DMA 本身就能按任意行号搬运。kernel 从 `f32[64,128]` 中按 SMEM 里的 8 个行号各取一行，拼成 `f32[8,128]`：

```python
pltpu.async_copy(r_hbm, r_smem, sem).wait()
copies = [pltpu.async_copy(t_hbm.at[pl.ds(r_smem[j], 1)], o_vmem.at[pl.ds(j, 1)], sem) for j in range(8)]
for copy in copies:
    copy.wait()
```

行号为 `[5, 3, 60, 0, 17, 17, 42, 9]` 时结果正确，重复的行号也没有问题。清单中是 8 条 `sld` 读行号、8 条 `length=1` 的 `dma.simple`，最后只有一条 `vwait.ge [sflag:52], 8`：8 次 DMA 共用一个信号量，编译器把 8 次等待合并成一次，这正是第 4 节所说的等待合并。

每次 DMA 只搬一个 granule，即一行 128 个 f32。表只有一列 tile，按第 4、8 节的结论，任意一行都是一个连续的 granule，起点不必与 tile 对齐。这是在 HBM 中按行查表的基本形式；表更宽时，一行在 HBM 中分散在各个列 tile 里，Mosaic 对这种窗口的对齐要求见第 4 节。

第 4 节测过，每次 DMA 有四百多个周期的固定开销，而同时进行的几次 DMA 的固定开销可以重叠。按行逐次发起并等待的 DMA 越多，固定开销越大；能一起发起的就一起发起、最后统一等待。第二章第 2 节给出完整的 DMA 开销模型。

## scatter：写的位置由索引决定

本小节实验[源码](04_pallas_row_scatter_dma.py)、[输出](04_pallas_row_scatter_dma.txt)。

scatter 把索引当作写的位置：

```text
out[s, i[s,l]] = update[s,l]
```

它比 gather 多一个问题：多个元素可能写到同一个位置。此时是覆盖、相加还是取最大，必须由算法规定。下面按与 gather 相同的两个层次看 scatter：TC VREG 内按 lane，以及 HBM 中按行。

### 沿通道：没有 scatter 指令

在 kernel 中直接写 scatter：

```python
rows = jax.lax.broadcasted_iota(jnp.int32, x.shape, 0)
x_vmem[...] = jnp.zeros(x.shape, x.dtype).at[rows, i_vmem[...]].set(x_vmem[...])
```

编译失败：

```text
Unimplemented primitive in Pallas TPU lowering for tc: scatter.
```

这与硬件一致。XLU 的 `vperm` 为每个输出位置指定“从哪个通道读”，是 gather 的形式；它回答不了“多个写者写同一个位置”的问题，所以 XLU 没有 scatter 指令。沿子通道方向则有上文的 `vst.iar`，它用停机来回答这个问题。

有一种重要的特殊情况：每行的索引恰好是 0–127 的一个排列，没有重复。这时先求出逆排列 `inverse[s, i[s,l]] = l`，scatter 就变成 gather：

```text
out[s, d] = update[s, inverse[s, d]]
```

实验在主机上用 `np.argsort(permutation, axis=1)` 求出逆排列，kernel 只改一行，做一次 lane gather：

```python
x_vmem[...] = jnp.take_along_axis(x_vmem[...], i_vmem[...], axis=1)
```

结果与 NumPy 的 `np.put_along_axis` 完全相同，清单中只有一次重排：`vsetperm`、`vperm`、`vpop` 各 1 条，加上处理负索引的 `vlt`、`vadd`、`vsel`。逆排列本身也是一次 scatter；如果排列在 kernel 中才得到，求逆排列要另想办法，例如按值排序（本节不展开）。手写 kernel 时，应当先识别出这类无冲突、且排列事先已知的结构，把写的重排改成读的重排。

### HBM 中：每行一次 DMA

整行的 scatter 与按行 gather 对称：DMA 的目的地址由行号决定。这里要修改一个已有的表，所以表作为 jax Ref 传入 kernel（第 14 节详细介绍这种写法），kernel 没有输出：

```python
def kernel(u_hbm: Ref, r_hbm: Ref, t_hbm: Ref, u_vmem: Ref, r_smem: Ref, sem: Ref) -> None:
    pltpu.async_copy(u_hbm, u_vmem, sem).wait()
    pltpu.async_copy(r_hbm, r_smem, sem).wait()
    # 与按行 gather 对称：源是更新的第 j 行，目的地是表的第 r[j] 行。8 次 DMA 先全部发出，再一起等待。
    copies = [pltpu.async_copy(u_vmem.at[pl.ds(j, 1)], t_hbm.at[pl.ds(r_smem[j], 1)], sem) for j in range(8)]
    for copy in copies:
        copy.wait()
```

与按行 gather 相比只交换了 DMA 的源和目的。行号互不相同时结果正确。清单中，每个目的地址由 `sadd` 把行号（已换算成 granule 的偏移）加到表的起点上算出，随后是一条 `length=1` 的 `dma.simple`，8 条 DMA 之后只有一次 `vwait.ge [sflag:52], 8`：

```tpuasm
{ s0: sadd.s32 s24, s19, s2 ; ... }
{ s0: dma.simple [hbm:s24], [vmem:s13], length=1, dst_flag=[sflag:52] }
...
{ misc: vwait.ge [sflag:52], 8 ; ... }
```

行号重复时，两次 DMA 写同一行。实验让第 4、5 个更新都写到第 17 行，运行 20 次：

```text
第 17 行最终等于哪个更新：{5: 20}
```

20 次都是后发出的第 5 个留下。但这只是观察：两次 DMA 同时在途，它们写入 HBM 的先后不是公开接口承诺的语义。需要确定的结果时，要么保证行号不重复（例如先在主机或标量单元上合并重复的行），要么让写同一行的 DMA 先后发出、中间等待完成。需要累加而不是覆盖时（例如梯度的 scatter-add），DMA 做不到，只能把行读进 TC VMEM、相加、再写回，并且同一行的更新必须串行。

## 对照：原生 XLA

本小节实验[源码](03_jax_gather_scatter.py)、[输出](03_jax_gather_scatter.txt)。

同样的三个操作交给 XLA：

| 操作 | HLO 段 bundle 数 | 要点 |
| --- | ---: | --- |
| 按行 take：`table[rows]` | 138 | 拆成 5 个 fusion，含索引截断与比较 |
| lane gather：`take_along_axis` | 537 | 64 条 `vperm`、130 条 `vpop` |
| lane scatter（索引为排列） | 1939 | 256 条 `vst.msk`、257 条 `veq`，只有 4 条 `vperm` |

XLA 对 lane gather 没有使用一次 `vperm` 完成整个 TC VREG 的写法，而是更通用、更长的展开。对 scatter，即使通过 `unique_indices=True` 告诉它索引互不重复，XLA 也没有把它改写成逆排列的 gather，而是逐个位置比较、用带掩码的 store 写回，产生了 1939 个 bundle 的程序。

这三个对照都说明同一件事：XLA 的通用实现必须对任意索引正确，开销可能比手写版本高出一到两个数量级。手写 kernel 的价值，就在于利用 XLA 不知道的结构，例如“索引在一个 tile 内”或“索引构成排列”。

### 周期数

本小节实验[源码](07_jax_pallas_cycles.py)、[输出](07_jax_pallas_cycles.txt)。

TC VREG 内的几种重排放进第 3 节的 TC VMEM 边界。Pallas 一侧是本节的写法；XLA 一侧是 `jnp.take_along_axis` 与 `jnp.roll`。两边的结果逐元素相同：

| 运算 | 输入 | XLA | Pallas | Pallas 的做法 |
| --- | --- | ---: | ---: | --- |
| lane gather | `f32[8,128]` | 9213（9211–9218） | 101 | 1 条 `vperm` |
| lane gather | `f32[64,128]` | 104627（104623–104630） | 151 | 8 条 `vperm`，分给两个 XLU |
| sublane gather | `f32[8,128]` | 9294（9291–9298） | 62 | 循环移位与选择，40 条 ALU 指令 |
| lane 循环移位 5 位 | `f32[8,128]` | 214（212–215） | 90 | 1 条 `vrot` |
| lane 循环移位 5 位 | `f32[64,128]` | 267（266–267） | 115 | 8 条 `vrot` |
| sublane 循环移位 3 位 | `f32[8,128]` | 133（133–134） | 46 | 5 条 `vrot.slane.down` |

- **Pallas 的周期数可以由延迟算出来。** 一个 TC VREG 的 lane gather 是 101 个周期：这种有等待的区间固定部分是 19 个周期（第 3 节），等 XLU 69 个，其余十来个是读入、处理负索引、装入模式、取回和写出。8 个 TC VREG 是 151 个，只多 50 个周期：8 条 `vperm` 分给两个 XLU，后面的在前面的等待期间提交。sublane 循环移位是 46 个：5 条移位串成的链是 10 个周期，加上读入和写出，其中几个周期同样与外壳标量一侧的指令重叠。
- **XLA 的 gather 慢两到三个数量级。** 区间里有 TC VMEM → CMEM、CMEM → SMEM 的 DMA，以及 192 条把标量广播成 TC VREG 的 `vmov`：XLA 的 gather 是一个对任意索引都成立的通用实现，要经过 SMEM 和标量单元，`f32[64,128]` 要十万个周期。它的循环移位由 slice 和 pad 拼成，没有用 `vrot`，也比 Pallas 慢一倍以上。XLA 的读数有几个周期的波动，因为区间里有 DMA。

上文用索引寄存器手写的 sublane gather 只要 11 个周期（不含外壳），是这张表里最快的重排。

