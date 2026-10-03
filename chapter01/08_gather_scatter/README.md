# gather 与 scatter

前面各节的数据都待在原来的位置：第 s 行第 l 列的输入，计算后仍写回第 s 行第 l 列。本节研究数据位置的改变，按范围从小到大分三层：在一个 TC VREG 内重排（lane 或 sublane 之间），按行从 HBM 取数据，以及把数据写到由索引指定的位置（scatter）。

## TC VREG 内的重排：lane gather

本小节实验[源码](01_pallas_lane_permute.py)、[输出](01_pallas_lane_permute.txt)。

本节的 kernel 有两个输入：数据 `x = f32[8,128]` 和索引 `i = int32[8,128]`。相对于第 1 节的最小 kernel，多一个输入和一个 TC VMEM buffer，计算换成一个函数 `f(x, i)`：

```python
x_vmem[...] = f(x_vmem[...], i_vmem[...])
```

第一个 `f` 是沿 lane 的 gather，每个输出位置从同一行的任意一列取值：

```python
f = lambda x, i: jnp.take_along_axis(x, i, axis=1)   # y[s,l] = x[s, i[s,l]]
```

清单中的计算指令：

```text
{ va0: vlt.8x128.s32 vm0, v0, 0 ;  va1: vadd.8x128.s32 v1, 128, v0 }
{ va0: vsel.8x128 v2, vm0, v1, v0 }          # 负索引加 128
{ vx0: vsetperm.2.all.u8 pcr0, v2 }          # 用索引设置重排模式
{ vx0: vperm.0.8x128 trf0, v3 }              # 按模式重排数据
{ vr0: vpop.8x128 v4, trf0 }                 # 取回结果
```

重排由跨 lane 单元完成，本教程称为 XLU。它与上一节的 EUP 一样是“提交—取回”的形式：`vsetperm` 把索引装进重排控制寄存器 `pcr0`，`vperm` 把数据送入 XLU，结果进入队列 `trf0`，再由 `vpop` 取回。一次 gather 处理一整个 TC VREG：8 行各自按自己的 128 个索引重排。`vsetperm` 的 `.all` 表示把整个 TC VREG 的索引都装进去；`u8` 说明每个索引按 8 位存放，足以表示 0–127。第三章第 6 节测得，`vperm` 提交之后 69 个周期才能取回，连续提交每 8 个周期一次：单次 gather 的等待很长，但许多个 TC VREG 一起重排时，XLU 每 8 个周期就能完成一个。

前三条指令处理负索引：`take_along_axis` 允许 `-1` 表示最后一列，编译器先把负数加 128。实验的索引都非负，但编译器无法知道，这几条指令照样存在。

## sublane gather：硬件没有，但可以合成

第二个 `f` 把索引轴改为 sublane：

```python
f = lambda x, i: jnp.take_along_axis(x, i, axis=0)   # y[s,l] = x[i[s,l], l]
```

编译失败：

```text
Sublane gather not supported by this TPU generation
```

TPU v4 的 XLU 只能沿 lane 方向按索引重排。但沿 sublane 方向有另一条指令：`vrot.slane.down` 把 8 个 sublane 循环移动一位。有了“整体移动”和“逐元素选择”，就能合成任意的 sublane gather：第 k 轮把 x 向上循环移动 k 行，这时第 s 行第 l 列放的是 `x[(s+k)%8, l]`；凡是 `i[s,l] == (s+k)%8` 的位置，就选这一轮的值。

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

`jax.lax.broadcasted_iota(dtype, shape, axis)` 生成沿 axis 递增的序号，这里得到每个位置的行号。`pltpu.roll(x, shift, axis)` 是循环移位，语义与 `np.roll` 相同：`out[s] = x[(s - shift) % 8]`，`shift=7` 即向上移动一行。

这个函数数值正确，共 40 条计算指令，其中 7 条 `vrot.slane.down`、8 条 `vsel`，其余是比较和行号计算。与 lane gather 的 3 条 XLU 指令相比，代价高出一个数量级。它说明了本教程反复出现的一种情况：编译器拒绝，不代表硬件做不到；但硬件没有直接支持的操作，合成出来通常很贵。设计数据布局时，应当让需要按索引重排的轴落在 lane 方向。

## 固定位移的循环移位

第三、四个 `f` 是固定位移的循环移位：

```python
f = lambda x, i: pltpu.roll(x, 5, axis=1)   # lane 方向移动 5 位
f = lambda x, i: pltpu.roll(x, 3, axis=0)   # sublane 方向移动 3 位
```

lane 方向的移位是一次 XLU 操作，位移量放在标量寄存器中：

```text
{ vx0: vrot.0.8x128 trf0, v0, s16 }
{ vr0: vpop.8x128 v1, trf0 }
```

sublane 方向则是 5 条 `vrot.slane.down.8x128.u32`。这条指令没有位移量操作数，每次只向下循环移动一个 sublane；向上移动 3 行等于向下移动 5 行，于是要 5 条。上面合成 sublane gather 时，每轮 `roll(rotated, 7, axis=0)` 正好是向下移动一行，所以只需一条。

## 按行 gather：每行一次 DMA

本小节实验[源码](02_pallas_row_gather_dma.py)、[输出](02_pallas_row_gather_dma.txt)。

当要取的是整行，而且数据还在 HBM 中时，重排就不必经过 TC VREG：DMA 本身就能按任意行号搬运。kernel 从 `f32[64,128]` 中按 SMEM 里的 8 个行号各取一行，拼成 `f32[8,128]`：

```python
pltpu.async_copy(r_hbm, r_smem, sem).wait()
copies = [pltpu.async_copy(t_hbm.at[pl.ds(r_smem[j], 1)], o_vmem.at[pl.ds(j, 1)], sem) for j in range(8)]
for copy in copies:
    copy.wait()
```

行号为 `[5, 3, 60, 0, 17, 17, 42, 9]` 时结果正确，重复的行号也没有问题。清单中是 8 条 `sld` 读行号、8 条 `length=1` 的 `dma.simple`，最后只有一条 `vwait.ge [sflag:52], 8`：8 次 DMA 共用一个信号量，编译器把 8 次等待合并成一次，这正是第 3 节所说的等待合并。

每次 DMA 只搬一个 granule，即一行 128 个 f32。表只有一列 tile，按第 3、7 节的结论，任意一行都是一个连续的 granule，起点不必与 tile 对齐。这是在 HBM 中按行查表的基本形式；表更宽时，一行在 HBM 中分散在各个列 tile 里，Mosaic 对这种窗口的对齐要求见第 3 节。

> 暂且可以理解为：每次 DMA 都有数百个周期的固定开销，按行逐次发起的 DMA 越多，固定开销越大，但多次 DMA 可以同时进行。第二章第 2 节给出 DMA 代价模型。

## scatter：写的位置由索引决定

本小节实验[源码](04_pallas_row_scatter_dma.py)、[输出](04_pallas_row_scatter_dma.txt)。

scatter 把索引当作写的位置：

```text
out[s, i[s,l]] = update[s,l]
```

它比 gather 多一个问题：多个元素可能写到同一个位置。此时是覆盖、相加还是取最大，必须由算法规定。下面按与 gather 相同的两个层次看 scatter：TC VREG 内按 lane，以及 HBM 中按行。

### TC VREG 内：没有 scatter 指令

在 kernel 中直接写 scatter：

```python
rows = jax.lax.broadcasted_iota(jnp.int32, x.shape, 0)
x_vmem[...] = jnp.zeros(x.shape, x.dtype).at[rows, i_vmem[...]].set(x_vmem[...])
```

编译失败：

```text
Unimplemented primitive in Pallas TPU lowering for tc: scatter.
```

这与硬件一致。XLU 的 `vperm` 为每个输出位置指定“从哪个 lane 读”，是 gather 的形式；它回答不了“多个写者写同一个位置”的问题，所以不存在一条 scatter 指令。

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

整行的 scatter 与按行 gather 对称：DMA 的目的地址由行号决定。这里要修改一个已有的表，所以表作为 jax Ref 传入 kernel（第 13 节详细介绍这种写法），kernel 没有输出：

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

```text
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
| 按行 take：`table[rows]` | 214 | 拆成 5 个 fusion，含索引截断与比较 |
| lane gather：`take_along_axis` | 568 | 64 条 `vperm`、130 条 `vpop` |
| lane scatter（索引为排列） | 2087 | 256 条 `vst.msk`、257 条 `veq`，没有 `vperm` |

XLA 对 lane gather 没有使用一次 `vperm` 完成整个 TC VREG 的写法，而是更通用、更长的展开。对 scatter，即使通过 `unique_indices=True` 告诉它索引互不重复，XLA 也没有把它改写成逆排列的 gather，而是逐个位置比较、用带掩码的 store 写回，产生了 2087 个 bundle 的程序。

这三个对照都说明同一件事：XLA 的通用实现必须对任意索引正确，代价可能比手写版本高出一到两个数量级。手写 kernel 的价值，就在于利用 XLA 不知道的结构，例如“索引在一个 tile 内”或“索引构成排列”。
