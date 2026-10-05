# 拼接

拼接不做任何算术，只决定数据放在哪里。所以它最能说明一个道理：数据的位置由谁来安排，向量单元、XLU 还是 DMA，开销差别很大。本节比较两种写法：在 TC VMEM 中对值做 `jnp.concatenate`，以及让 DMA 把两个输入直接写进输出的两个窗口。再只改 shape，看不对齐时会发生什么；最后看在已有数组的运行时位置追加数据，怎样只付新数据的开销。

## 两种写法

本小节实验[源码](01_pallas_concatenate.py)、[输出](01_pallas_concatenate.txt)。

第一种写法与第 1 节的最小 kernel 结构相同，两个输入各进一个 TC VMEM buffer，计算换成拼接：

```python
o_vmem[...] = jnp.concatenate([a_vmem[...], b_vmem[...]], axis=axis)
```

第二种写法不分配 TC VMEM，两次 DMA 直接从输入 HBM 写到输出 HBM 的两个窗口：

```python
if axis == 0:
    windows = (o_hbm.at[pl.ds(0, rows), :], o_hbm.at[pl.ds(rows, rows), :])
else:
    windows = (o_hbm.at[:, pl.ds(0, columns)], o_hbm.at[:, pl.ds(columns, columns)])
copies = [pltpu.async_copy(a_hbm, windows[0], sem), pltpu.async_copy(b_hbm, windows[1], sem)]
for copy in copies:
    copy.wait()
```

两次 DMA 先后发出、一起等待，可以同时进行（第 3 节）。

## 对齐的 shape

两个 `f32[8,128]` 沿 axis=0 或 axis=1 拼接，两种写法都正确：

| 写法 | 向量指令 | DMA |
| --- | --- | --- |
| TC VMEM 中 `jnp.concatenate` | 2 条 `vld`、2 条 `vst` | 2 次输入、1 次输出 |
| DMA 写入输出窗口 | 无 | 2 次 HBM→HBM 的 `dma.general` |

在 TC VMEM 中拼接时，计算部分只是把两个 TC VREG 原样读出、写到输出 buffer 的两个位置，没有任何运算；但它仍要先把数据搬进 TC VMEM、再搬出来。拼接是“重新安排位置”，而 DMA 本身就能决定写到哪里，所以第二种写法完全不经过 TensorCore 的向量单元。

## 只改 shape：不对齐的行

两个 `f32[9,128]` 沿 axis=0 拼接成 `f32[18,128]`：

- TC VMEM 中拼接：b 的第 0 行要写到输出的第 9 行，即第二个 tile 的第 1 个子通道，所以 b 的每个 tile 都要移动一个子通道。清单中有 14 条 `vrot.slane.down` 和 2 条 `vsel`：

  ```tpuasm
  { va0: vlaneseq.8x128.u32 v8 ; vld: vld.8x128 v0, [vmem:0x0] }                  # a 的第 0–7 行
  { va1: vshrl.8x128.s32 v13, v8, 0x7 ; vst: vst.8x128 [vmem:0x20], v0 ; vld: vld.8x128 v1, [vmem:0x10] }   # 原样写出；读 b 的第 0–7 行
  { va0: vrot.slane.down.8x128.u32 v2, v1 ; va1: veq.8x128.s32 vm0, v13, 0 ; vld: vld.8x128 v3, [vmem:0x18, sm=1] }   # vm0 = 第 0 个 sublane；读 b 的第 8 行
  { va0: vrot.slane.down.8x128.u32 v4, v3 ; vld: vld.8x128 v16, [vmem:0x8, sm=1] }   # 读 a 的第 8 行
  ...                                                                              # 两条链各再转 6 次，交替排列
  { va0: vsel.8x128 v20, vm0, v16, v18 }                                           # 输出的第 8–15 行
  { va0: vsel.8x128 v21, vm0, v18, v19 ; vst: vst.8x128 [vmem:0x28], v20 }         # 输出的第 16、17 行
  { vst: vst.8x128 [vmem:0x30, sm=3], v21 }
  ```

  `vrot.slane.down` 每次只把 8 个子通道循环移动一个位置：第 s 个子通道的内容移到第 s − 1 个，第 0 个绕到第 7 个。这里需要的是反方向移一个位置（b 的第 0 行从第 0 个子通道移到第 1 个），硬件没有反方向的指令，于是转 7 次：b 的两个 tile 各一条 7 次的链，共 14 条。转完之后，`v18` 的第 1–7 个子通道是 b 的第 0–6 行，第 0 个子通道是绕回来的第 7 行；`v19` 的第 1 个子通道是 b 的第 8 行。两条 `vsel` 用“第 0 个子通道”的掩码拼出输出的后两个 tile：第一个 tile 的第 0 个子通道取 a 的第 8 行，其余取 `v18`；第二个 tile 的第 0 个子通道取 `v18` 中绕回来的第 7 行，第 1 个取 `v19`，用 `sm=3` 只写这两行。每条 `vrot.slane.down` 的结果要 2 个周期后才可用（第三章第 5 节），一条 7 次的链是 14 个周期。
- DMA 写入窗口：两次 DMA 的长度都是 9 个 granule，第二次从第 9 行写起。数组只有一列 tile（第 3 节），任意行窗口都连续，DMA 不需要额外工作。

## 只改 shape：不对齐的列

两个 `f32[8,129]` 沿 axis=1 拼接成 `f32[8,258]`：

- TC VMEM 中拼接：b 的第 0 列要放到输出的第 129 列，跨越 lane 边界，需要 XLU 的循环移位（`vrot` 加 `vpop`，第 8 节）加掩码选择和带掩码的 store：

  ```tpuasm
  { s0: simm.s32 s16, 1 ; va0: vlaneseq.8x128.u32 v2 ; vld: vld.8x128 v0, [vmem:0x10] }   # b 的第 0–127 列
  { va0: vand.8x128.u32 v3, 0x7f, v2 ; vld: vld.8x128 v1, [vmem:0x18] ; vx0: vrot.2.8x128 trf0, v0, s16 }   # 沿 lane 循环右移 1
  { va0: veq.8x128.s32 vm0, v3, 0 ; va1: vlt.8x128.s32 vm1, v3, 2 ; vld: vld.8x128 v4, [vmem:0x8] }         # vm0 = 第 0 个 lane，vm1 = 前 2 个 lane
  { vld: vld.8x128 v7, [vmem:0x0] }
  { vst: vst.8x128 [vmem:0x20], v7 }                    # 输出的第 0–127 列 = a 的第 0–127 列
  { vx0: vrot.0.8x128 trf0, v1, s16 }                   # b 的第 128 列所在的 tile 也右移 1
  { vr0: vpop.8x128 v5, trf0 }
  { va0: vsel.8x128 v6, vm0, v4, v5 }                   # 输出的第 128–255 列
  { vst: vst.8x128 [vmem:0x28], v6 }
  { vr0: vpop.8x128 v8, trf0 }
  { va0: vsel.8x128 v9, vm0, v5, v8 }                   # 输出的第 256、257 列
  { vst: vst.msk.8x128 [vmem:0x30], vm1, v9 }
  ```

  输出有 3 个 tile。第一个是 a 的第 0–127 列，原样写出。第二个是输出的第 128–255 列：第 0 个通道是 a 的第 128 列，其余 127 个通道是 b 的第 0–126 列。b 的第一个 tile 沿通道循环右移 1 之后（`v5`），b 的第 j 列在第 j + 1 个通道，第 127 列绕回第 0 个通道；`vsel` 把第 0 个通道换成 a 的第 128 列。第三个 tile 只有两列：第 256 列是 b 的第 127 列，正是 `v5` 中绕回第 0 个通道的那一个；第 257 列是 b 的第 128 列，来自 b 的第二个 tile 右移 1 之后的第 1 个通道（`v8`）。最后用 `vst.msk` 只写前 2 个通道。两次 `vrot` 各要 69 个周期才能取回（第三章第 5 节），这段拼接的时间主要花在等 XLU 上。
- DMA 写入窗口：编译失败，`Slice sizes along tiled dimensions must be aligned to tiles`。

列方向上，一个 granule 是一个子通道的 128 个通道。DMA 的地址和长度都以 granule 为单位（第 3 节），表达不了从某个 granule 中间开始的窗口，所以从第 129 列开始的窗口不能由 DMA 描述。这与行方向不同：行方向的最小单位是一个 granule（一行），列方向的最小单位是 128 列。所以不对齐的列只能交给 XLU 在 TC VREG 内移动。

## 只改位置：在运行时位置追加

本小节实验[源码](03_pallas_append_in_place.py)、[输出](03_pallas_append_in_place.txt)。

上面的拼接都生成一个新数组。另一种常见的需要是：一个已有的大数组，每次在它的某个位置写入几行新数据，位置在运行时才知道，例如每次追加到上一次的末尾。如果每次都用 `jnp.concatenate` 或 `.at[].set()` 生成新数组，就要搬运整个数组；硬件上需要的只是一次写入目标窗口的 DMA。

做法是把大数组作为 JAX 的 Ref 传给 kernel。`jax.new_ref(array)` 创建一个可修改的数组，kernel 收到它在 HBM 中的 Ref，写入它就是修改它，不需要输出：

```python
@pl.kernel(
    out_type=(),
    mesh=tc_mesh,
    scratch_types=(pltpu.SMEM((1,), jnp.int32), pltpu.SemaphoreType.DMA),
    ...
)
def kernel(rows_hbm: Ref, position_hbm: Ref, buffer_hbm: Ref, position_smem: Ref, sem: Ref) -> None:
    pltpu.async_copy(position_hbm, position_smem, sem).wait()
    # 写入的起始行在运行时才知道；数组只有一列 tile，任意行窗口都可以由一次 DMA 描述。
    pltpu.async_copy(rows_hbm, buffer_hbm.at[pl.ds(position_smem[0], NEW)], sem).wait()

buffer = jax.new_ref(jnp.zeros((ROWS, 128), jnp.float32))
compiled(buffer, jnp.full((NEW, 128), step + 1.0), jnp.array([start], jnp.int32))
```

与前两种写法相比，改了三处：`out_type=()`，kernel 没有输出；大数组 `buffer` 是 Ref，作为最后一个参数传入；写入位置从 SMEM 读出（第 7 节），作为 DMA 目标窗口的起点。`f32[64,128]` 只有一列 tile，运行时的行起点不需要对齐（第 3、7 节）。

连续在第 5、8、11 行追加三次，每次 3 行：

```text
连续追加 3 次（起始行 5、8、11）后数值检查：True；前 16 行的第 0 列：[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 0.0, 0.0]
```

编译后的 HLO 说明了为什么没有搬运整个数组：

```text
input_output_alias：{ {}: (0, {}, may-alias) }
ROOT %append.1 = f32[64,128]{1,0:T(8,128)} custom-call(%args_0_.1, %copy.4.args_1_, %args_2_.1)
```

程序的结果与第 0 个参数（`buffer`）共用同一块内存（`input_output_alias`），kernel 的结果就是写回后的 `buffer`，没有任何 copy。清单中写入的只有一条 HBM → HBM 的 `dma.general`，长度 3 个 granule，目标地址在寄存器中：

```tpuasm
{ s0: dma.general [hbm:s20], [hbm:s0], length=3, ... }
```

每次追加的开销只取决于新数据的大小，与数组的大小无关。

## 对照：原生 XLA

本小节实验[源码](02_jax_concatenate.py)、[输出](02_jax_concatenate.txt)。

XLA 把拼接改写成 `pad + maximum`：先把两个输入分别填充到输出的大小，再逐元素取最大值；四组拼接的 fusion 都叫 `pad_maximum_fusion`。但编译出来的指令因情况而异：

- 对齐的两组：清单中只有 2 条 `vld`、2 条 `vst` 和 3 次 DMA，与 Pallas 在 TC VMEM 中拼接完全相同。填充和取最大值在编译时就被化简掉了。
- 不对齐的两组：分别有 14 条 `vrot.slane.down` 或一次 XLU 循环移位，与 Pallas 的做法相同；此外还留有一条 `vmax` 和生成、选择填充值的 `vge`、`vsel`（各 4 条 `vsel`，Pallas 是 2 条）。填充值必须不大于任何输入，取最大值才不影响结果。

所以在 TC VMEM 中拼接这条路上，XLA 与 Pallas 相差不多。差别在于 XLA 总要把数据搬进 TC VMEM 再搬出来；手写时，对齐的拼接可以直接交给 DMA，完全不经过向量单元。只有列方向不对齐时，才需要 XLU。
