# 拼接

拼接不做任何算术，只决定数据放在哪里。所以它最能说明一个道理：数据的位置由谁来安排，向量单元、XLU 还是 DMA，代价差别很大。本节比较两种写法：在 TC VMEM 中对值做 `jnp.concatenate`，以及让 DMA 把两个输入直接写进输出的两个窗口。再只改 shape，看不对齐时会发生什么。

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

- TC VMEM 中拼接：b 的第 0 行要写到输出的第 9 行，即第二个 tile 的第 1 个 sublane，所以 b 的每个 tile 都要向下移动一个 sublane。清单中有 14 条 `vrot.slane.down` 和 2 条 `vsel`。
- DMA 写入窗口：两次 DMA 的长度都是 9 个 granule，第二次从第 9 行写起。数组只有一列 tile（第 3 节），任意行窗口都连续，DMA 不需要额外工作。

## 只改 shape：不对齐的列

两个 `f32[8,129]` 沿 axis=1 拼接成 `f32[8,258]`：

- TC VMEM 中拼接：b 的第 0 列要放到输出的第 129 列，跨越 lane 边界，需要一次 XLU 循环移位（`vrot` 加 `vpop`，第 8 节）加掩码选择和带掩码的 store。
- DMA 写入窗口：编译失败，`Slice sizes along tiled dimensions must be aligned to tiles`。

列方向上，一个 granule 是一个 sublane 的 128 个 lane。DMA 的地址和长度都以 granule 为单位（第 3 节），表达不了从某个 granule 中间开始的窗口，所以从第 129 列开始的窗口不能由 DMA 描述。这与行方向不同：行方向的最小单位是一个 granule（一行），列方向的最小单位是 128 列。所以不对齐的列只能交给 XLU 在 TC VREG 内移动。

## 对照：原生 XLA

本小节实验[源码](02_jax_concatenate.py)、[输出](02_jax_concatenate.txt)。

XLA 对四组拼接都生成一个名为 `pad_maximum_fusion` 的 fusion：先把两个输入分别填充到输出的大小，再逐元素取最大值；为了不影响结果，填充值必须不大于任何输入。对齐的情况下，清单中除了 load/store，还有 `vlt`、`vsel` 生成和选择填充值，以及一条 `vmax`；不对齐的情况分别多出 14 条 `vrot.slane.down` 或一次 XLU 循环移位，与 Pallas 在 TC VMEM 中拼接的做法相同，再加上填充和取最大值的指令。

`pad + maximum` 是一种统一的改写，对任何拼接都正确，但即使在最简单的对齐情况下也多出了比较、选择和取最大值。手写时，对齐的拼接应当直接交给 DMA；只有列方向不对齐时，才需要 XLU。
