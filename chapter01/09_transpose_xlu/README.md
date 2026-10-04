# 矩阵转置

上一节的 lane gather 只在一个 TC VREG 内移动数据。转置要把第 s 行第 l 列送到第 l 行第 s 列：行变成列。一个 TC VREG 是 8 行 × 128 列，它的 8 行在转置后变成 8 列，分散到 128 个不同的输出行，也就是 16 个不同的输出 TC VREG。这种跨越多个 TC VREG 的重排，向量单元做不了，TPU v4 用 XLU 完成。本节从一个 `128×128` 的转置出发，先读懂清单中的每一类指令，再依次只改同时转置的个数、dtype、矩阵大小和对齐。

## 写法：`.T` 与多个输出

本小节 Pallas 实验[源码](01_pallas_transpose.py)、[输出](01_pallas_transpose.txt)；实验条件和统计函数在 [transpose_common.py](transpose_common.py) 中。

kernel 中的转置就是值的 `.T`：

```python
o_vmem[i][...] = x_vmem[i][...].T
```

为了同时转置 count 个矩阵，相对于第 1 节的最小 kernel，`out_type` 改为 tuple，`scratch_types` 中放两个 TC VMEM buffer 的列表：

```python
@pl.kernel(
    out_type=tuple(jax.ShapeDtypeStruct(out_shape, dtype) for _ in range(count)),
    mesh=tc_mesh,
    scratch_types=(
        [pltpu.VMEM(shape, dtype) for _ in range(count)],
        [pltpu.VMEM(out_shape, dtype) for _ in range(count)],
        pltpu.SemaphoreType.DMA,
    ),
    ...
)
def kernel(*refs: Ref) -> None:
    x_hbm, o_hbm, (x_vmem, o_vmem, sem) = refs[:count], refs[count:2 * count], refs[2 * count:]
```

`out_type` 是 tuple 时，kernel 的输出 Ref 也有多个，返回值是同样结构的 tuple。`scratch_types` 可以是嵌套的 tuple 和 list，kernel 收到的 scratch Ref 保持同样的嵌套结构。因为 Ref 个数随 count 变化，kernel 用 `*refs` 接收全部参数再按位置拆开。每个条件都与 NumPy 的 `x.T` 逐元素比较。

## 基本形式：一个 128×128 的转置

`i32[128,128]` 有 16 个 TC VREG。清单中的计算部分（省略 DMA）：

```text
{ vld: vld.8x128 v0, [vmem:0x0] ; ... }
{ vld: vld.8x128 v1, [vmem:0x8] ; vx0: vxpose.0.start.8x128 trf0, v0, 128 }
{ vld: vld.8x128 v2, [vmem:0x10] }
...
{ vld: vld.8x128 v9, [vmem:0x48] ; vx0: vxpose.0.8x128 trf0, v1, 128 }
...
{ vx0: vxpose.0.8x128 trf0, v14, 128 }
{ vx0: vxpose.0.end.8x128 trf0, v15, 128 }
{ vr0: vpop.8x128 v16, trf0 }
{ vst: vst.8x128 [vmem:0x80], v16 }
{ vr0: vpop.8x128 v17, trf0 }
{ vst: vst.8x128 [vmem:0x88], v17 }
...
```

| 指令 | 条数 | 作用 |
| --- | ---: | --- |
| `vld.8x128` | 16 | 依次读入输入的 16 个 TC VREG，即第 0–7 行、第 8–15 行……第 120–127 行 |
| `vxpose.0.start.8x128 trf0, v0, 128` | 1 | 把第一个 TC VREG 交给 XLU，开始一次新的转置 |
| `vxpose.0.8x128 trf0, vN, 128` | 14 | 依次交出中间的 14 个 |
| `vxpose.0.end.8x128 trf0, v15, 128` | 1 | 交出最后一个，转置在 XLU 中完成 |
| `vpop.8x128 vN, trf0` | 16 | 从队列 `trf0` 依次取回结果 |
| `vst.8x128` | 16 | 写进输出 buffer |

转置仍是“提交—取回”的形式（与第 6 节的 EUP 相同），但一次提交不对应一次取回：XLU 要收齐一整块 `128×128` 才能产生结果。`.start` 与 `.end` 标出这一块的第一个和最后一个 TC VREG。

每条 `vxpose` 有三个操作数：结果进入的队列 `trf0`、提交的 TC VREG、以及 `128`。下文 `129×129` 的实验中这个数会变成 `8`，那时每块只取回 1 次而不是 16 次：它是这一块转置后的行数，也就是输入的有效列数（向上取到 8 的倍数），决定了要取回几个 TC VREG。

取回的顺序就是输出的行序。第 k 次 `vpop` 得到输出的第 8k 到 8k + 7 行，也就是输入的第 8k 到 8k + 7 列，紧接着写到输出 buffer 的第 k 个 tile（`[vmem:0x80]`、`[vmem:0x88]`……，输出 buffer 从 TC VMEM 地址 `0x80` 开始）。

`vld` 与 `vxpose` 被排进同一个 bundle 时可以同时发射；从清单中可以看到，编译器先把大部分 TC VREG 读进来，再连续提交。XLU 每 8 个周期接收一个 TC VREG、送出一个 TC VREG（第三章第 5 节），所以 16 次提交和 16 次取回各占 16 × 8 = 128 个周期；第三章第 5 节实测一次完整转置的片段为 259 个周期（含计时本身的 12 个）。

## 只改个数：两个 XLU

同时转置 2 个、3 个互不相关的矩阵时，提交和取回按目的队列统计：

| 矩阵个数 | 提交到 `trf0` | 提交到 `trf1` | 从 `trf0` 取回 | 从 `trf1` 取回 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 0 | 16 | 0 |
| 2 | 16 | 16 | 16 | 16 |
| 3 | 32 | 16 | 32 | 16 |

TPU v4 的 TensorCore 有两个 XLU，结果分别进入 `trf0` 和 `trf1`。两个矩阵时，编译器把它们分给两个 XLU；第三个矩阵又回到第一个 XLU。

两个矩阵时提交指令的写法：

```text
7 × vx0: vxpose.0.8x128 trf0, vN, 128
7 × vx0: vxpose.1.8x128 trf1, vN, 128
7 × vx0: vxpose.2.8x128 trf0, vN, 128
7 × vx0: vxpose.3.8x128 trf1, vN, 128
1 × vx0: vxpose.2.start.8x128 trf0, vN, 128
1 × vx0: vxpose.3.start.8x128 trf1, vN, 128
1 × vx0: vxpose.0.end.8x128 trf0, vN, 128
1 × vx0: vxpose.1.end.8x128 trf1, vN, 128
```

助记符中的数字由 tpuasm 根据编码给出：`.0`、`.2` 进入 `trf0`，`.1`、`.3` 进入 `trf1`（[指令索引](../../../tpuasm/docs/references/tpu_v4_tc_isa.md)列出了每种写法对应的队列）。同一次转置中 `.0` 与 `.2` 交替出现，都进入 `trf0`；单独转置一个矩阵时则全部是 `.0`。两者在编码中代表什么，本节没有确定，但它不影响结果的去向。三个矩阵时，部分提交出现在 `vx1` 槽，其中既有进入 `trf1` 的，也有进入 `trf0` 的：指令走哪个槽与结果进入哪个 XLU 无关，XLU 由助记符中的数字决定。

两个 XLU 各有自己的队列，可以同时工作。第三章第 5 节测得，两次转置分给 `trf0` 和 `trf1`、并把两边的 `vpop` 放进同一个 bundle 时，总时间与一次转置相同（259 个周期）；若先取完 `trf0` 再取 `trf1`，要 380 个周期。

## 只改 dtype：bf16 直接以打包格式转置

`bf16[128,128]` 只有 8 个打包的 TC VREG（第 4 节），每个装 16 行。清单：

```text
{ vld: vld.8x128 v1, [vmem:0x8] ; vx0: vxpose.0.packed.start.8x128 trf0, v0, 128 }
...
{ vx0: vxpose.0.packed.end.8x128 trf0, v7, 128 }
{ vr0: vpop.8x128 v8, trf0 }
{ vst: vst.8x128 [vmem:0x40], v8 }
...
```

8 次 `vxpose.0.packed`、8 次 `vpop`、8 次 `vst`，没有任何 unpack 或 pack。`.packed` 告诉 XLU 输入是打包的 16 bit 数据；取回的结果同样是打包格式，每次 `vpop` 得到输出的 16 行。提交和取回的次数都是 f32 的一半，但时间并不减半。第三章第 5 节用同样的手写序列计时：8 次打包提交加 8 次取回共 259 个周期，与 f32 的 16 次提交加 16 次取回完全相同；单看提交，8 次打包提交要 125 个周期，即每次 16 个周期。XLU 每个周期处理一行：f32 的 TC VREG 有 8 行，打包的 bf16 有 16 行，所以转置的时间由矩阵的行数决定，与 dtype 无关。打包省掉的是 unpack、pack 和一半的指令条数，而不是 XLU 的时间。

## 只改大小：256×256 是四块

`f32[256,256]` 被切成 4 个 `128×128` 块：输入块 (i, j) 转置后写到输出块 (j, i)。清单中有 64 次提交、64 次取回，`trf0` 与 `trf1` 各 32 次，即每个 XLU 处理两块；`.start` 和 `.end` 各出现 4 次，对应 4 次独立的转置。矩阵变大不会使 XLU 处理更大的块，只是块数增加；时间随块数线性增长，两个 XLU 并行时减半。

## 只改对齐：129×129 的边界

`f32[129,129]` 的容量是 `17 × 2 = 34` 个 TC VREG（第 4 节）：行方向 17 个 tile（最后一个只有 1 行有效），列方向 2 个 tile（第二个只有 1 列有效）。按 `128×128` 的块切开，有四个区域：

| 区域 | 输入 → 输出 | 提交 | 第三个操作数 | 取回 |
| --- | --- | ---: | ---: | ---: |
| A | `128×128 → 128×128` | 16 | 128 | 16 |
| B | `128×1 → 1×128` | 16 | 8 | 1 |
| C | `1×128 → 128×1` | 1 | 128 | 16 |
| D | `1×1 → 1×1` | 1 | 8 | 1 |

清单中提交的写法与上表对应：

```text
7 × vxpose.2.8x128 trf0, vN, 128 、 7 × vxpose.0.8x128 trf0, vN, 128 、 1 × vxpose.0.start、1 × vxpose.2.end（区域 A）
7 × vxpose.3.8x128 trf1, vN, 8   、 7 × vxpose.1.8x128 trf1, vN, 8   、 1 × vxpose.1.start、1 × vxpose.3.end（区域 B）
1 × vxpose.0.start.end.8x128 trf0, vN, 128（区域 C）
1 × vxpose.1.start.end.8x128 trf1, vN, 8（区域 D）
```

- 区域 B 的输入只有 1 列有效，转置后只有 1 行；第三个操作数是 8，这一块只取回 1 个 TC VREG。若仍按 128 处理，就要多取回 15 个无用的 TC VREG。第三章第 5 节测得，16 次宽度为 8 的提交加 1 次取回共 139 个周期：提交指令仍然每 8 个周期发射一条，省下的是 15 次取回。
- 区域 C 和 D 的输入只有一个 TC VREG，开始与结束是同一次提交，写作 `.start.end`。
- 两个 XLU 分工：区域 A、C 进入 `trf0`（取回 32 次），区域 B、D 进入 `trf1`（取回 2 次）。

结果的写回：34 次取回中，17 次是满的 TC VREG（`vst`），17 次只有部分有效（`vst.msk`）。带掩码的 store 只写有效区，掩码由 `vlaneseq`、`vand`、`vshrl`、`veq` 生成，与第 4 节相同。

`bf16[129,129]` 的边界处理复杂得多：35 次 `vunpackl`、33 次 `vunpacku`、35 次 `vpackc`，另有 9 次 `vsupp` 提交和 9 次打包提交。从同时出现的 unpack 与 pack 看，边界上不满 16 行的打包 tile 被先展开成 32 bit 的 TC VREG，再交给 XLU，取回后重新打包；清单中这部分提交用的是另一种指令 `vsupp`，它的确切语义本节没有确定。非对齐的 bf16 转置代价远高于对齐的情况，设计数据布局时应尽量让转置的两个维度都是 128 的倍数。

## 对照：原生 XLA

本小节实验[源码](02_jax_transpose.py)、[输出](02_jax_transpose.txt)。

XLA 中转置是一条改变 layout 的 `copy`。同样的七个条件中，有两处差别值得注意：

- bf16：XLA 先 unpack 成 f32，按 16 个 b32 TC VREG 提交，再 pack 写回（8 次 `vunpackl`、8 次 `vunpacku`、16 次提交、8 次 `vpackc`）。Pallas 直接以打包格式提交 8 次，XLU 的工作量是 XLA 的一半。
- 多个矩阵：XLA 为每个矩阵生成一条独立的 `copy`，三个矩阵的 48 次提交全部进入 `trf0`，第二个 XLU 闲置。Pallas 在同一个 kernel 中把它们分给两个 XLU。

`f32[256,256]` 和 `f32[129,129]` 的条件下，XLA 与 Pallas 一样把四个区域分给 `trf0` 和 `trf1`，提交和取回的次数相同。
