# 矩阵转置与 XLU

上一节的 lane gather 只在一个 TC VREG 内移动数据。转置要把第 s 行第 l 列送到第 l 行第 s 列：行变成列，一个 TC VREG 的 8 行要分散到 8 个不同的 lane，必须跨越多个 TC VREG。TPU v4 用 XLU 完成这件事。本节从一个 `128×128` 的转置出发，依次只改同时转置的个数、dtype、矩阵大小和对齐。

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

`out_type` 是 tuple 时，kernel 的输出 Ref 也有多个，返回值是同样结构的 tuple。`scratch_types` 可以是嵌套的 tuple 和 list，kernel 收到的 scratch Ref 保持同样的嵌套结构。因为 Ref 个数随 count 变化，kernel 用 `*refs` 接收全部参数再按位置拆开。

## 基本形式：一个 128×128 的转置

`i32[128,128]` 有 16 个 TC VREG，清单中的指令：

| 指令 | 条数 |
| --- | ---: |
| `vld.8x128` | 16 |
| `vxpose.0.start.8x128 trf0, ...` | 1 |
| `vxpose.0.8x128 trf0, ...` | 14 |
| `vxpose.0.end.8x128 trf0, ...` | 1 |
| `vpop.8x128 ..., trf0` | 16 |
| `vst.8x128` | 16 |

转置仍是“提交—取回”的形式。16 个 TC VREG 依次用 `vxpose` 提交给 XLU：第一次带 `.start`，最后一次带 `.end`，中间 14 次不带后缀。XLU 收齐 `128×128` 的一块之后，把转置结果放进队列 `trf0`，由 16 次 `vpop` 依次取回，每次得到输出的 8 行。

XLU 的基本单位就是 `128×128`：输入恰好是 16 个 TC VREG，输出也是 16 个。提交和取回可以与 load/store 交错进行；从清单中可以看到，下一个 TC VREG 的 `vld` 与当前的 `vxpose` 被排进同一个 bundle。

## 只改个数：两个 XLU

同时转置 2 个、3 个互不相关的矩阵时，提交和取回按目的队列统计：

| 矩阵个数 | 提交到 `trf0` | 提交到 `trf1` | 从 `trf0` 取回 | 从 `trf1` 取回 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 0 | 16 | 0 |
| 2 | 16 | 16 | 16 | 16 |
| 3 | 32 | 16 | 32 | 16 |

TPU v4 的 TensorCore 有两个 XLU，结果分别进入 `trf0` 和 `trf1`。两个矩阵时，编译器把它们分给两个 XLU；第三个矩阵又回到第一个 XLU。助记符中的数字（`vxpose.0`、`.2` 进入 `trf0`，`.1`、`.3` 进入 `trf1`）由 tpuasm 根据编码给出，[指令索引](../../../tpuasm/docs/references/tpu_v4_tc_isa.md)列出了每种写法对应的队列。

> 暂且可以理解为：两个 XLU 可以同时工作，分给两个 XLU 的两次转置所需时间接近一次。第三章第 6 节说明如何从清单估算这类单元的占用时间，并用真机计数器验证。

## 只改 dtype：bf16 直接以打包格式转置

`bf16[128,128]` 只有 8 个打包的 TC VREG（第 4 节），清单中是 8 次 `vxpose.0.packed`、8 次 `vpop`、8 次 `vst`，没有任何 unpack 或 pack。XLU 能直接转置 16 bit 的打包数据，提交和取回的次数都减半。

## 只改大小：256×256 是四块

`f32[256,256]` 被切成 4 个 `128×128` 块：输入块 (i, j) 转置后写到输出块 (j, i)。清单中有 64 次提交、64 次取回，两个 XLU 各 32 次。矩阵变大不会使 XLU 处理更大的块，只是块数增加。

## 只改对齐：129×129 的边界

`f32[129,129]` 的容量是 `17 × 2 = 34` 个 TC VREG（第 4 节）。按 `128×128` 的块切开，有四个区域：

| 区域 | 输入 → 输出 |
| --- | --- |
| A | `128×128 → 128×128` |
| B | `128×1 → 1×128` |
| C | `1×128 → 128×1` |
| D | `1×1 → 1×1` |

清单中有 34 次 `vld`、34 次提交、34 次取回、17 次 `vst` 和 17 次 `vst.msk`。提交带 `.start.end` 的两次，是区域 C 和 D：输入只有一个 TC VREG，开始和结束是同一次提交。17 次带掩码的 store 只写有效区，掩码由 `vlaneseq`、`vand`、`vshrl`、`veq` 生成，与第 4 节相同。

`bf16[129,129]` 的边界处理复杂得多：35 次 `vunpackl`、33 次 `vunpacku`、35 次 `vpackc`，另有 9 次 `vsupp` 提交和 9 次打包提交。边界上不满 16 行的打包 tile 要先展开再重新打包。非对齐的 bf16 转置代价远高于对齐的情况，设计数据布局时应尽量避免。

## 对照：原生 XLA

本小节实验[源码](02_jax_transpose.py)、[输出](02_jax_transpose.txt)。

XLA 中转置是一条改变 layout 的 `copy`。同样的七个条件中，有两处差别值得注意：

- bf16：XLA 先 unpack 成 f32，按 16 个 b32 TC VREG 提交，再 pack 写回（8 次 `vunpackl`、8 次 `vunpacku`、16 次提交、8 次 `vpackc`）。Pallas 直接以打包格式提交 8 次。
- 多个矩阵：XLA 为每个矩阵生成一条独立的 `copy`，三个矩阵的 48 次提交全部进入 `trf0`。Pallas 在同一个 kernel 中把它们分给两个 XLU。

`f32[256,256]` 和两个 `129×129` 的条件下，XLA 清单中的 `vld` 只有 Pallas 的一半左右（32、17、9 次）。XLA 把这些转置分给了两个 TensorCore，清单中的计数是每个 TensorCore 各执行一份，与 Pallas 单个 TensorCore 的计数不能直接比较。
