# 标量单元与控制流

前面各节的 kernel 都是直线代码：每条指令执行一次，所有地址和次数都是编译期常数。实际的 kernel 还需要运行时才知道的数：循环次数、窗口位置、是否执行某一步。这些数由 TensorCore 的标量单元处理：它有自己的寄存器（`s0`–`s31`）、自己的内存 SMEM、自己的发射槽 `s0`/`s1`，并负责分支。本节依次看标量怎样进入 kernel、循环和条件怎样写、在清单中是什么样子，以及循环和标量指令各要多少周期。

## 标量从 HBM 进入 SMEM

本小节实验[源码](01_pallas_smem_scalar.py)、[输出](01_pallas_smem_scalar.txt)。

相对于第 1 节的最小 kernel，多一个输入 `s = f32[2]`，`scratch_types` 中多一个 SMEM buffer，计算改为 `x * s[0] + s[1]`：

```python
scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.SMEM(s.shape, s.dtype), pltpu.SemaphoreType.DMA),
...
def kernel(x_hbm: Ref, s_hbm: Ref, o_hbm: Ref, x_vmem: Ref, s_smem: Ref, sem: Ref) -> None:
    pltpu.async_copy(x_hbm, x_vmem, sem).wait()
    pltpu.async_copy(s_hbm, s_smem, sem).wait()
    scale = s_smem[0]
    shift = s_smem[1]
    x_vmem[...] = x_vmem[...] * scale + shift
    pltpu.async_copy(x_vmem, o_hbm, sem).wait()
```

`pltpu.SMEM(shape, dtype)` 在标量内存 SMEM 中分配 buffer。标量参数和向量数据一样在 HBM 中，同样要由 DMA 搬运，只是目的地换成 SMEM。读 SMEM Ref 的单个元素（`s_smem[0]`）得到一个标量值。

清单中的对应部分：

```tpuasm
{ s0: dma.simple [smem:s13], [hbm:s1], length=1, dst_flag=[sflag:52] }
{ misc: vwait.ge [sflag:52], 1 }
{ misc: vsyncadd.s32 [sflag:52], -1 }
{ s0: sfence }
{ s1: sld s16, [smem:0x3e] ; vld: vld.8x128 v0, [vmem:0x0] }
{ s1: sld s17, [smem:0x3f] ; vld: vld.8x128 v1, [vmem:0x8] }
{ va0: vmov.8x128 v2, s16 }
{ va0: vmul.8x128.f32 v3, v2, v0 ; va1: vmov.8x128 v4, s17 }
...
{ va1: vadd.8x128.f32 v6, v4, v3 }
```

- `sld` 从 SMEM 读一个 32 bit 数到标量寄存器，只能在 `s1` 槽。`s_smem[0]` 和 `s_smem[1]` 的地址是 `0x3e` 和 `0x3f`：SMEM 的地址以 32 bit 为单位。
- 等待 SMEM 的 DMA 之后，编译器插入了一条 `sfence`，然后才 `sld`。等待向量数据时没有这条指令。

这正是第 3 节的两级发射：`vwait` 在向量一侧等待，只挡住排在它后面的向量指令；`sld` 是标量指令，在标量发射时执行，可以越过还没有等到的 `vwait`。`sfence` 让后面的 bundle 等到此前的 bundle（包括 `vwait`）都已向量发射，保证 `sld` 读到的是 DMA 写入后的值。读 TC VMEM 的 `vld` 本身就在向量一侧，排在 `vwait` 后面，不需要 `sfence`。

> 暂且可以理解为：去掉这条 `sfence`，`sld` 读到的是上一次运行留在 SMEM 中的旧值。第三章第 3 节用 tpuasm 把它换成 `vnop`，演示了这个后果。

- 标量参与向量运算前，要先用 `vmov` 把它广播成一个 TC VREG（`vmov.8x128 v2, s16`），再做向量乘法。与之对照，常数 2.0 可以直接写成 `vmul` 的立即数操作数，不需要广播。

SMEM 的容量是 1 MiB，即 262144 个 32 bit 字，其中最高的一小段由 runtime 使用（[源码](05_pallas_smem_capacity.py)、[输出](05_pallas_smem_capacity.txt)）：

```text
pltpu.SMEM((131072,), f32)，512 KiB：数值检查通过
pltpu.SMEM((262144,), f32)，1024 KiB：编译失败：... Used 1.00M of 1.00M smem. Exceeded smem capacity by 136B.
```

前面清单中读 TensorCore 编号的 `[smem:0x1]`，就是 runtime 放在 SMEM 最低处的值。

## 循环：pl.loop

本小节实验[源码](02_pallas_loops.py)、[输出](02_pallas_loops.txt)。

`f32[64,128]` 有 8 个 tile，逐个乘以 2。实验比较四种写法。

第一种是 Python 的 `for`：

```python
for i in range(8):
    x_vmem[pl.ds(i * 8, 8)] = x_vmem[pl.ds(i * 8, 8)] * 2.0
```

Python 循环在 tracing 时就执行完了，Mosaic 看到的是 8 段起点为常数的直线代码。

第二种是 `pl.loop`：

```python
@pl.loop(0, 8)
def _(i: jax.Array) -> None:
    start = i * 8
    x_vmem[pl.ds(start, 8)] = x_vmem[pl.ds(start, 8)] * 2.0
```

`pl.loop(lower, upper)` 是一个装饰器：被装饰的函数是循环体，参数 `i` 是运行时的循环变量，依次取 `lower` 到 `upper - 1`，函数定义完立即执行整个循环。`i` 不是 Python 整数，`i * 8` 是运行时的标量值，所以切片要用 `pl.ds(start, 8)`，不能写 `x_vmem[start:start + 8]`。

第三种是 `pl.loop(0, 8, unroll=4)`：循环体复制 4 份，每次迭代处理 4 个 tile，只循环 2 次。

第四种的循环次数来自输入：先把 `n = int32[2]` 搬进 SMEM，再 `pl.loop(0, n_smem[0])`。

| 写法 | kernel 段 bundle 数 | `vld`/`vmul`/`vst` 条数 |
| --- | ---: | ---: |
| Python `for` | 38 | 8 / 8 / 8 |
| `pl.loop(0, 8)` | 35 | 1 / 1 / 1 |
| `pl.loop(0, 8, unroll=4)` | 37 | 4 / 4 / 4 |
| `pl.loop(0, n_smem[0])` | 43 | 1 / 1 / 1 |

静态边界的 `pl.loop` 在清单中是一个真正的循环：

```tpuasm
L_0200:
{ s0: sshll.u32 s11, s10, 0x3 ;  s1: sadd.s32 s10, 1, s10 }   # start = i * 8；i += 1
{ s0: sadd.s32 s12, 0, s11 ;     s1: sge.s32 p1, s10, 8 }     # 地址；p1 = (i >= 8)
{ vld: vld.8x128 v0, [vmem:s12] }
{ va0: vmul.8x128.f32 v1, 2.0, v0 }
{ vst: vst.8x128 [vmem:s12], v1 }
{ s0: @p1 dma.simple [hbm:s1], [vmem:s7], length=64, dst_flag=[sflag:52] }
{ s0: @!p1 sbr.rel L_0200 }                                    # 未结束则跳回
{}
```

- 循环变量、地址计算和退出判断都在标量槽中完成，`vld`/`vst` 的地址改用标量寄存器 `[vmem:s12]`。
- `sbr.rel` 是相对跳转，谓词 `@!p1` 使它只在循环未结束时跳转。
- 跳转指令后面的一个 bundle 无论是否跳转都会执行，称为延迟槽。TPU v4 的分支延迟是 1 个 bundle；这里编译器没有找到可放的指令，填了一个空 bundle `{}`。
- 循环之后的输出 DMA 被编译器提前放进了循环体，用谓词 `@p1` 限定只在最后一次迭代执行。

Python `for` 与 `pl.loop` 的差别不只在代码长度。展开后的 8 个 tile 互不依赖，编译器把前一个 tile 的 store、当前 tile 的乘法和后一个 tile 的 load 放进同一个 bundle，8 个 tile 的计算部分只占约 11 个 bundle。`pl.loop` 的循环体每次只处理一个 tile，`vld`、`vmul`、`vst` 各占一个 bundle，每次迭代都要重新走一遍地址计算、判断和跳转。`unroll=4` 在两者之间：一次迭代内的 4 个 tile 可以交错，循环开销也减为 2 次。

### 周期数：循环体的 bundle 数乘以迭代次数

本小节实验[源码](07_pallas_loop_cycles.py)、[输出](07_pallas_loop_cycles.txt)。

清单中的 bundle 数是静态的代码长度，循环实际要多少周期，要放进第 3 节的 TC VMEM 边界去测。实验把输入加大到 `f32[512,128]`（64 个 tile），kernel 的操作数和结果都在 TC VMEM，只改循环的写法：

```python
if style == 'for':
    for i in range(TILES):
        o_vmem[pl.ds(i * 8, 8)] = x_vmem[pl.ds(i * 8, 8)] * 2.0
else:
    @pl.loop(0, TILES, unroll=unroll)
    def _(i: jax.Array) -> None:
        start = i * 8
        o_vmem[pl.ds(start, 8)] = x_vmem[pl.ds(start, 8)] * 2.0
```

| 写法 | 区间的 bundle 数 | 循环体的 bundle 数 | 迭代次数 | 周期数 | 每个 tile |
| --- | ---: | ---: | ---: | ---: | ---: |
| Python `for`（展开） | 90 | — | — | 106 | 1 |
| `pl.loop(0, 64)` | 30 | 6 | 64 | 424 | 6 |
| `pl.loop(0, 64, unroll=2)` | 31 | 7 | 32 | 264 | 3.5 |
| `pl.loop(0, 64, unroll=4)` | 33 | 9 | 16 | 184 | 2.25 |
| `pl.loop(0, 64, unroll=8)` | 37 | 13 | 8 | 144 | 1.6 |

四种 `pl.loop` 的周期数都恰好是“40 + 循环体的 bundle 数 × 迭代次数”：424 = 40 + 6 × 64，264 = 40 + 7 × 32，184 = 40 + 9 × 16，144 = 40 + 13 × 8。其中 40 是外壳固定的 39 个周期加循环之前的一个 bundle。循环体的 bundle 数包含跳转之后的延迟槽；跳转本身不多占周期。`pl.loop(0, 64)` 的循环体是：

```tpuasm
L_0210:
{ s0: sshll.u32 s7, s6, 0x3 ; s1: sadd.s32 s6, 1, s6 }         # start = i * 8；i += 1
{ s0: sadd.s32 s10, s7, s0 ; s1: sadd.s32 s13, s7, s1 }        # 输入、输出的地址
{ s0: sge.s32 p0, s6, 64 ; vld: vld.8x128 v0, [vmem:s10] }
{ va0: vmul.8x128.f32 v1, 2.0, v0 }
{ s0: @!p0 sbr.rel L_0210 }
{ vst: vst.8x128 [vmem:s13], v1 }                              # 延迟槽
```

这一次编译器把 `vst` 放进了延迟槽，6 个 bundle 中没有空的。但每个 tile 仍然要 6 个周期，是展开版本的 6 倍：`vld`、`vmul`、`vst` 三条指令之间有依赖（乘法的结果 2 个周期后才可用），一次迭代内没有别的 tile 的指令可以填进这些 bundle。`unroll=2` 每次迭代多处理一个 tile，循环体只多 1 个 bundle；`unroll=8` 时每个 tile 1.6 个周期，接近展开版本。

所以在 TPU 上，循环展开是性能的主要手段之一：它让独立的工作出现在同一段直线代码中，供编译器填满各个发射槽。循环的周期数可以直接从清单算出：**循环体的 bundle 数（含延迟槽）乘以迭代次数，再加上循环体内的等待。** 代价是代码变长；TensorCore 的指令存储有限，循环次数大时不能全部展开，`unroll` 取 4 到 8 通常已经拿到大部分好处。

运行时边界的 `pl.loop` 多了一段循环前的检查：

```tpuasm
{ s1: sld s17, [smem:0x3f] }
{ s0: sle.s32 p1, s17, 0 }
{ s0: @p1 sbr.rel L_023a }      # n <= 0 时跳过整个循环
```

循环体本身与静态边界相同，退出判断改为与寄存器 `s17` 比较。

## 条件与运行时窗口：pl.when 与 pl.ds

本小节实验[源码](03_pallas_dynamic_window_when.py)、[输出](03_pallas_dynamic_window_when.txt)。

kernel 输入 `p = int32[2]`：从 `f32[64,128]` 中取第 `p[0]` 个行 tile，`p[1] > 0` 时才乘以 2。相对于第 1 节的最小 kernel，输出改为 `f32[8,128]`，主体改为：

```python
pltpu.async_copy(p_hbm, p_smem, sem).wait()
start = p_smem[0] * 8
pltpu.async_copy(x_hbm.at[pl.ds(start, 8)], x_vmem, sem).wait()

@pl.when(p_smem[1] > 0)
def _() -> None:
    x_vmem[...] = x_vmem[...] * 2.0

pltpu.async_copy(x_vmem, o_hbm, sem).wait()
```

`pl.when(条件)` 也是装饰器：条件为真时执行被装饰的函数，函数没有返回值，只通过写 Ref 产生效果。同一个编译结果用 `p = [3, 1]`、`[5, 0]`、`[7, 1]` 三组参数调用，结果都正确：窗口位置和是否计算在运行时决定，不需要重新编译。

清单中，运行时的窗口起点就是一次标量乘法和一次加法：

```tpuasm
{ s1: sld s13, [smem:0x3e] }
{ s0: sshll.u32 s14, s13, 0x3 }           # p[0] * 8
{ s0: sadd.s32 s17, s14, s0 }             # HBM 基址 + 偏移
{ s0: dma.simple [vmem:s18], [hbm:s17], length=8, dst_flag=[sflag:52] }
```

这与第 4 节的常数窗口相同，只是偏移从立即数变成了寄存器。`pl.when` 是一次条件跳转：

```tpuasm
{ s0: sle.s32 p1, s19, 0 }
{ s0: @p1 sbr.rel L_020e ;  vld: @!p1 vld.8x128 v0, [vmem:0x0] }
{ va0: @!p1 vmul.8x128.f32 v1, 2.0, v0 }     # 延迟槽
{}
{ vst: vst.8x128 [vmem:0x0], v1 }
L_020e:
```

编译器把被跳过部分的 `vld` 与跳转放在同一个 bundle，把 `vmul` 放进延迟槽，都加上相反的谓词 `@!p1`：条件不成立时，这两条指令照样经过，但不产生效果。这样被跳过的代码也利用了跳转前后的空闲槽位。

## 运行时的起点与 pl.multiple_of

本小节实验[源码](04_pallas_runtime_unaligned_start.py)、[输出](04_pallas_runtime_unaligned_start.txt)。

窗口起点来自运行时的标量时，Mosaic 在编译期不知道它的值。实验从 `f32[64,128]` 的第 `p[0]` 行起取 8 行，用两种方式实现：

```python
# 一：DMA 窗口的起点来自 SMEM
pltpu.async_copy(x_hbm.at[pl.ds(p_smem[0], 8)], o_vmem, sem).wait()
# 二：整个数组先进 TC VMEM，再从第 p[0] 行起 load 8 行
o_vmem[...] = x_vmem[pl.ds(p_smem[0], 8)]
```

同一份编译结果分别用起点 8、3、13、56 调用，结果都等于 `x[start:start+8]`。清单中两者各只有一条指令：

```tpuasm
{ s0: dma.simple [vmem:s17], [hbm:s16], length=8, dst_flag=[sflag:52] }   # 一
{ vld: vld.8x128 v0, [vmem:s17] }                                         # 二
```

第 4 节说过，一列 tile 的数组中相邻的行就是相邻的 granule；TC VMEM 中也一样，`vld.8x128` 从任意行地址读连续 8 行，起点为 3 时跨越两个 tile，仍是一条指令。

数组改为 `f32[64,256]`（两列 tile）后，同样的 `x_vmem[pl.ds(p_smem[0], 8)]` 编译失败：

```text
E2003: CompileTimeMosaicUnprovenMemoryAccessAlignment: cannot statically prove that index in dimension 0 is a multiple of 8
```

两列 tile 时，第 3–10 行在 TC VMEM 中不连续，一条 `vld` 读不出来，所以 Mosaic 要求能证明起点是 8 的倍数。如果程序员知道这个起点一定对齐，可以用 `pl.multiple_of` 告诉编译器：

```python
start = pl.multiple_of(p_smem[0], 8)
o_vmem[...] = x_vmem[pl.ds(start, 8)]
```

编译通过，起点 8、16、56 的结果都正确。`pl.multiple_of(value, n)` 不产生任何指令，只是向编译器保证 `value` 是 n 的倍数；这个保证由程序员负责，起点实际不对齐时，结果是错的，编译器和硬件都不会报错。前面循环中的 `start = i * 8`，编译器自己就能推出它是 8 的倍数，所以不需要写。

## 周期数：标量单元中会等待的指令

本小节实验[源码](06_tpuasm_scalar_timing.py)、[输出](06_tpuasm_scalar_timing.txt)。

第 3 节测过，相互依赖的标量加法每个周期一条，不让发射停下。标量单元还有乘法和读写 SMEM 的指令。实验用 `LccProbe`，片段的形式与第 3 节相同（R0、一段序列、R1、`sfence`、R2），只换中间的序列。标量指令在标量发射时执行，它的等待直接反映在 R1 上：

| 序列 | R1 − R0 | 结论 |
| --- | --- | --- |
| N 条相互依赖的 `smul.u32 s24, 3, s24`，N = 1、4、16 | 2、5、17 | 标量乘法也是下一周期可用 |
| N 条连续的 `sld s24, [smem:0x0]`，N = 1、2、4、16 | 2、6、14、62 | 每条 `sld` 占 4 个周期 |
| N 组“`sld` + 3 条无关的 `sadd`”，N = 2、4 | 9、17 | 中间的 3 条填满了间隔，不再等待 |
| `sld` 后隔 d − 1 个空 bundle 使用结果，d = 1–5 | 6、6、6、6、7 | 结果 4 个周期后才可用 |
| N 条连续的 `sst [smem:0x7f0], s24`，N = 1、4、16 | 2、5、17 | 每周期一条 |
| `sst` → 读同一地址的 `sld` → 使用 | 7 | `sld` 不必等前面的 `sst` |

最后一组检查结果的值：先把 7 存进 SMEM、把 `s24` 清零，再 `sld s24`，紧接着用 `vmov` 或 `sadd` 使用它：

```text
vmov v11, s24，d = 1、2、3：[7, 7, 7]
sadd s23 = s24 + 100，再 vmov，d = 1、2、3：[107, 107, 107]
```

无论隔几个 bundle，读到的都是新值：硬件会等待，程序员不必自己插入空 bundle。

`sld` 有两个限制：连续两条 `sld` 至少相隔 4 个周期，使用 `sld` 结果的指令最早在它之后 4 个周期发射。两种情况下，标量发射都会停下来等待，后面所有的 bundle 一起推迟。但间隔中可以放别的标量指令：“`sld` + 3 条无关的 `sadd`”每组恰好 4 个 bundle，读数与没有任何等待时相同。`sst` 则每周期一条，紧随其后的 `sld` 读同一地址也只按 `sld` 自己的规则等待。

这解释了两处现象。本节第一个清单中，编译器把两个标量参数的 `sld` 放在相邻两个 bundle，第二条要等 4 个周期；标量参数很多时，按每个 4 个周期估算读入的时间，或者把它们的读取与其他标量工作交错。第 3 节的计时区间里，每条 HLO 指令之前 runtime 的 4 个 bundle 占 10 个周期，也是因为其中有两条相邻的 `sld`，后一条的结果又马上被用到。

## 小结：标量单元能做什么

标量槽 `s0`、`s1` 负责地址计算、循环变量、比较、分支，以及发起 DMA（只在 `s0`）和读写 SMEM（只在 `s1`）。查 [tpuasm 的指令索引](../../../tpuasm/docs/references/tpu_v4_tc_isa.md) 可见，标量单元还有 f32 的加、乘、比较和整数与浮点的相互转换。

需要注意的是，标量与向量之间的数据通路并不对称。标量进入向量单元很便宜，一条 `vmov` 广播即可；而向量结果要回到标量单元，例如根据向量的比较结果决定是否跳转，就要经过额外的通路，每次等待 43 个周期。第 12 节的归约会遇到这种从向量到标量的方向。
