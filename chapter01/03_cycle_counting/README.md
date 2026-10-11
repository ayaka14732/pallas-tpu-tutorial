# 周期数：用 LCC 计时

上一节的清单回答了“TensorCore 执行了哪些指令”，这一节回答“这些指令要多少周期”。TensorCore 上的一切都按时钟周期进行：一个 bundle 的发射、一个结果要等多久、一次 DMA 的开销，原本都是周期计数。TPU v4 的时钟是 1050 MHz，一个周期 0.952 ns，100 个周期约 95 ns。本教程从这一节起，对每一种操作都同时给出它的指令和它的周期数。

周期数不用主机的时钟测。主机上调用一次 kernel 并等到结果，要一百多微秒，也就是十几万个周期，其中绝大部分是提交、启动和完成通知（第三章第 1 节）；而本章的一段计算往往只有几十到几百个周期。周期数直接在设备上读：TensorCore 有一个本地周期计数器 LCC（local cycle counter），每个周期加 1，标量单元可以用一条指令读出它的值。在一段代码的两端各读一次，相减就是这段代码的周期数，没有换算，也不依赖 profiler。

本节先写出读 LCC 的指令，用它测几段手写的 bundle，得到“bundle 数不是周期数”的第一个例子；再给第 1 节的最小 kernel 计时，看它的时间花在哪里；然后把计时的边界放在 TC VMEM，得到本章之后各节都要用的测法；最后归纳周期数由哪几类开销组成。

本节实验：N 个 bundle [源码](01_tpuasm_lcc_intervals.py)、[输出](01_tpuasm_lcc_intervals.txt)；相互依赖的向量指令 [源码](02_tpuasm_dependent_chain.py)、[输出](02_tpuasm_dependent_chain.txt)；最小 kernel 的周期数 [源码](03_pallas_first_kernel_cycles.py)、[输出](03_pallas_first_kernel_cycles.txt)；TC VMEM 边界 [源码](04_jax_pallas_vmem_interval.py)、[输出](04_jax_pallas_vmem_interval.txt)。

## 读 LCC 的指令

LCC 是 64 位的，标量寄存器是 32 位的，所以要分两半读：

```tpuasm
{ s0: srdreg.lcclo s20 ; s1: srdreg.lcchi s25 }
```

`srdreg` 读一个特殊寄存器（special register），`lcclo` 和 `lcchi` 是 LCC 的低、高 32 位，结果写进标量寄存器。两条指令放在同一个 bundle 的两个标量槽 `s0`、`s1` 中，在同一个周期执行，得到的两半属于同一个时刻，拼起来就是一致的 64 位读数。如果分在两个 bundle 读，两次读之间低位可能恰好从 `0xFFFFFFFF` 进位到 0，拼出的值会差 2³²。

Pallas 的公开接口没有读 LCC 的函数，所以读数用 tpuasm 直接插进机器程序。本教程把插入读数的步骤封装成 [`tpuasm_tools.py`](../../tpuasm_tools.py) 中的两个工具：`LccProbe` 测手写的片段，`KernelClock` 测编译出的程序。下面依次介绍。

## 手写片段：LccProbe

`LccProbe` 把“编译载体、插入片段、取回读数”包装成一个类。它先用 Pallas 编译一个普通的 kernel 作为载体：

```python
def kernel(x_hbm, out_hbm, data, sem) -> None:
    pltpu.async_copy(x_hbm, data, sem).wait()
    # 带唯一立即数的 vxor 标出插入位置。
    data[:8, :] = data[:8, :] ^ jnp.uint32(0x13579BDF)
    core = jax.lax.axis_index('tc')
    pltpu.async_copy(data.at[:72, :], out_hbm.at[pl.ds(core * 72, 72)], sem).wait()
```

输入是 `u32[256,128]` 的随机数，先整块搬进 TC VMEM 的 `data`；然后对前 8 行做一次异或，常数 `0x13579BDF` 在整个程序中只出现一次，`find_bundles` 由它找到这条 `vxor` 所在的 bundle，作为插入位置；最后把 `data` 的前 72 行（9 个 tile）写回 HBM。

`run(body)` 在插入位置之前放入三段清单：

1. **前缀。** 保存载体此后还要用到的标量寄存器，把输入的第 0 个 tile 读进 `v10` 供片段当作数据使用，执行调用者给的 `setup`，最后用一条 `sfence` 让此前的工作全部发射完（`sfence` 的含义见下文）。
2. **片段 `body`。** 由调用者写出，其中用 `read_lcc(20)`、`read_lcc(21)`、`read_lcc(22)` 三次读 LCC，记作 R0、R1、R2。
3. **后缀。** 把读数用 `vmov` 广播成 TC VREG，用 `vst` 存进 TC VMEM 的第 1–6 个 tile，随载体的输出 DMA 返回主机；再恢复标量寄存器。

主机把每对半字拼成 64 位，返回每次运行的 `(R1 − R0, R2 − R0)`。插入后的程序不再经过编译器，bundle 的内容和顺序就是设备实际执行的内容。片段用两个小函数写出：`bundle(text)` 生成一个 bundle 的清单文本，`text` 为空时是空 bundle；`read_lcc(low)` 生成上面那个读 LCC 的 bundle，低位写进 `s{low}`、高位写进 `s{low + 5}`。

## 实验：N 个 bundle 要几个周期

片段是“R0、N 个相同的 bundle、R1、一条 `sfence`、R2”：

```python
body = read_lcc(20) + bundle(instruction) * count + read_lcc(21) + bundle('s0: sfence') + read_lcc(22)
```

实验只改变两处：`instruction` 取四种 bundle 之一，`count` 取 0、1、4、16、64。

| 中间的 bundle | `instruction` |
| --- | --- |
| 空 bundle | `''` |
| 相互依赖的标量加法 | `'s0: sadd.s32 s24, 1, s24'` |
| 独立的向量加法 | `'va0: vadd.8x128.s32 v11, 1, v10'` |
| 从 TC VMEM 读一个 TC VREG | `'vld: vld.8x128 v11, [vmem:0x0]'` |

四种 bundle 的结果完全相同，每种配置 8 次运行的读数也完全相同：

| N | 0 | 1 | 4 | 16 | 64 |
| --- | ---: | ---: | ---: | ---: | ---: |
| R1 − R0 | 1 | 2 | 5 | 17 | 65 |
| R2 − R0 | 13 | 14 | 17 | 29 | 77 |

从中读出三条规则：

- **没有等待时，每个周期发射一个 bundle。** R1 − R0 = N + 1：两个相邻的读数相差 1，中间每多一个 bundle，差值加 1。标量加法链中每条依赖前一条的结果，也没有变慢：标量加法的结果在下一个周期就可用。
- **`sfence` 加 11 个周期。** R2 − R1 总是 12：一个是相邻读数之间的 1，另外 11 来自 `sfence`。
- **读数没有波动。** 同一段片段每次运行的读数都相同。指令什么时候发射只取决于程序本身，与哪一次运行无关。

所以 R2 − R0 的固定部分是 13 个周期。后文凡是写“R2 − R0”的地方，都包含这 13 个周期的读数与 `sfence` 的开销。

## bundle 数不是周期数

上面的向量加法互不依赖。把它们换成相互依赖的：每一条用上一条写出的 `v11`。实验比较四种序列，每种 N 个 bundle：

```python
'互不依赖的 vadd.f32': ('va1: vadd.8x128.f32 v11, v10, v10', 'va1: vadd.8x128.f32 v11, v10, v10'),
'相互依赖的 vadd.f32': ('va1: vadd.8x128.f32 v11, v10, v10', 'va1: vadd.8x128.f32 v11, v11, v10'),
```

`vmul.f32` 的两种序列写法相同。结果：

| 序列 | R1 − R0（N = 1、4、16） | R2 − R0（N = 1、4、16） |
| --- | --- | --- |
| 互不依赖的 `vadd.f32` | 2、5、17 | 14、17、29 |
| 相互依赖的 `vadd.f32` | 2、5、17 | 14、19、43 |
| 互不依赖的 `vmul.f32` | 2、5、17 | 14、17、29 |
| 相互依赖的 `vmul.f32` | 2、5、17 | 14、19、43 |

同样是 16 个 bundle，互不依赖时 R2 − R0 = N + 13，相互依赖时是 2N + 11。f32 的加法和乘法，结果要 2 个周期后才能被下一条指令使用；相互依赖的 16 条就要 32 个周期，而不是 16 个。清单上看不出这个差别：两种序列都是 16 个 bundle，每个 bundle 一条指令。

还有一点同样重要：四种序列的 R1 − R0 完全相同，只有 R2 − R0 不同。原因在于 TensorCore 的 bundle 要经过两次发射：

- **标量发射。** bundle 按顺序进入标量单元，没有阻塞时每个周期一个。标量槽中的指令（包括读 LCC 的 `srdreg`）在这时执行。
- **向量发射。** 含向量指令的 bundle 随后在一个队列里排队，按顺序向量发射。一条向量指令要等它的源操作数就绪、要用的单元空闲才能发射；它在等，排在它后面的 bundle 都跟着等。

R1 紧跟在 16 条乘法后面，它记录的是标量一侧走到这里的时刻，这时向量一侧的乘法还没有发射完。`sfence` 是一条标量指令，它让下一个 bundle 等到此前所有的 bundle 都已向量发射之后才标量发射。所以 `sfence` 之后的 R2 才包含了向量一侧的等待。

> 暂且可以理解为：标量一侧可以领先向量一侧最多 20 个 bundle；`sfence` 等的是此前的指令全部“发射”，不是全部“完成”。第三章第 3 节详细介绍这个队列（VIF）、`sfence` 的确切语义，以及它对计时和正确性的影响。

由此得到计时的两条规则，本教程的两个工具都已经按它们实现：

- **读数之前要有一条 `sfence`**，否则读到的只是标量一侧的时刻。
- **bundle 数只是周期数的下限。** 每个周期最多发射一个 bundle；一条指令要等的东西没有就绪，它和它后面的指令都要等。一段代码要多少周期，取决于其中的指令各自要等什么。

## 给编译出的程序计时：KernelClock

`LccProbe` 测的是插进载体的手写片段。要测一个编译出的 kernel，读数可以照样插入，问题是读数怎样带出来：被测的程序没有多余的输出可用。办法来自 SMEM 的一个性质：它的内容在程序之间保留（第三章第 1 节用实验确认）。读数先留在 SMEM 中，等被测的程序运行结束，再用一个专门的读取程序取回。`KernelClock` 的每次读数是插入的 8 个 bundle：

```tpuasm
{ s1: sst [smem:0x20100], s30 }                     # 借用 s30、s31，原值先存起来
{ s1: sst [smem:0x20101], s31 }
{ s0: sfence }                                      # 等此前的向量工作全部发射
{ s0: srdreg.lcclo s30 ; s1: srdreg.lcchi s31 }     # 读数
{ s1: sst [smem:0x20000], s30 }                     # 第 i 个读数存进 0x20000 + 2i、+ 2i + 1
{ s1: sst [smem:0x20001], s31 }
{ s1: sld s30, [smem:0x20100] }                     # 恢复
{ s1: sld s31, [smem:0x20101] }
```

`sst`、`sld` 是标量单元写、读 SMEM 的指令（本章第 8 节）。被测程序的寄存器和输出都不受影响，所以它可以是任何已编译的程序，包括原生 XLA 的程序。用法是三步：

```python
clock = tpuasm_tools.KernelClock(num_cores=2)
timed = clock.instrument(compiled, [pc0, pc1])    # 在原 bundle 编号 pc0、pc1 之前各插入一次读数
timed(x).block_until_ready()
readings = clock.read(2)                          # (TensorCore 数, 2) 的 64 位读数
```

紧挨着的两次读数相差 20 个周期，这是读数自身的开销。本教程约定：**区间的读数差减去 20，记作这段程序的周期数**；`clock.interval(compiled, first, last, call)` 把这几步包在一起，返回区间 `[first, last)` 的周期数。这个约定带有一个固定的偏移：n 个互不等待的 bundle 读作 n − 2（第三章第 3 节说明这 2 个周期的来历）。两个实现用同一种读法比较时，偏移相互抵消。

### 最小 kernel 的时间花在哪里

第 1 节的最小 kernel 把 `f32[8,128]` 从 HBM 搬进 TC VMEM，乘以 2，再搬回 HBM。它的主体是 8 个 bundle：

```tpuasm
509: { s0: dma.simple [vmem:s7], [hbm:s0], length=8, dst_flag=[sflag:52] }
510: { misc: vwait.ge [sflag:52], 8 }
511: { vld: vld.8x128 v0, [vmem:0x0] ; misc: vsyncadd.s32 [sflag:52], -8 }
512: { va0: vmul.8x128.f32 v1, 2.0, v0 }
513: { vst: vst.8x128 [vmem:0x0], v1 }
514: { s0: dma.simple [hbm:s1], [vmem:s7], length=8, dst_flag=[sflag:52] }
515: { misc: vwait.ge [sflag:52], 8 }
516: { misc: vsyncadd.s32 [sflag:52], -8 }
```

实验对四个区间各测 8 次，每个区间都只在两端读数：

```python
intervals = {
    '整个 kernel（两个 vtrace 标记之间）': (start, end + 1),
    '输入 DMA：发起到等待结束': (first_dma, first_dma + 2),
    '计算：vld、vmul、vst': (first_dma + 2, second_dma),
    '输出 DMA：发起到等待结束': (second_dma, second_dma + 3),
}
for name, (first, last) in intervals.items():
    cycles = clock.interval(compiled, first, last, call, samples=8)
```

| 区间 | 8 次运行的周期数 | 中位数 |
| --- | --- | ---: |
| 整个 kernel | 874、874、918、992、901、875、876、876 | 876 |
| 输入 DMA：发起到等待结束 | 445、445、617、446、445、445、446、446 | 445 |
| 计算：`vld`、`vmul`、`vst` | 1、1、1、1、1、1、1、1 | 1 |
| 输出 DMA：发起到等待结束 | 407、407、407、406、407、408、408、407 | 407 |

三个 bundle 的计算读作 1 个周期（3 − 2），两次 DMA 各四百多个周期。这个 kernel 的 876 个周期几乎全部在等 DMA：一次只搬 4 KiB 的 DMA，开销相当于四百多条向量指令。本章第 4 节测 DMA 的开销怎样随大小变化，第二章第 2 节讨论怎样安排 DMA 与计算的先后，使等待重叠起来。

两类读数的性质也不同。计算的读数 8 次完全相同；DMA 的读数每次相差一两个周期，偶尔多出一两百个。指令的发射是确定的；DMA 由 TensorCore 之外的搬运机构完成，什么时候完成取决于它当时的状态。所以区间内只有指令时，报告一个值；区间内有 DMA 时，报告中位数和范围。

## 把计时的边界放在 TC VMEM

上面的 kernel 以 HBM 为边界，周期数被搬运淹没了。本章研究的是每种操作本身的开销：数据已经在 TC VMEM 中，做完这一步之后结果也留在 TC VMEM 中，中间要多少周期。一个真实的 kernel 中间的每一步都是这种情形。所以本章的周期数统一这样测：**输入和结果都在 TC VMEM，区间从输入就绪开始，到结果就绪结束，区间内没有进出 HBM 的 DMA。** 原生 XLA 与 Pallas kernel 放进同一个边界比较。

### Pallas 一侧

Pallas kernel 把操作数和结果都声明在 TC VMEM 即可。这样的 kernel 没有 DMA，也没有信号量：

```python
@pl.kernel(
    out_type=pltpu.VMEM(x.shape, x.dtype),
    mesh=tc_mesh,
    name='scale',
    compiler_params=pltpu.CompilerParams(
        disable_bounds_checks=True,
        disable_semaphore_checks=True,
    ),
)
def kernel(x_vmem: Ref, o_vmem: Ref) -> None:
    o_vmem[...] = x_vmem[...] * 2.0

return kernel(pltpu.with_memory_space_constraint(x, pltpu.VMEM))
```

`out_type=pltpu.VMEM(...)` 让结果留在 TC VMEM，`pltpu.with_memory_space_constraint(x, pltpu.VMEM)` 要求操作数在进入 kernel 之前已经在 TC VMEM。这也是把一个 kernel 接在另一段片上计算后面时的写法。`tpuasm_tools.vmem_kernel(f)` 把任意函数 `f` 包成这样的 kernel：`f` 的每个参数、每个结果各对应一个 TC VMEM 中的 Ref。

### 外壳

原生 XLA 的程序没有“操作数在 TC VMEM”这种声明，要在程序里另外表达。被测的函数放进一个外壳，原生 XLA 与 Pallas kernel 用同一个（`tpuasm_tools.vmem_shell`，这里是只有一个输入时的样子）：

```python
from jax._src.state import primitives as state
from jax.experimental.layout import Layout, with_layout_constraint

ROW = Layout(major_to_minor=(0, 1), tiling=((8, 128),))

def pinned(x):
    return with_layout_constraint(state.unpin(state.pin(x, to='vmem')), ROW)

def program(x):
    result = function(pinned(x))                  # 原生 XLA 的运算，或上面的 Pallas kernel
    if isinstance(result, jax.Array):
        return pinned(result)
    return jax.lax.optimization_barrier(tuple(pinned(item) for item in result))
```

外壳做了三件事：

- **内存空间。** `state.pin(x, to='vmem')` 生成一条名为 `Pin` 的 custom call，把一个值固定在 TC VMEM；`state.unpin` 把它变回普通的数组，内存空间不变。编译后的 HLO 在每个数组的 layout 末尾标出它所在的内存空间：不写的在 HBM，`S(1)` 是 TC VMEM。
- **布局。** `with_layout_constraint(..., ROW)` 把每个端点固定成第 1 节见过的布局 `{1,0:T(8,128)}`，即 Pallas kernel 在 TC VMEM 中的二维数组所用的布局。
- **共同的终点。** 有多个结果时，它们共同经过 `jax.lax.optimization_barrier`，XLA 不能在全部结果就绪之前先处理其中一个。

> 暂且可以理解为：这三件事缺一样，两边量到的就不是同一件事。第三章第 2 节逐一去掉它们，看各自防住了什么；`pin`、`unpin` 是 JAX 的内部接口，那一节也说明为什么公开接口表达不了。

计时区间从输入的 Unpin 结束开始，到最晚的一个结果的 Unpin 结束为止，两端各读一次 LCC。`tpuasm_tools.time_in_vmem` 完成“放进外壳、编译、找出区间、插入读数、运行并核对输出”这几步：

```python
clock = tpuasm_tools.KernelClock(num_cores=1)
candidates = {'XLA': scale, 'Pallas': tpuasm_tools.vmem_kernel(scale, 'scale')}
timing = tpuasm_tools.time_in_vmem(clock, function, runs, check)
```

`runs` 是若干份输入，每份运行一次；`check` 核对每次的输出。返回值中有每次的周期数、区间内的清单，以及三项证据：各个端点在 HLO 中的类型与 layout、区间内有代码的 HLO 指令、区间内的 DMA。脚本开头调用的是 `tpu_init.initialize_one_core()`，原生 XLA 也只用一个 TensorCore（第 1 节）。

### x * 2 的周期数

对 `x * 2` 测三种大小，每种 8 份不同的输入：

```text
## x * 2，f32[8,128]
  XLA：unpin.2 是 f32[8,128]{1,0:T(8,128)S(1)}；unpin.3 是 f32[8,128]{1,0:T(8,128)S(1)}
    区间是 bundle [521, 547)，共 26 个；其中的 HLO 指令：broadcast_multiply_fusion；DMA：无
    区间内的向量指令：vld.8x128 1，vmul.8x128.f32 1，vst.8x128 1
    8 份输入的周期数：[42, 42, 42, 42, 42, 42, 42, 42]
  Pallas：unpin.2.refs_0_ 是 f32[8,128]{1,0:T(8,128)S(1)}；unpin.3 是 f32[8,128]{1,0:T(8,128)S(1)}
    区间是 bundle [521, 547)，共 26 个；其中的 HLO 指令：scale.1；DMA：无
    区间内的向量指令：vld.8x128 1，vmul.8x128.f32 1，vst.8x128 1
    8 份输入的周期数：[42, 42, 42, 42, 42, 42, 42, 42]
```

| 输入 | TC VREG 数 | 区间的 bundle 数 | XLA | Pallas |
| --- | ---: | ---: | ---: | ---: |
| `f32[8,128]` | 1 | 26 | 42 | 42 |
| `f32[64,128]` | 8 | 34 | 50 | 50 |
| `f32[1024,128]` | 128 | 154 | 170 | 170 |

两个端点都带 `S(1)`，区间内没有 DMA。三点结论：

- **两边相同。** XLA 的 fusion 与 Pallas kernel 生成的指令相同，周期数也相同。逐元素的乘法没有什么可以手写得更好的地方。
- **读数每次相同。** 8 份不同的输入，周期数完全一样。区间内只有指令，周期数是程序的性质，测一次就够，差一个周期也是确定的差别。
- **每多一个 TC VREG 多 1 个周期。** 每个 TC VREG 是 `vld`、`vmul`、`vst` 三条指令，但它们落在三个不同的槽，不同 TC VREG 的指令可以排进同一个 bundle，所以每个周期完成一个 TC VREG。128 个 TC VREG、384 条指令只要 128 个周期。

`f32[64,128]` 的清单中间是这样排的：

```tpuasm
{ va0: vmul.8x128.f32 v5, 2.0, v4 ; vst: vst.8x128 [vmem:s1], v1 ;        vld: vld.8x128 v6, [vmem:s0 + 0x18] }
{ va0: vmul.8x128.f32 v7, 2.0, v6 ; vst: vst.8x128 [vmem:s1 + 0x8], v3 ;  vld: vld.8x128 v8, [vmem:s0 + 0x20] }
{ va0: vmul.8x128.f32 v9, 2.0, v8 ; vst: vst.8x128 [vmem:s1 + 0x10], v5 ; vld: vld.8x128 v10, [vmem:s0 + 0x28] }
```

每个 bundle 做三件事：乘上一个 bundle 读进来的 TC VREG，读下一个，写出两个 bundle 之前乘好的那个。`vst` 比对应的 `vmul` 晚两个 bundle，正是上面测到的“f32 乘法的结果 2 个周期后可用”：编译器知道这个延迟，把写出排在结果就绪的那个周期，中间不留空等。

### 区间里固定的 39 个周期

一个 TC VREG 是 42 个周期，而计算只有 3 个 bundle。其余的周期在哪里，可以直接量出来：起点的读数不动，把终点的读数逐个 bundle 向后移，每个位置各运行一次。

```python
for end in range(first + 1, last + 1):
    cycles, = clock.interval(compiled, first, end, lambda timed: check(timed(x), x))
```

```text
  到 bundle 521 为止：  0    { s1: sld s1, [smem:0x3ffe6] }
  到 bundle 522 为止：  3    { s0: sadd.s32 s14, s6, s1 ; s1: sld s13, [smem:s6 + s1] }
  到 bundle 523 为止：  7    { s0: sadd.s32 s15, 1, s13 }
  到 bundle 524 为止：  8    { s1: sst [smem:s14], s15 }
  到 bundle 525 为止：  9    { misc: vtrace 0x80000003 }    # scale.1 开始
  到 bundle 526 为止： 10    { s0: simm.s32 s0, 8 ; s1: simm.s32 s1, 0 }
  到 bundle 527 为止： 11    {}
  到 bundle 528 为止： 12    { vld: vld.8x128 v0, [vmem:s0] }
  到 bundle 529 为止： 13    { va0: vmul.8x128.f32 v1, 2.0, v0 }
  到 bundle 530 为止： 14    { vst: vst.8x128 [vmem:s1], v1 }
  到 bundle 531 为止： 15    { misc: cfence }
  到 bundle 532 为止： 16    { misc: vtrace 0x90000003 }
  到 bundle 533 为止： 17    { s0: simm.s32 s17, 0 ; s1: sld s16, [smem:0x3ffe6] }
  到 bundle 534 为止： 21    { s0: sadd.s32 s19, s17, s16 ; s1: sld s18, [smem:s17 + s16] }
  到 bundle 535 为止： 25    { s0: sadd.s32 s20, 1, s18 }
  到 bundle 536 为止： 26    { s1: sst [smem:s19], s20 }
  到 bundle 537 为止： 27    { misc: vtrace 0x80000004 }    # pin.3 开始
  到 bundle 538 为止： 28    { misc: cfence }
  到 bundle 539 为止： 29    { misc: vtrace 0x90000004 }
  到 bundle 540 为止： 30    { s1: sld s21, [smem:0x3ffe6] }
  到 bundle 541 为止： 34    { s0: sadd.s32 s23, s17, s21 ; s1: sld s22, [smem:s17 + s21] }
  到 bundle 542 为止： 38    { s0: sadd.s32 s24, 1, s22 }
  到 bundle 543 为止： 39    { s1: sst [smem:s23], s24 }
  到 bundle 544 为止： 40    { misc: vtrace 0x80000005 }    # unpin.3 开始
  到 bundle 545 为止： 41    { misc: cfence }
  到 bundle 546 为止： 42    { misc: vtrace 0x90000005 }
```

区间里有三条 HLO 指令：被测的运算、结果的 Pin、结果的 Unpin。每条的结构相同：

- 前面是 runtime 的 4 个 bundle：从 SMEM 读一个指针、读一个计数、加 1、写回。相邻两条 `sld` 至少相隔 4 个周期，用 `sld` 结果的指令也要等 4 个周期（本章第 8 节），所以这 4 个 bundle 占 10 个周期（区间开头的那一组因为读法的偏移读作 8）。
- 然后是一对 `vtrace` 标记，中间是这条指令自己的代码，以一条 `cfence` 结尾。Pin 和 Unpin 没有代码，只有这三个 bundle。

Pin 与 Unpin 各 13 个周期，被测运算的框架（4 个 runtime bundle、两个标记、`cfence`、装入地址的 `simm` 和一个空 bundle）也是 13 个周期，合计 39 个；剩下的 3 个周期才是 `vld`、`vmul`、`vst`。中间的每个 bundle 恰好一个周期。

所以读本章之后的周期数时要记住：**区间里有 39 个周期是外壳固定的，对原生 XLA 和 Pallas 相同；超出的部分才是被测的运算。** 一个运算如果被 XLA 拆成几条 HLO 指令，每多一条就多一份 13 个周期左右的框架，这是 XLA 那种实现的一部分，计入它的周期数。

这 39 个周期有一处会缩短。Pin 和 Unpin 前面各有 10 个周期花在标量一侧的 `sld` 上。被测的运算如果自己要在向量一侧等很久（例如等某个单元算出结果），标量一侧不会跟着停，这两组 `sld` 在等待期间就已经发射完，20 个周期与等待重叠，固定的部分只剩 19 个。后面遇到的运算中，没有等待的固定部分是 39 个周期，有几十个周期等待的是 19 个，等待较短的介于两者之间。

这种“起点不动、逐个移动终点”的办法以后还会用到：它给出区间里每个 bundle 发射的时刻，等待集中在哪一条指令上，一眼就能看出来。注意它每次运行只有两次读数。不能在一次运行里插很多次读数再逐段相减：每次读数带一条 `sfence`，会打断前后指令的重叠（第三章第 3 节）。

## 周期数由哪几类开销组成

本节已经出现了几种不同性质的开销，后面各节测到的周期数都由它们组成：

| 开销 | 例子 | 性质 |
| --- | --- | --- |
| 发射一个 bundle | 任何 bundle | 1 个周期；同一个 bundle 中不同槽的指令同时发射 |
| 结果可用的延迟 | f32 加、乘 2 个周期，`sld` 4 个周期 | 只有下一条指令依赖它时才要等；不依赖它的指令可以填在中间 |
| 提交—取回的往返 | EUP、XLU、MXU（本章第 7、9–12 节） | 几个到几十个周期；期间可以提交别的工作 |
| 单元的发射间隔 | XLU、MXU 每 8 个周期接受一次提交 | 往返很多、互不依赖时，时间由它决定 |
| 等待外部事件 | DMA（本章第 4 节、第二章） | 几百个周期起，每次略有不同 |

设计一个 kernel，就是数清这些资源各被谁占用、占多久，再安排先后。上面 `x * 2` 的例子是最简单的情形：三条指令落在三个不同的槽，互相错开一个 bundle，每个周期完成一个 TC VREG。后面每认识一个硬件单元，就测出它的延迟和发射间隔；这些数汇总在[本章的首页](../README.md)。把它们组合成一个可以从清单推算周期数的模型，是第三章第 4 节的内容；用周期数优化一个 kernel 的完整顺序，见[第三章的首页](../../chapter03/README.md)。

## LCC 的适用范围

LCC 数的是本芯片时钟的周期，每个 TensorCore 各有一个。由此有两条限制：

- **读数只在同一个 TensorCore 上可比。** 两次读数相减才有意义。不同 TensorCore 的 LCC 起点不同，不能直接相减；不同芯片各有自己的时钟，周期的长短还相差百万分之几。
- **周期数不是秒。** 换成秒要知道时钟的频率。第三章第 5 节用主机时钟标定出 1050 MHz。

LCC 适合回答“这段代码在这个 TensorCore 上要多少周期”，这正是优化一个 kernel 时要问的问题。要比较不同 TensorCore、不同芯片上的事件，用第三章第 5 节的全局时钟 GTC。

[tpu-v4-latency-numbers](../../../tpu-v4-latency-numbers/README.md) 用同样的载体与读法测量了第二章引用的全部 DMA 开销。[研究报告 48](../../../pallas-tpu-readings-dev/research_reports/48_scalar_cycle_lowering.md) 走了另一条路：为读 LCC 增加一个 Pallas primitive 和它的 lowering，由编译器生成读数指令。两条路得到的读法相同。
