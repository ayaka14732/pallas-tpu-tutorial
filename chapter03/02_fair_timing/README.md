# 公平计时

比较两个实现谁更快，首先要保证两边测的是同一件事：同样的输入从同样的地方出发，结果写到同样的地方，计时覆盖同样的范围。本节用第二章第 2 节的例子，把原生 XLA 与 Pallas kernel 作为完整的程序（HBM 到 HBM）放在一起，用五种方法计时，看每种方法在哪里出错，以及怎样从编译结果判断两边实际做了什么；再回到第一章一直在用的 TC VMEM 边界，逐一去掉外壳的各个部分，看它们各自防住了什么。

## 比较的对象

本小节实验[源码](01_jax_pallas_timing_methods.py)、[输出](01_jax_pallas_timing_methods.txt)。

运算是 `f32[32768,128]`（16 MiB）的 `y = 2x + 1`，有三个候选：

```python
functions = {
    'XLA': lambda x: x * 2.0 + 1.0,
    'Pallas（out_type=pltpu.HBM）': pallas_hbm,
    'Pallas（out_type=ShapeDtypeStruct）': pallas_any,
}
```

两个 Pallas 候选都是第二章第 2 节输入输出双缓冲、`f32[2048,128]` tile 的 kernel，只改了一行：

```python
out_type=pltpu.HBM(x.shape, x.dtype) if out_hbm else jax.ShapeDtypeStruct(x.shape, x.dtype),
```

## 先看编译结果

计时之前，先检查编译后的 HLO 中三件与计时边界有关的事：有没有跨程序预取（`cross_program_prefetch`，把输入提前搬进片上内存）、有没有数组被放进 Megacore Shared CMEM（layout 中的 `S(3)`）、程序的结果（`ROOT`）是哪条指令：

```text
XLA：cross_program_prefetch 无，S(3) 0 处，ROOT 是 fusion
Pallas（out_type=pltpu.HBM）：cross_program_prefetch 无，S(3) 0 处，ROOT 是 copy
Pallas（out_type=ShapeDtypeStruct）：cross_program_prefetch 无，S(3) 0 处，ROOT 是 custom-call
```

`out_type=pltpu.HBM` 的版本在 kernel 之后多了一次 copy：kernel 写出的数组被要求留在 HBM 内存空间，而程序的结果使用默认内存空间，XLA 只好再复制一遍。这次 16 MiB 的复制占 36256 个周期，与 kernel 本身相当。第二章第 2 节用 `pltpu.HBM` 是为了在 kernel 内部重复计时时固定数据通路，那里测的是 kernel 的每遍时间，不受这次复制影响；但作为一个完整的程序与 XLA 比较时，应当用 `ShapeDtypeStruct`，让 kernel 直接写程序的结果。

## 五种方法

| 方法 | XLA | Pallas，`pltpu.HBM` | Pallas，`ShapeDtypeStruct` |
| --- | ---: | ---: | ---: |
| 1. 调用并等到结果，固定同一个输入 | 231.5 µs | 275.3 µs | 235.5 µs |
| 2. 调用并等到结果，每次一个新输入 | 232.1 µs | 271.1 µs | 236.8 µs |
| 3. LCC，kernel 或 fusion | 36256 个周期 | 43774 + 36256 个周期 | 43846 个周期 |
| 3. LCC，TensorCore 1 的整个程序 | 36373 个周期 | 80211 个周期 | 43964 个周期 |
| 4. `fori_loop` 重复 16 与 32 次之差 | 9.5 µs | 43.3 µs | 31.2 µs |
| 5. 方法 2 按正序、倒序交替做 4 轮 | 220.9–224.5 µs | 266.1–268.9 µs | 227.7–234.8 µs |

方法 1、2、4 用主机时钟，单位是微秒；方法 3 是设备上读出的周期数（36256 个周期是 34.5 µs）。

**方法 1、2：主机计时。** 第 1 节说过，等到结果的时间包含 100 多微秒的固定开销，本次约 200 µs。XLA 与 `ShapeDtypeStruct` 版本在设备上差 7590 个周期（约 7.2 µs）；主机计时的差在方法 1、2 中分别为 4.0、4.7 µs。同一个候选隔一会儿再测，结果也能相差数微秒（方法 5）。主机计时只适合差别远大于开销波动的比较。

方法 1 与方法 2 的区别是输入是否每次都是新数组。这里两者走相同的设备数据通路，因为 XLA 没有做跨程序预取；在第二章第 9 节的矩阵乘法中，XLA 会把输入预取进 CMEM，固定输入时下一次调用可能直接用上片上的副本。正式比较应当每次用一个新的输入数组（[研究报告 28](../../../pallas-tpu-readings-dev/research_reports/28_xla_pallas_kernel_fair_timing.md) 称为 fresh-HBM），并且在预热之后计时。

**方法 3：设备上的 LCC。** 第 1 节的 `KernelClock.time_ops` 在整个程序和每条 HLO 指令的起止标记处读周期计数器：

```python
for op, cycles in clock.time_ops(compiled, lambda timed: timed(next(inputs)).block_until_ready(), samples=16):
    print(f'{name}，{op}：TensorCore 0 {cycles[0]} 个周期，TensorCore 1 {cycles[1]} 个周期')
```

统计时先对每次调用的结束、开始 LCC 读数求差并扣除读数开销，再取这些耗时的中位数。不能先分别对绝对时钟取中位数再相减：时钟单调递增，那样只会选中时间顺序中间的一两次调用。本节输出已用修正后的 `KernelClock.time_ops` 重跑。

差别清楚：XLA 的 fusion 36256 个周期，Pallas kernel 43846 个周期，16 次运行的中位数，每次都是新的输入。比较完整程序时应看整个程序的周期数：它包含 kernel 之外的 copy 等其他指令。TensorCore 0 的整个程序还包含一段 kernel 之前的 runtime 代码，这里约 3 万个周期，每次运行都不一样（第 1 节）；两个候选都有这一段，用 TensorCore 1 的整个程序比较。

**方法 4：在 jit 中循环。** 把调用放进 `jax.lax.fori_loop`，用循环 32 次与 16 次之差消去固定开销：

```python
looped = tpuasm_tools.compile(lambda x: jax.lax.fori_loop(0, count, lambda i, y: function(y), x), x, mesh=meshes[name])
```

这看起来与第二章第 2 节“在 kernel 内部重复”的思路相同，结果却完全不同：XLA 每次只要 9.5 µs（约 1 万个周期），比方法 3 的 36256 个周期快了近 4 倍。原因在编译结果中：

```text
XLA：循环版本的 HLO 中 cross_program_prefetch 无，S(3) 13 处，ROOT 是 copy-done
Pallas（out_type=ShapeDtypeStruct）：循环版本的 HLO 中 cross_program_prefetch 无，S(3) 12 处，ROOT 是 copy-done
```

循环中传递的数组被 XLA 放进了 Megacore Shared CMEM，每次迭代读写的都是片上内存，不再经过 HBM。测到的是另一个程序。`pltpu.HBM` 版本的输出被固定在 HBM，所以循环中的时间（43.3 µs，约 45000 个周期）仍接近 kernel 本身。改变程序结构的计时方法，都要重新检查编译结果。

**方法 5：轮次与顺序。** 方法 2 按正序、倒序交替做 4 轮：

```text
第 1 轮：XLA 224.5 µs，Pallas（pltpu.HBM）266.9 µs，Pallas（ShapeDtypeStruct）229.9 µs
第 2 轮：XLA 223.5 µs，Pallas（pltpu.HBM）266.1 µs，Pallas（ShapeDtypeStruct）227.7 µs
第 3 轮：XLA 220.9 µs，Pallas（pltpu.HBM）266.3 µs，Pallas（ShapeDtypeStruct）234.8 µs
第 4 轮：XLA 223.9 µs，Pallas（pltpu.HBM）268.9 µs，Pallas（ShapeDtypeStruct）232.3 µs
```

四轮之中，XLA 的中位数在 220.9–224.5 µs 之间，`ShapeDtypeStruct` 在 227.7–234.8 µs 之间；它们与几分钟前的方法 2 又不相同。主机一侧的开销在随时间漂移，幅度与设备上的差别（约 7.2 µs）相近。两个候选若不在相邻的时间测，只跑一轮，结论可能取决于运气。交替顺序可以抵消“先跑的吃亏”或“后跑的吃亏”这类与顺序有关的漂移。研究报告 28 的做法是：每个候选在自己的连续块中运行，块内取中位数，多轮之间正反交替，最后取各轮中位数的中位数；需要彻底清除片上状态时，每轮用一个新进程。

## TC VMEM 边界：外壳的三件事各防住了什么

本小节实验[源码](02_jax_pallas_vmem_boundary.py)、[输出](02_jax_pallas_vmem_boundary.txt)。

上面的比较以 HBM 为边界，周期数的大部分是搬运。第一章从第 3 节起用的是另一种边界：输入已经在 TC VMEM，结果也留在 TC VMEM，区间内没有进出 HBM 的搬运。一个融合的 kernel 中间的任何一步都是这种情形。那里给出了做法（`tpuasm_tools.vmem_shell`），这里说明它为什么必须这样做。

Pallas 一侧，kernel 把操作数和结果都声明在 TC VMEM（`out_type=pltpu.VMEM(...)` 与 `pltpu.with_memory_space_constraint(x, pltpu.VMEM)`），没有 DMA，也没有信号量。原生 XLA 的程序没有这种声明，所以两边都放进同一个外壳：

```python
def pinned(x):
    return with_layout_constraint(state.unpin(state.pin(x, to='vmem')), ROW)

def program(x):
    result = function(pinned(x))                  # 原生 XLA 的运算，或 Pallas kernel
    if isinstance(result, jax.Array):
        return pinned(result)
    return jax.lax.optimization_barrier(tuple(pinned(item) for item in result))
```

外壳表达了三件事，下面的实验逐一去掉它们。

**内存空间。** 编译后的 HLO 在每个数组的 layout 末尾用 `S(n)` 标明它所在的内存空间：不写的在 HBM，`S(1)` 是 TC VMEM，`S(3)` 是 Megacore Shared CMEM。`state.pin(x, to='vmem')` 生成一条名为 `Pin` 的 custom call，把一个值固定在 TC VMEM；`state.unpin` 把它变回普通的数组，但内存空间不变。这是 JAX 的内部接口（`jax._src.state.primitives`），因为公开接口表达不了：数组的 `memory_kind` 只有 `device`（HBM）和两种主机内存；公开的 `jax.ref.new_ref(x, memory_space=pltpu.VMEM, pin=True)` 在当前版本中没有把目标内存空间传给 Pin，结果正确但值仍在 HBM。

**布局。** layout 中的 `{1,0:T(8,128)}` 描述数组怎样摆放：`{1,0}` 是各维从变化最快到最慢的次序，第 1 维（列）变化最快；`T(8,128)` 表示按 8 个子通道乘 128 个通道切成 tile（第一章第 1 节）。Pallas kernel 在 TC VMEM 中的二维数组就是这种布局。`with_layout_constraint(..., ROW)` 把端点固定成它，其中 `Layout` 的 `major_to_minor=(0, 1)` 是从最慢到最快的次序，与 HLO 的写法相反。

**共同的终点。** 有多个结果时，它们共同经过 `jax.lax.optimization_barrier`，XLA 不能在全部结果就绪之前先处理其中一个。

计时区间从输入的 Unpin 结束开始，到最晚的一个结果的 Unpin 结束为止，两端各读一次 LCC。`hlo_ops` 给每一对 `vtrace` 起止标记配上 HLO 指令的名字（第 1 节），区间的两端由此找到：

```python
ops = tpuasm_tools.hlo_ops(compiled)
unpins = [end for name, _, end in ops if name.startswith('unpin')]
first, last = unpins[0] + 1, max(unpins[1:]) + 1
timed = clock.instrument(compiled, [first, last])
```

脚本对每个程序打印三项证据：各个端点在 HLO 中的类型与 layout、区间内有代码的 HLO 指令、区间内的 DMA。

### 基本形式：y = 2x + 1

`f32[1024,128]` 的 `y = 2x + 1`，8 份不同的输入各运行一次：

```text
XLA：unpin.2 是 f32[1024,128]{1,0:T(8,128)S(1)}；unpin.3 是 f32[1024,128]{1,0:T(8,128)S(1)}
  区间是 bundle [521, 677)，其中的 HLO 指令：multiply_add_fusion；DMA：无
  区间内的向量指令：vadd.8x128.f32 128，vld.8x128 128，vmul.8x128.f32 128，vst.8x128 128
  8 份输入的周期数：[172, 172, 172, 172, 172, 172, 172, 172]
Pallas：unpin.2.x_vmem 是 f32[1024,128]{1,0:T(8,128)S(1)}；unpin.3 是 f32[1024,128]{1,0:T(8,128)S(1)}
  区间是 bundle [521, 677)，其中的 HLO 指令：affine.1；DMA：无
  区间内的向量指令：vadd.8x128.f32 128，vld.8x128 128，vmul.8x128.f32 128，vst.8x128 128
  8 份输入的周期数：[172, 172, 172, 172, 172, 172, 172, 172]
```

两个端点都带 `S(1)`，区间内没有 DMA，XLA 的 fusion 与 Pallas kernel 都是 172 个周期。本节开头的比较中 XLA 的 36256 个周期与 Pallas 的 43846 个周期之差，全部来自搬运的安排，与计算无关。边界不同，回答的是不同的问题。

### 只去掉布局约束

换成 `f32[16,8]`，其余不变，再去掉 `with_layout_constraint`：

| | 有布局约束 | 去掉布局约束 |
| --- | --- | --- |
| XLA 的端点 | `f32[16,8]{1,0:T(8,128)S(1)}` | `f32[16,8]{0,1:T(8,128)S(1)}` |
| XLA 的周期数 | 46 | 43 |
| Pallas | 46 | 编译失败 |

没有约束时，XLA 为 `[16,8]` 选了转置的布局 `{0,1}`：16 行沿通道方向摆放，整个数组只占一个 tile，行优先时要占两个。它比行优先的版本少 3 个周期，但两边已经不是从同样摆放的数据出发。Pallas 一侧则直接编译失败：

```text
INVALID_ARGUMENT: during context [post-layout]: Seen buffers while buffers aren't allowed in this context: %copy.1 = b(f32[16,8]{0,1:T(8,128)}) copy(%pin.3)
```

kernel 的结果是行优先的，程序的结果被 XLA 选成了转置的布局，XLA 于是在 Pin 与 Unpin 之间插入一条改变布局的 `copy`，而这个位置不允许复制。布局约束要加在 Unpin 之后，输入和每个结果都要加。

### 两个结果：top-k

把运算换成第一章第 15 节的 `jax.lax.top_k(x, 8)`，输入是 `f32[8,128]`，每行是 0 到 127 的随机排列，结果是值和下标两个数组：

| | 区间内的 HLO 指令 | 区间内的 DMA | 8 份输入的周期数 |
| --- | --- | --- | --- |
| XLA，`is_stable=True` | `iota`、`sort` | cmem → vmem | 2721–2725 |
| XLA，`is_stable=False` | `iota`、`sort` | cmem → vmem | 2563–2566 |
| Pallas | `top_k.1` | 无 | 685（8 次相同） |

三个端点都是 `{1,0:T(8,128)S(1)}`。这就是第一章第 15 节的那组数：Pallas 的 8 轮 argmax 是 685 个周期，XLA 把整行连同下标一起排序，较快的 `is_stable=False` 也要约 2564 个周期。

XLA 的读数不再每次相同。它的 `iota` 生成的下标数组放在 CMEM（HLO 中是 `s32[8,128]{1,0:T(8,128)S(3)}`），排序之前先用一次 DMA 搬进 TC VMEM。等这次 DMA 的那条 `vwait` 何时放行，取决于存储系统当时的状态，所以读数有几个周期的波动。这次搬运是 XLA 所选算法的一部分，计入它的周期数；波动则说明，区间内只要有 DMA，就要多次运行、报告中位数和范围。

### 只去掉 optimization_barrier

| | 区间内的 HLO 指令 | 区间内的 DMA | 周期数 |
| --- | --- | --- | --- |
| XLA，`is_stable=True` | `iota`、`sort`、`copy.4` | cmem → vmem、vmem → hbm | 3136–3140 |
| Pallas | `top_k.1`、`copy.4` | vmem → hbm | 1099、1100 |

没有 barrier 时，XLA 先把第一个结果写出到 HBM（`copy.4`），再处理第二个结果的 Pin 和 Unpin。“两个结果都在 TC VMEM”的时刻之前，已经发生了一次对外的搬运，它落在区间内，两边都多出 400 多个周期。经过 barrier 之后，两个结果的 Pin、Unpin 都排在第一次对外写出之前。

### 不要把逐条指令的读数相加

`KernelClock.time_ops` 给每条 HLO 指令各插一对读数，可以用来看时间花在哪条指令上。但用它们的和代替连续的区间是不对的：

```text
XLA（is_stable=True）：iota 43，sort 2634；合计 2677
Pallas：top_k.1 661；合计 661
```

合计比区间少四十多个和 24 个周期：Pin、Unpin 的标记和指令之间的衔接代码不属于任何一条被求和的指令。更重要的是，每次读数都带一条 `sfence`，会打断它前后的指令原本的重叠（第 3 节用实验说明这一点）。比较两个实现时，区间两端各读一次，中间不放任何读数。

区间也不能缩到 kernel 中间的某一段。与输入无关的准备工作，例如常数、掩码、下标数组，同样是这种写法的开销；如果把起点放在 kernel 内部某次等待之后，编译器会把这些工作排到起点之前，读数就偏小，而且不同的写法偏小的程度不同。所以脚本列出区间内的 HLO 指令：被测运算的全部指令（包括 XLA 那条不依赖输入的 `iota`）都应当在其中。

这套外壳来自 [top-k 论文](../../../tpu-v4-top-k/report.md)的 3.4 节，论文用它在 25 个形状上比较原生 XLA、Pallas 的官方实现和论文的写法。那里还记录了一处相关的缺陷：XLA 把排序与取前 k 列融合成一条 fusion 时，Pin 的内存空间没有传到 fusion 内部，在这个外壳下编译失败（第一章第 15 节的 `f32[8,256]` 就是这种情形）。

## 公平比较的清单

1. 写明比较的边界。比较完整的程序时：输入从哪里出发（HBM，每次新数组）、结果写到哪里（程序的结果，在 HBM）、用几个 TensorCore。比较一段计算时：输入和结果都在 TC VMEM，用第一章第 3 节的外壳，区间内不应有进出 HBM 的 DMA。
2. 检查两边的编译结果：有没有额外的 copy、跨程序预取、放进 CMEM 的数组，端点的内存空间与布局是否相同。先确认输入起点与输出终点相同，再区分额外输出 copy、提前准备的输入，以及算法内部的 CMEM 暂存。程序内部把本次 HBM 输入复制到 CMEM，只要这次 copy 包含在计时内，仍属于同一边界下的实现选择；不能仅因 `S(3)` 数量不同就认定比较不公平。
3. 设备时间用设备上的周期计数器，区间两端各读一次，被测运算的全部指令都要落在区间内；主机时间只用于差别远大于开销波动的场合。
4. 区间内只有指令时，读数每次相同，一次就够。区间内有 DMA 或用主机时钟时，多轮、交替顺序，报告中位数，并保留每轮的结果。
5. 插入读数之后的程序也要核对输出。读数改写了程序，改写之后的程序算错时，它的周期数没有意义。

按这张清单，本例的结论是：在同样从 HBM 读、写回 HBM 的边界下，XLA 的 fusion 36256 个周期，单个 TensorCore 的 Pallas kernel 43846 个周期。XLA 用了两个 TensorCore（第二章第 2 节），这是两个实现之间真实的差别，而不是计时的问题；第二章第 1 节的方法可以让 Pallas kernel 也用上两个 TensorCore。
