# LCC：指令级计时

主机计时（第 1、2 节）只能看到整个程序的时间，而且淹没在主机的开销里。要知道一个 kernel、或其中某几个 bundle 花了多少周期，需要在 kernel 内部读一个周期计数器。在一个 TensorCore 上计时，这是本教程首选的方法：读数就是周期，没有换算，也不依赖 profiler。TensorCore 有一个本地周期计数器 LCC（local cycle counter），每个周期加 1，标量单元可以用一条指令读出它的值。本节写出读取 LCC 的方法，用它验证第一章第 2 节的说法“没有等待时，TensorCore 每个周期发射一个 bundle”，并测出标量单元中哪些指令会让发射停下来。

## 读 LCC 的指令

LCC 是 64 位的，标量寄存器是 32 位的，所以要分两半读：

```text
{ s0: srdreg.lcclo s20 ; s1: srdreg.lcchi s25 }
```

`srdreg` 读一个特殊寄存器（special register），`lcclo` 和 `lcchi` 是 LCC 的低、高 32 位，结果写进标量寄存器。两条指令放在同一个 bundle 的两个标量槽 `s0`、`s1` 中，在同一个周期执行，得到的两半属于同一个时刻，拼起来就是一致的 64 位读数。如果分在两个 bundle 读，两次读之间低位可能恰好从 `0xFFFFFFFF` 进位到 0，拼出的值会差 2³²。

Pallas 的公开接口没有读 LCC 的函数，所以本章用 tpuasm 把这样的 bundle 直接插进机器程序。

## 载体：LccProbe

[`tpuasm_tools.LccProbe`](../../tpuasm_tools.py) 把“编译载体、插入片段、取回读数”包装成一个类。

### 载体 kernel

它先用 Pallas 编译一个普通的 kernel：

```python
def kernel(x_hbm, out_hbm, data, sem) -> None:
    pltpu.async_copy(x_hbm, data, sem).wait()
    # 带唯一立即数的 vxor 标出插入位置。
    data[:8, :] = data[:8, :] ^ jnp.uint32(0x13579BDF)
    core = jax.lax.axis_index('tc')
    pltpu.async_copy(data.at[:72, :], out_hbm.at[pl.ds(core * 72, 72)], sem).wait()
```

输入是 `u32[256,128]` 的随机数，先整块搬进 TC VMEM 的 `data`；然后对前 8 行做一次异或，常数 `0x13579BDF` 在整个程序中只出现一次，`find_bundles` 由它找到这条 `vxor` 所在的 bundle，作为插入位置；最后把 `data` 的前 72 行（9 个 tile）写回 HBM。`num_cores=2` 时两个 TensorCore 执行同一个程序，各自写回输出的一半（第 6 节用到）。

构造时还用 `edit_bundles` 把这条 `vxor` 改成 `vmov`，让输出的第 0 个 tile 等于输入，不再依赖异或的结果。

### 插入的三段

`run(body)` 在插入位置之前放入三段清单：

1. **前缀。** 先把载体此后还要用到的标量寄存器 s20–s30 广播进 v20–v30 保存（载体在插入点之后用这些寄存器计算输出地址，片段会改写它们）；再把输入的第 0 个 tile 读进 `v10`，供片段当作数据使用；最后执行 `setup`，并用一条 `sfence` 让此前的工作全部发射完，使计时从一个空的队列开始（第 4 节解释为什么需要这一条）。
2. **片段 `body`。** 由调用者写出，其中用 `read_lcc(20)`、`read_lcc(21)`、`read_lcc(22)` 三次读 LCC，记作 R0、R1、R2。
3. **后缀。** 把六个半字读数各用 `vmov` 广播成一个 TC VREG，用 `vst` 存进 TC VMEM 的第 1–6 个 tile，随载体的输出 DMA 返回主机；再用 `vpush`/`spop` 从 v20–v30 恢复标量寄存器。

主机把每对半字拼成 64 位，返回每次运行的 `(R1 − R0, R2 − R0)`。插入后的程序不再经过编译器，bundle 的内容和顺序就是设备实际执行的内容。`LccProbe.program(body)` 返回插入后的程序，可以用 `full_listing` 查看：

```text
{ s0: sfence }
{ s0: srdreg.lcclo s20 ;
  s1: srdreg.lcchi s25 }
{ va0: vadd.8x128.s32 v11, 1, v10 }
{ va0: vadd.8x128.s32 v11, 1, v10 }
{ s0: srdreg.lcclo s21 ;
  s1: srdreg.lcchi s26 }
{ s0: sfence }
{ s0: srdreg.lcclo s22 ;
  s1: srdreg.lcchi s27 }
{ misc: vnop }  …
{ va0: vmov.8x128 v12, s20 }
{ misc: vnop }  …
{ vst: vst.8x128 [vmem:0x8], v12 }
```

这是下面实验中 N = 2 条向量加法的片段：前缀最后的 `sfence`，R0，两条 `vadd`，R1，`sfence`，R2，接着是后缀中第一个读数的广播与写回（`vmov` 与 `vst` 之间的 8 个 `vnop` 等待广播结果写完）。

片段用两个小函数写出：`bundle(text)` 生成一个 bundle 的清单文本，`text` 为空时是空 bundle；`read_lcc(low)` 生成上面那个读 LCC 的 bundle，低位写进 `s{low}`、高位写进 `s{low + 5}`。

## 实验：几种 N 个 bundle 的序列

本小节实验[源码](01_tpuasm_lcc_intervals.py)、[输出](01_tpuasm_lcc_intervals.txt)。

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

- **R1 − R0 = N + 1。** 两个相邻的读数相差 1，中间每多一个 bundle，差值加 1：这些 bundle 都是一个周期发射一个。标量加法链中每条依赖前一条的结果，也没有变慢：标量加法的结果在下一个周期就可用。
- **向量指令不改变 R1。** 向量加法和 `vld` 与空 bundle 的读数相同。这不说明向量指令都只要一个周期，而是说明 R1 只反映标量一侧走到了哪里。第 4 节会看到，向量一侧可以远远落在后面，R1 却看不到。
- **`sfence` 加 11 个周期。** R2 − R1 总是 12：一个是相邻读数之间的 1，另外 11 来自 `sfence`。第 4 节解释它的含义。

第一章第 2 节说“TensorCore 每个周期最多发射一个 bundle”，这里给出了确切的版本：每个周期一个 bundle 指的是标量发射，LCC 读数数的也是标量发射。这四种序列的读数完全确定，没有任何波动。但不能由此把 bundle 数当作周期数：向量指令在标量发射之后还要由硬件的计分板放行，源操作数未就绪、要用的单元未空闲时就得等待，这些等待 R1 看不到。本实验的向量指令互不依赖，又有末尾的 `sfence` 兜底，才显得每个 bundle 一个周期；下一节换成相互依赖的向量指令，差别就出现了。

## 实验：标量单元中会等待的指令

本小节实验[源码](02_tpuasm_scalar_timing.py)、[输出](02_tpuasm_scalar_timing.txt)。

上面的标量加法不会让发射停下。标量单元还有乘法、读写 SMEM 的指令，第一章第 7 节的清单中就有成对的 `sld`。片段的形式不变，只换中间的序列：

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

这解释了第一章第 7 节清单中的一个现象：编译器把两个标量参数的 `sld` 放在相邻两个 bundle 中，第二条要等 4 个周期；如果标量参数很多，按每个 4 个周期估算读入的时间，或者把它们的读取与其他标量工作交错。反过来把标量放进 TC VREG 再取出并不划算：向量到标量要经过 `vpush`/`spop`，等待 43 个周期（第 5 节）。

## 给整个 kernel 计时：KernelClock

本小节实验[源码](03_tpuasm_kernel_clock.py)、[输出](03_tpuasm_kernel_clock.txt)。

`LccProbe` 测的是插进载体的手写片段。要测一个真实的 kernel，或 kernel 中的一段，读数可以照样插入，问题是读数怎样带出来：被测的程序没有多余的输出可用。办法来自 SMEM 的一个性质：它的内容在程序之间保留。实验让程序 A 把一个数写进 SMEM 的某个地址，再让另一个程序 B 去读：

```text
程序 A 写入 0 之后，程序 B 读到：[0, 0]（两个 TensorCore）
程序 A 写入 20261004 之后，程序 B 读到：[20261004, 20261004]
```

于是读数可以先留在 SMEM 中，等被测的程序运行结束，再用一个专门的读取程序取回。[`tpuasm_tools.KernelClock`](../../tpuasm_tools.py) 就是这样做的。每次读数是插入的 8 个 bundle（`clock_read`）：

```text
{ s1: sst [smem:0x20100], s30 }                     # 借用 s30、s31，原值先存起来
{ s1: sst [smem:0x20101], s31 }
{ s0: sfence }                                      # 等此前的向量工作全部发射（第 4 节）
{ s0: srdreg.lcclo s30 ; s1: srdreg.lcchi s31 }     # 读数
{ s1: sst [smem:0x20000], s30 }                     # 第 i 个读数存进 0x20000 + 2i、+ 2i + 1
{ s1: sst [smem:0x20001], s31 }
{ s1: sld s30, [smem:0x20100] }                     # 恢复
{ s1: sld s31, [smem:0x20101] }
```

地址 `0x20000` 在 SMEM 的中部，远离 kernel 的 scratch（从低地址分配）和 runtime 使用的最高一段。被测程序的寄存器、输出都不受影响，所以它可以是任何已编译的程序，包括原生 XLA 的程序。

`KernelClock` 的用法是三步：

```python
clock = tpuasm_tools.KernelClock(num_cores=2)
timed = clock.instrument(compiled, [pc0, pc1, ...])    # 在原 bundle 编号 pc_i 之前插入第 i 次读数
timed(x).block_until_ready()
readings = clock.read(count)                           # (TensorCore 数, count) 的 64 位读数
```

### 在哪里读：程序与 kernel 的起止标记

第一章第 2 节的清单中，kernel 段以 `vtrace 0x80000000` 开始、以 `vtrace 0x90000000` 结束。这是编译器为每条 HLO 指令放置的起止标记，整个程序也有一对，序号是 `0xfffffff`（第 7 节解释它们的编码）。`tpuasm_tools.hlo_ops` 把它们找出来：

```text
[('module', 460, 550), ('wait.1', 503, 535)]
```

实验用第 1 节那个设备时间已知的 kernel（`vdelay` 10500 或 1050000 个周期），在这四个位置读数；kernel 的起点和终点各连读两次，看读数自身占多少周期：

```python
timed = clock.instrument(compiled, [module_start, start, start, end + 1, end + 1, module_end + 1])
```

| | 程序开始 → kernel 开始 | 相邻两次读数 | kernel | kernel 结束 → 程序结束 |
| --- | ---: | ---: | ---: | ---: |
| `pl.delay(10000)`，TensorCore 0 | 57956、21670、30863 | 20 | 11402、11400、11398 | 94 |
| `pl.delay(10000)`，TensorCore 1 | 103 | 20 | 11422、11420、11418 | 35 |
| `pl.delay(1000000)`，TensorCore 0 | 38205、20392、30946 | 20 | 1050900、1051104、1050996 | 94 |
| `pl.delay(1000000)`，TensorCore 1 | 103 | 20 | 1050920、1051124、1051016 | 35 |

三次运行的读数：

- **读数自身的开销恰好是 20 个周期。** 紧挨着的两次读数总是相差 20：前一次读数之后的两条 `sst`、两条 `sld`，加上后一次读数之前的两条 `sst` 和 `sfence`。任何区间的读数差减去 20，就是这段程序本身的周期数；`KernelClock.time_ops` 把这些步骤包在一起，直接返回程序和每条 HLO 指令扣除开销后的周期数。
- **kernel 是 11380 和 1050880 个周期左右**（扣除 20 之后），比 `vdelay` 多约 880 个周期，是两次小 DMA 和计算；各次运行之间相差几个到两百个周期，来自 DMA。
- **TensorCore 1 的 kernel 比 TensorCore 0 多 20 个周期。** 它跳过了 kernel 主体，但要在出口等 TensorCore 0 汇合（第二章第 1 节），而 TensorCore 0 在到达出口之前还要做一次读数。
- **TensorCore 0 在 kernel 之前有 2–6 万个周期的 runtime 代码**，每次运行都不一样；TensorCore 1 只有 103 个周期。比较两个程序时，用 TensorCore 1 的整个程序，或者只比较 kernel。

读数用了 `sfence`，所以每个读数都是“此前的向量工作全部发射之后”的时刻；kernel 结束处，输出 DMA 已经被 `vwait` 等到。要测 kernel 中的一段，把 `find_bundles` 找到的 bundle 编号交给 `instrument` 即可；插入点在循环中时，SMEM 中留下的是最后一次迭代的读数。

## LCC 的适用范围

LCC 数的是本芯片时钟的周期，每个 TensorCore 各有一个。由此有两条限制：

- **读数只在同一个 TensorCore 上可比。** 两次读数相减才有意义。不同 TensorCore 的 LCC 起点不同，即使在同一颗芯片上也不能直接相减（同一颗芯片的两个 TensorCore 共用一个时钟，起点之差是常数，第 6 节把它测出来之后才可以换算）；不同芯片各有自己的时钟，周期的长短还相差百万分之几。
- **周期数不是秒。** 换成秒要知道时钟的频率（约 1.05 GHz）。

所以 LCC 适合回答“这段代码在这个 TensorCore 上要多少周期”，这也是优化一个 kernel 时要问的问题。要比较不同 TensorCore、不同芯片上的事件，或者要一个以秒为单位的时间，用第 6 节的全局时钟 GTC；LCC 的频率也是在那里借助 GTC 和主机时钟标定的。

[tpu-v4-latency-numbers](../../../tpu-v4-latency-numbers/README.md) 用同样的载体与读法测量了第二章引用的全部 DMA 代价。[研究报告 48](../../../pallas-tpu-readings-dev/research_reports/48_scalar_cycle_lowering.md) 则走了另一条路：为读 LCC 增加一个 Pallas primitive 和它的 lowering，由编译器生成读数指令。两条路得到的读法相同。
