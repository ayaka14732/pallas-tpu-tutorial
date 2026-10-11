# 五种时间

“这个 kernel 要多久”可以有五种不同的答案，取决于从哪里开始计、到哪里结束。它们相差可以超过一个数量级。本节用一个设备时间已知的 kernel，把五种时间并排测出来，说明每一种包含什么、能回答什么问题。

一个 TensorCore 上的时间，本教程从第一章第 3 节起就用周期数表示，直接在设备上读它的周期计数器 LCC 得到。那里测的都是一段计算，这里把视野放大到整个程序和主机：调用方实际等待的时间里，设备上的周期数只占多少。主机上测到的时间来自主机的时钟，用微秒。两者需要比较时，按 1050 MHz 换算（第 5 节）：1 µs 是 1050 个周期。周期数只在同一个 TensorCore 之内有意义；要比较不同 TensorCore、不同芯片上的事件时，改用全局时钟 GTC（第 5 节）。本节到第 4 节的实验都只涉及一颗芯片。

## 一个设备时间已知的 kernel

本小节实验[源码](01_pallas_host_and_device_time.py)、[输出](01_pallas_host_and_device_time.txt)。

kernel 读入一个 `f32[8,128]`，用 `pl.delay` 让 TensorCore 停住指定的纳秒数，再加 1 写回：

```python
def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, sem: Ref) -> None:
    pltpu.async_copy(x_hbm, x_vmem, sem).wait()
    # 让 TensorCore 停住 delay_ns 纳秒。
    pl.delay(delay_ns)
    x_vmem[...] = x_vmem[...] + 1.0
    pltpu.async_copy(x_vmem, o_hbm, sem).wait()
```

`pl.delay` 在清单中是一条 `vdelay`，编译器按 1.05 GHz 把纳秒换算成周期：

```text
pl.delay(10000)：   { misc: vdelay 10500 }
pl.delay(1000000)： { s0: simm.s32 s10, 1050000 } ... { misc: vdelay s10 }
```

`vdelay` 让向量一侧停住指定的周期数（第 3 节）。所以这个 kernel 的设备时间是已知的：10500 或 1050000 个周期（10 µs 或 1 ms），再加两次小 DMA。实验只改 `delay_ns` 一处。

## 五种时间

| | 测法 | `pl.delay(10000)` | `pl.delay(1000000)` |
| --- | --- | ---: | ---: |
| 1 | Python 调用返回 | 73.7 µs | 83.6 µs |
| 2 | 调用并等到结果 | 171.1 µs | 1166.9 µs |
| 3 | 连续调用 64 次，后 30 次每次 | 80.8 µs | 999.2 µs |
| 4 | LCC：TensorCore 0 的整个程序 | 42702 个周期 | 1082970 个周期 |
| 4 | LCC：TensorCore 1 的整个程序 | 11500 个周期 | 1050996 个周期 |
| 4 | LCC：kernel | 11382 个周期 | 1050878 个周期 |
| 5 | 清单：`vdelay` 的周期数 | 10500 | 1050000 |

**1. Python 调用返回的时间。** JAX 的调用是异步的：`compiled(x)` 把计算提交给 runtime 就返回，返回的数组是一个尚未完成的 future。这个时间约 74–84 µs，与 kernel 的长短无关，测到的是主机提交一次计算的开销。

**2. 调用并等到结果的时间。** 加上 `.block_until_ready()`，才等到设备完成：

```python
start = time.perf_counter()
result = compiled(x)
result.block_until_ready()
```

TensorCore 0 的整个程序 42702 个周期是 41 µs，1082970 个周期是 1031 µs，所以等到结果的时间比设备时间多出约 130–136 µs：提交、启动程序、完成通知回到主机。对 1 万个周期（约 10 µs）的 kernel，这部分开销是 kernel 本身的十几倍。用这种时间比较两个只差几千个周期的 kernel，差别会淹没在开销的波动中。第 6 节用 XProf 的主机事件把这段开销拆开。

**3. 连续调用的时间。** 不等结果，连续调用 64 次，每次的输入是上一次的输出：

```python
for _ in range(CALLS):
    start = time.perf_counter()
    result = compiled(result)
    calls.append(time.perf_counter() - start)
```

105 万个周期（1 ms）的 kernel 中，前 30 次调用每次只要约 57 µs，之后每次约 999 µs。runtime 最多允许约 32 个计算同时在途；在途的计算达到上限后，每次新的调用要等一个旧的计算完成才能提交（[研究报告 51](../../../pallas-tpu-readings-dev/research_reports/51_tpu_async_dispatch_backpressure.md)）。所以背压之后的调用时间反映的是设备完成计算的速率，而不是这一次调用的延迟。1 万个周期的 kernel 中，主机提交一次就要约 81 µs，设备总在等主机，在途计算永远攒不满，每次调用的时间就是提交的时间。

**4. 设备上的周期数。** 在程序和 kernel 的起止位置各读一次设备上的周期计数器 LCC，相减：

```python
(_, module_start, module_end), (_, start, end) = tpuasm_tools.hlo_ops(compiled)
timed = clock.instrument(compiled, [module_start, start, end + 1, module_end + 1])
timed(x).block_until_ready()
readings = clock.read(4)
```

`KernelClock` 就是第一章第 3 节的那个工具；程序和 kernel 的起止位置怎样找到，见下一小节。表中已经扣除每次读数 20 个周期的开销。

kernel 是 11382 与 1050878 个周期，比 `vdelay` 多 882 与 878 个周期，是两次 DMA 和计算。这是 kernel 本身的时间，精确到周期。

TensorCore 0 的整个程序比 kernel 长约 3.1 万个周期，这段时间在 kernel 之前，由 runtime 的代码占用，每次运行都不一样；TensorCore 1 的整个程序只比 kernel 多约 100 个周期。`num_cores=1` 时 TensorCore 1 跳过 kernel 主体（第一章第 1 节），但它在 kernel 出口等 TensorCore 0 汇合，所以它的 kernel 同样长。

**5. 从清单读出的时间。** 清单给出设备上要执行的指令，但 bundle 数不是周期数。每个周期最多标量发射一个 bundle，所以 bundle 数只是下限；向量指令还要经过硬件的计分板（scoreboard）：它记录每个寄存器、每个结果队列何时就绪，每个单元何时空闲，一条指令要等自己的源操作数就绪、要用的单元空闲才能向量发射，排在它后面的 bundle 一起等待（第 3 节）。周期数要按第 4 节的发射模型逐个 bundle 推算。这个时间不需要运行程序，但不包含 DMA 等异步工作的时间，必须由实测验证。

## 在哪里读：程序与 kernel 的起止标记

本小节实验[源码](02_tpuasm_kernel_clock.py)、[输出](02_tpuasm_kernel_clock.txt)。

`KernelClock` 的读数先留在 SMEM 中，程序结束后再由另一个程序取回（第一章第 3 节）。这依赖 SMEM 的内容在程序之间保留。实验让程序 A 把一个数写进 SMEM 的某个地址，再让另一个程序 B 去读：

```text
程序 A 写入 0 之后，程序 B 读到：[0, 0]（两个 TensorCore）
程序 A 写入 20261004 之后，程序 B 读到：[20261004, 20261004]
```

读数存放的地址 `0x20000` 在 SMEM 的中部，远离 kernel 的 scratch（从低地址分配）和 runtime 使用的最高一段。

第一章第 2 节的清单中，kernel 段以 `vtrace 0x80000000` 开始、以 `vtrace 0x90000000` 结束。这是编译器为每条 HLO 指令放置的起止标记，整个程序也有一对，序号是 `0xfffffff`（第 6 节解释它们的编码）。`tpuasm_tools.hlo_ops` 把它们找出来：

```text
[('module', 460, 550), ('wait.1', 503, 535)]
```

名字取自清单中的注释（`# entry bundle: %名称 = ...`、`# function ... [symbol N, 名称]`）；没有注释的指令按标记中的序号到编译后的 HLO 里去数。get-tuple-element、bitcast 这类不生成代码的 HLO 指令没有标记，也不占序号。`KernelClock.time_ops` 在每一对标记处各插一次读数，直接返回程序和每条 HLO 指令扣除开销后的周期数。

实验在上面那个 kernel 的这四个位置读数；kernel 的起点和终点各连读两次，看读数自身占多少周期：

```python
timed = clock.instrument(compiled, [module_start, start, start, end + 1, end + 1, module_end + 1])
```

| | 程序开始 → kernel 开始 | 相邻两次读数 | kernel | kernel 结束 → 程序结束 |
| --- | ---: | ---: | ---: | ---: |
| `pl.delay(10000)`，TensorCore 0 | 48565、21395、19461 | 20 | 11401、11399、11396 | 94 |
| `pl.delay(10000)`，TensorCore 1 | 103 | 20 | 11421、11419、11416 | 35 |
| `pl.delay(1000000)`，TensorCore 0 | 50493、18912、17525 | 20 | 1050900、1050900、1050899 | 94 |
| `pl.delay(1000000)`，TensorCore 1 | 103 | 20 | 1050920、1050920、1050919 | 35 |

三次运行的读数：

- **读数自身的开销恰好是 20 个周期。** 紧挨着的两次读数总是相差 20：前一次读数之后的两条 `sst`、两条 `sld`，加上后一次读数之前的两条 `sst` 和 `sfence`。“读数差减去 20”的约定由此而来；它带有的固定偏移（n 个互不等待的 bundle 读作 n − 2）在第 3 节用实验说明。
- **kernel 是 11380 和 1050880 个周期左右**（扣除 20 之后），比 `vdelay` 多约 880 个周期，是两次小 DMA 和计算；各次运行之间通常只差几个周期，偶尔相差一两百个，来自 DMA。
- **TensorCore 1 的 kernel 比 TensorCore 0 多 20 个周期。** 它跳过了 kernel 主体，但要在出口等 TensorCore 0 汇合（第二章第 1 节），而 TensorCore 0 在到达出口之前还要做一次读数。
- **TensorCore 0 在 kernel 之前有 2–5 万个周期的 runtime 代码**，每次运行都不一样；TensorCore 1 只有 103 个周期。比较两个完整的程序时，用 TensorCore 1 的整个程序，或者只比较 kernel。

读数用了 `sfence`，所以每个读数都是“此前的向量工作全部发射之后”的时刻；kernel 结束处，输出 DMA 已经被 `vwait` 等到。插入点在循环中时，SMEM 中留下的是最后一次迭代的读数。

## 各自回答什么

| 时间 | 回答的问题 |
| --- | --- |
| Python 调用返回 | 主机提交是否成为瓶颈 |
| 调用并等到结果 | 调用方实际要等多久 |
| 连续调用 | 系统的吞吐：设备和主机谁是瓶颈 |
| 设备上的周期数（LCC） | kernel 或程序在设备上执行多久 |
| 清单与发射模型 | 不运行程序，推算一段指令要多少周期 |

优化 kernel 本身时，关心的是后两种。前三种包含了与 kernel 无关的主机开销：kernel 在一千个周期的量级时，主机时间几乎完全由开销决定。所以前两章的计时都没有用主机时钟，而是直接读设备上的周期计数器。下一节讨论还有哪些因素会让同一个 kernel 测出不同的时间。
