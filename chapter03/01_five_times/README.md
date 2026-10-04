# 五种时间

“这个 kernel 要多久”可以有五种不同的答案，取决于从哪里开始计、到哪里结束。它们相差可以超过一个数量级。本节用一个设备时间已知的 kernel，把五种时间并排测出来，说明每一种包含什么、能回答什么问题。

一个 TensorCore 上的时间，本章用周期数表示，并且直接在设备上读它的周期计数器 LCC 得到。芯片上的一切都按周期进行：清单中每个 bundle 的发射、`vdelay` 的停顿、DMA 的开销，原本都是周期计数。主机上测到的时间来自主机的时钟，用微秒。两者需要比较时，按周期计数器的频率约 1.05 GHz 换算（第 6 节）：1 µs 是 1050 个周期。周期数只在同一个 TensorCore 之内有意义；要比较不同 TensorCore、不同芯片上的事件时，改用全局时钟 GTC（第 6 节）。本节到第 5 节的实验都只涉及一颗芯片。

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

`vdelay` 让向量一侧停住指定的周期数（第 4 节）。所以这个 kernel 的设备时间是已知的：10500 或 1050000 个周期（10 µs 或 1 ms），再加两次小 DMA。实验只改 `delay_ns` 一处。

## 五种时间

| | 测法 | `pl.delay(10000)` | `pl.delay(1000000)` |
| --- | --- | ---: | ---: |
| 1 | Python 调用返回 | 76.6 µs | 83.3 µs |
| 2 | 调用并等到结果 | 158.9 µs | 1163.0 µs |
| 3 | 连续调用 64 次，后 30 次每次 | 94.5 µs | 1001.7 µs |
| 4 | LCC：TensorCore 0 的整个程序 | 30318 个周期 | 1070163 个周期 |
| 4 | LCC：TensorCore 1 的整个程序 | 11497 个周期 | 1050998 个周期 |
| 4 | LCC：kernel | 11379 个周期 | 1050880 个周期 |
| 5 | 清单：`vdelay` 的周期数 | 10500 | 1050000 |

**1. Python 调用返回的时间。** JAX 的调用是异步的：`compiled(x)` 把计算提交给 runtime 就返回，返回的数组是一个尚未完成的 future。这个时间约 77–83 µs，与 kernel 的长短无关，测到的是主机提交一次计算的开销。

**2. 调用并等到结果的时间。** 加上 `.block_until_ready()`，才等到设备完成：

```python
start = time.perf_counter()
result = compiled(x)
result.block_until_ready()
```

TensorCore 0 的整个程序 30318 个周期是 29 µs，1070163 个周期是 1019 µs，所以等到结果的时间比设备时间多出约 130–144 µs：提交、启动程序、完成通知回到主机。对 1 万个周期（约 10 µs）的 kernel，这部分开销是 kernel 本身的十几倍。用这种时间比较两个只差几千个周期的 kernel，差别会淹没在开销的波动中。第 7 节用 XProf 的主机事件把这段开销拆开。

**3. 连续调用的时间。** 不等结果，连续调用 64 次，每次的输入是上一次的输出：

```python
for _ in range(CALLS):
    start = time.perf_counter()
    result = compiled(result)
    calls.append(time.perf_counter() - start)
```

105 万个周期（1 ms）的 kernel 中，前 30 次调用每次只要约 58 µs，之后每次约 1002 µs。runtime 最多允许约 32 个计算同时在途；在途的计算达到上限后，每次新的调用要等一个旧的计算完成才能提交（[研究报告 51](../../../pallas-tpu-readings-dev/research_reports/51_tpu_async_dispatch_backpressure.md)）。所以背压之后的调用时间反映的是设备完成计算的速率，而不是这一次调用的延迟。1 万个周期的 kernel 中，主机提交一次就要约 95 µs，设备总在等主机，在途计算永远攒不满，每次调用的时间就是提交的时间。

**4. 设备上的周期数。** 在程序和 kernel 的起止位置各读一次设备上的周期计数器 LCC，相减：

```python
(_, module_start, module_end), (_, start, end) = tpuasm_tools.hlo_ops(compiled)
timed = clock.instrument(compiled, [module_start, start, end + 1, module_end + 1])
timed(x).block_until_ready()
readings = clock.read(4)
```

> 暂且可以理解为：TensorCore 有一个每周期加 1 的计数器 LCC，`KernelClock` 用 tpuasm 在已编译的程序中指定的位置插入读数的指令，程序运行之后再把读数取回。第 3 节详细介绍 LCC 的读法、读数怎样带出来，以及读数自身 20 个周期的开销（表中已经扣除）。

kernel 是 11379 与 1050880 个周期，比 `vdelay` 多 879 与 880 个周期，是两次 DMA 和计算。这是 kernel 本身的时间，精确到周期。

TensorCore 0 的整个程序比 kernel 长约 1.9 万个周期，这段时间在 kernel 之前，由 runtime 的代码占用，每次运行都不一样；TensorCore 1 的整个程序只比 kernel 多约 100 个周期。`num_cores=1` 时 TensorCore 1 跳过 kernel 主体（第一章第 1 节），但它在 kernel 出口等 TensorCore 0 汇合，所以它的 kernel 同样长。

**5. 从清单读出的时间。** 清单给出设备上要执行的指令，但 bundle 数不是周期数。每个周期最多标量发射一个 bundle，所以 bundle 数只是下限；向量指令还要经过硬件的计分板（scoreboard）：它记录每个寄存器、每个结果队列何时就绪，每个单元何时空闲，一条指令要等自己的源操作数就绪、要用的单元空闲才能向量发射，排在它后面的 bundle 一起等待（第 4 节）。周期数要按第 5 节的发射模型逐个 bundle 推算。这个时间不需要运行程序，但不包含 DMA 等异步工作的时间，必须由实测验证。

## 各自回答什么

| 时间 | 回答的问题 |
| --- | --- |
| Python 调用返回 | 主机提交是否成为瓶颈 |
| 调用并等到结果 | 调用方实际要等多久 |
| 连续调用 | 系统的吞吐：设备和主机谁是瓶颈 |
| 设备上的周期数（LCC） | kernel 或程序在设备上执行多久 |
| 清单与发射模型 | 不运行程序，推算一段指令要多少周期 |

优化 kernel 本身时，关心的是后两种。前三种包含了与 kernel 无关的主机开销：kernel 在一千个周期的量级时，主机时间几乎完全由开销决定。所以第二章的计时都没有用主机时钟，而是直接读设备上的周期计数器。下一节讨论还有哪些因素会让同一个 kernel 测出不同的时间。
