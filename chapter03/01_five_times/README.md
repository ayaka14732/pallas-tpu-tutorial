# 五种时间

“这个 kernel 要多久”可以有五种不同的答案，取决于从哪里开始计、到哪里结束。它们相差可以超过一个数量级。本节用一个设备时间已知的 kernel，把五种时间并排测出来，说明每一种包含什么、能回答什么问题。

本章的时间一律用 TensorCore 的周期数表示：一个周期约 0.95 ns，1 µs 是 1050 个周期（周期计数器约 1.05 GHz，第 7 节）。清单中的指令、DMA 的代价、发射模型都以周期为单位，用同一个单位才能把各个层次的数字直接相比。主机时钟和 XProf 给出的是秒，实验脚本按 1.05 GHz 换算；主机时间同时附上原始的微秒数。

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

`vdelay` 让向量一侧停住指定的周期数（第 5 节）。所以这个 kernel 的设备时间是已知的：10500 或 1050000 个周期，再加两次小 DMA。实验只改 `delay_ns` 一处。

## 五种时间

| | 测法 | `pl.delay(10000)` | `pl.delay(1000000)` |
| --- | --- | ---: | ---: |
| 1 | Python 调用返回（主机时钟） | 77076（73.4 µs） | 82829（78.9 µs） |
| 2 | 调用并等到结果（主机时钟） | 167218（159.3 µs） | 1218037（1160.0 µs） |
| 3 | 连续调用 64 次，后 30 次每次（主机时钟） | 81065（77.2 µs） | 1050904（1000.9 µs） |
| 4 | XProf：TensorCore 0 的 module | 28933 | 1070276 |
| 4 | XProf：kernel（op） | 11373 | 1050872 |
| 5 | 清单：`vdelay` 的周期数 | 10500 | 1050000 |

单位都是周期。

**1. Python 调用返回的时间。** JAX 的调用是异步的：`compiled(x)` 把计算提交给 runtime 就返回，返回的数组是一个尚未完成的 future。这个时间约 8 万个周期（73–79 µs），与 kernel 的长短无关，测到的是主机提交一次计算的开销。

**2. 调用并等到结果的时间。** 加上 `.block_until_ready()`，才等到设备完成：

```python
start = time.perf_counter()
result = compiled(x)
result.block_until_ready()
```

它比 module 的设备时间多出约 14–15 万个周期：提交、启动程序、完成通知回到主机。对 1 万个周期的 kernel，这部分开销是 kernel 本身的十几倍。用这种时间比较两个只差几千个周期的 kernel，差别会淹没在开销的波动中。

**3. 连续调用的时间。** 不等结果，连续调用 64 次，每次的输入是上一次的输出：

```python
for _ in range(CALLS):
    start = time.perf_counter()
    result = compiled(result)
    calls.append(time.perf_counter() - start)
```

105 万个周期的 kernel 中，前 30 次调用每次只要约 5.8 万个周期，之后每次约 105 万个周期。runtime 最多允许约 32 个计算同时在途；在途的计算达到上限后，每次新的调用要等一个旧的计算完成才能提交（[研究报告 51](../../../pallas-tpu-readings-dev/research_reports/51_tpu_async_dispatch_backpressure.md)）。所以背压之后的调用时间反映的是设备完成计算的速率，而不是这一次调用的延迟。1 万个周期的 kernel 中，主机提交一次就要约 8 万个周期，设备总在等主机，在途计算永远攒不满，每次调用的时间就是提交的时间。

**4. XProf 中的设备时间。** 用 `jax.profiler.trace` 采集，读 trace 中设备上的事件（[`xprof_tools.py`](../../xprof_tools.py)）：

```python
events = xprof_tools.device_events(xprof_tools.capture(lambda: [compiled(x).block_until_ready() for _ in range(8)], path))
```

每个 TensorCore 一条轨道（`/device:TPU:0`、`/device:TPU:1`），其中 `XLA Modules` 是整个程序，`XLA Ops` 是程序中的每个 HLO 指令，这里只有 kernel 一个。kernel 的时间是 11373 与 1050872 个周期，比 `vdelay` 多 873 与 872 个周期，是两次 DMA 和计算。这是最接近“kernel 本身”的数。

TensorCore 0 的 module 比 kernel 长约 1.8–1.9 万个周期，这段时间在 kernel 之外，由 runtime 的代码占用；TensorCore 1 的 module 与 kernel 几乎相同。`num_cores=1` 时 TensorCore 1 跳过 kernel 主体（第一章第 1 节），但它在 kernel 出口等 TensorCore 0 汇合，所以它的 kernel 事件同样长。

> 暂且可以理解为：XProf 的事件由程序中的 `vtrace` 指令标出起止，第一章第 2 节清单中开头的 `vtrace 0x80000000` 和结尾的 `vtrace 0x90000000` 就是 kernel 事件的两端。第 3 节详细介绍这些记录怎样变成 XProf 中的区间。

**5. 从清单读出的时间。** 清单给出设备上要执行的指令，但 bundle 数不是周期数。每个周期最多标量发射一个 bundle，所以 bundle 数只是下限；向量指令还要经过硬件的计分板（scoreboard）：它记录每个寄存器、每个结果队列何时就绪，每个单元何时空闲，一条指令要等自己的源操作数就绪、要用的单元空闲才能向量发射，排在它后面的 bundle 一起等待（第 5 节）。周期数要按第 6 节的发射模型逐个 bundle 推算。这个时间不需要运行程序，但不包含 DMA 等异步工作的时间，必须由实测验证。

## 等到结果的时间花在哪里

本小节实验[源码](02_jax_host_events.py)、[输出](02_jax_host_events.txt)。

第 2 种时间比设备时间多出十几万个周期。XProf 同时记录了主机一侧的事件，可以把一次调用拆开。实验对 `pl.delay(100000)` 的 kernel 连续调用 8 次，每次用 `jax.profiler.TraceAnnotation('call')` 标出整个调用，统计调用区间内各个主机事件的开始时刻与持续时间：

```python
with jax.profiler.TraceAnnotation('call'):
    compiled(x).block_until_ready()
```

8 次调用的中位数（开始时刻相对调用开始，单位为周期，由主机时钟换算）：

| 事件 | 开始 | 持续 | 含义 |
| --- | ---: | ---: | --- |
| `PjitFunction(jit(wait))` | 2284 | 80462 | Python 一侧的 `jit` 调用（嵌套出现两次） |
| `PjRtCApiLoadedExecutable::Execute` | 11198 | 66297 | 交给 runtime 执行 |
| `CommonPjRtLoadedExecutable::ExecutePrepare` | 16217 | 10747 | 准备参数，分配输出 buffer |
| `TpuLoadedExecutable::ExecuteLaunch` | 27379 | 44142 | 把程序放进设备的执行队列 |
| `tpu::System::Execute`（两次） | 28696、54085 | 23494、15708 | 其中的两次入队 |
| `ReadSyncFlag`（两次） | 211822、212179 | 27794、28308 | 读取设备的完成标志 |
| `CompleteCallbacks`（两次） | 239547、240361 | 17052、20202 | 完成后的回调 |
| `tpu::System::Execute=>Done`（两次） | 250661、254767 | 5066、5014 | 标记执行结束 |
| 整个调用 | 0 | 265424 | |

`tpu::System::Execute` 等事件每次调用出现两次，与一颗芯片上 TensorCore 的个数相同；本节没有进一步确认两次各对应什么。按时间顺序，一次调用分成三段：

- **提交，约 7.2 万个周期。** 从调用开始到 `ExecuteLaunch` 结束（27379 + 44142），主机在 Python、参数处理、输出分配和入队上花掉的时间。第 1 种时间（只等调用返回）主要就是这一段。
- **等待设备，约 14 万个周期。** 从入队结束到主机开始读完成标志（211822）。设备上的 module 124954 个周期、kernel 105873 个周期都在这一段之内，其余是程序启动与结束、完成标志传回主机的时间。
- **完成处理，约 5.4 万个周期。** 读完成标志、执行回调、标记结束，直到 `block_until_ready()` 返回（265424）。

所以 1 万个周期的 kernel 调用并等到结果要 16.7 万个周期，不是因为 kernel 慢：提交与完成处理这两段在主机上就占了约 12.6 万个周期，与 kernel 的长短无关。要缩短调用方的等待，只能减少调用次数，例如把多步工作合进一个程序、在 kernel 内部循环，或者让多个调用在途重叠（第 3 种时间）。

## 各自回答什么

| 时间 | 回答的问题 |
| --- | --- |
| Python 调用返回 | 主机提交是否成为瓶颈 |
| 调用并等到结果 | 调用方实际要等多久 |
| 连续调用 | 系统的吞吐：设备和主机谁是瓶颈 |
| XProf 设备时间 | kernel 或程序在设备上执行多久 |
| 清单与 LCC | kernel 内部的某一段花了多少周期 |

优化 kernel 本身时，关心的是后两种。前三种包含了与 kernel 无关的主机开销：kernel 在一千个周期的量级时，主机时间几乎完全由开销决定。第二章第 2 节的“在 kernel 内部重复 4 遍和 8 遍，取主机时间之差”就是为了从第二种时间中消去这部分开销。下一节讨论还有哪些因素会让同一个 kernel 测出不同的时间。
