# XProf 与 vtrace

本教程计时首选 LCC：读数就是周期，可以放在程序的任何位置（第 4 节）。XProf 是 JAX 自带的 profiler，不改程序就能看到每个程序、每条 HLO 指令在设备上的起止，还能看到主机一侧的事件。本节简要介绍它：采集结果怎样解析，设备上的事件来自哪条指令，怎样自己写这条指令，以及哪些信息只有它能给出。之后各章只在需要主机事件时才用 XProf。

## 采集与解析

`jax.profiler.trace(目录)` 包住要观察的调用，结束时在目录下写出一个 `.xplane.pb` 文件。[`xprof_tools.py`](../../xprof_tools.py) 直接解析这个文件：

```python
events = xprof_tools.capture(lambda: [compiled(x).block_until_ready() for _ in range(16)], path)
```

XPlane 是一个 protobuf，结构只有三层：

```text
XSpace
└─ XPlane：/device:TPU:0、/device:TPU:1（每个 TensorCore 一个）、/host:CPU ……
   ├─ event_metadata：编号 → 事件名
   ├─ stat_metadata： 编号 → 属性名
   └─ XLine：设备上是轨道（XLA Modules、XLA Ops、XLA TraceMe），主机上是线程
      └─ XEvent：metadata_id、offset_ps、duration_ps，以及若干属性
```

事件本身只存编号，名字要到所在 plane 的 `event_metadata` 中查；属性的名字同样要到 `stat_metadata` 中查。`xprof_tools.read_xplane` 用不到一百行完成这件事：一个最小的 protobuf 解码器按字段编号读出上面这些消息，再把每个事件展开成一个字典（plane、line、name、起始时刻、持续时间、属性）。设备事件有一个属性 `device_duration_ps`，是由设备上的计数直接换算的持续时间；本节按约 1.05 GHz 把它换回周期数（`xprof_tools.device_cycles`）。

设备上的三条轨道：

| 轨道 | 一个事件是 |
| --- | --- |
| `XLA Modules` | 一次程序的执行 |
| `XLA Ops` | 程序中一条 HLO 指令的执行，例如一个 Pallas kernel、一个 fusion |
| `XLA TraceMe` | kernel 内部标出的一个区域 |

## 事件来自 vtrace

设备上的每个事件都由程序中的一对 `vtrace` 指令产生。第一章第 2 节的清单中，kernel 段的第一条和最后一条指令是：

```text
{ misc: vtrace 0x80000000 }
...
{ misc: vtrace 0x90000000 }
```

`vtrace` 在 `misc` 槽中，执行时让 TensorCore 把一个 32 位的操作数连同当时的时间写进 trace 缓冲区；profiler 开启时，主机读回这些记录，按操作数配对成区间，再换算成主机时间。硬件不理解操作数的含义，含义是编译器与主机软件的约定：

| 操作数的高 4 位 | 低 28 位 | 配成的事件 |
| --- | --- | --- |
| `0x8` 开始、`0x9` 结束 | HLO 指令的序号；`0xfffffff` 表示整个程序 | `XLA Ops`；`0xfffffff` 是 `XLA Modules` |
| `0xb` 开始、`0xc` 结束 | 区域的编号 | `XLA TraceMe` |

第 4 节的 `tpuasm_tools.hlo_ops` 找的就是 `0x8`、`0x9` 这两种标记。

## 公开接口：named_scope 标出区域

本小节实验[源码](01_pallas_trace_regions.py)、[输出](01_pallas_trace_regions.txt)。

Pallas 把 `jax.named_scope` 的边界降低为 `0xb`、`0xc` 的一对 `vtrace`，但默认不生成指令；要用编译选项 `xla_enable_custom_call_region_trace` 打开：

```python
compiled = tpuasm_tools.compile(affine, x, mesh=mesh, compiler_options={'xla_enable_custom_call_region_trace': flag})
```

kernel 读入 `f32[1024,128]`（512 KiB），计算 `2x + 1` 写回，用 `named_scope` 标出四个区域：

```python
load = pltpu.make_async_copy(x_hbm, x_vmem, sem)
with jax.named_scope('load_start'):
    load.start()
with jax.named_scope('load_wait'):
    load.wait()
with jax.named_scope('compute'):
    x_vmem[...] = x_vmem[...] * 2.0 + 1.0
with jax.named_scope('store'):
    pltpu.async_copy(x_vmem, o_hbm, sem).wait()
```

实验只改编译选项。关闭时 kernel 段 159 个 bundle，只有开头和结尾两条 `vtrace`；打开时 168 个 bundle，10 条 `vtrace`（连续 6 个以上不含 `vtrace` 的 bundle 折叠成一行）：

```text
[0] { misc: vtrace 0x80000000 }
[1] { s1: sld s6, [smem:0x1] }
[2] { s0: sne.s32 p0, s6, 0 }
[3] { s0: @p0 sbr.rel L_0293 }
[4] { misc: vtrace 0xb8000000 }
[5] { s0: simm.s32 s7, 0 }
[6] { s0: dma.simple [vmem:s7], [hbm:s0], length=1024, dst_flag=[sflag:52] }
[7] { misc: vtrace 0xc8000000 }
[8] { misc: vtrace 0xb8000001 }
[9] { misc: vwait.ge [sflag:52], 1024 }
[10] { misc: vsyncadd.s32 [sflag:52], -1024 }
[11] { misc: vtrace 0xc8000001 }
[12] { misc: vtrace 0xb8000002 }
[13] { vld: vld.8x128 v0, [vmem:0x0] }
[14] { va0: vmul.8x128.f32 v1, 2.0, v0 ; vld: vld.8x128 v2, [vmem:0x8] }
… 129 个 bundle
[144] { vst: vst.8x128 [vmem:0x3f0], v30 }
[145] { vst: vst.8x128 [vmem:0x3f8], v31 }
[146] { misc: vtrace 0xc8000002 }
[147] { misc: vtrace 0xb8000003 }
[148] { s0: dma.simple [hbm:s1], [vmem:s7], length=1024, dst_flag=[sflag:52] }
[149] { misc: vwait.ge [sflag:52], 1024 }
[150] { misc: vsyncadd.s32 [sflag:52], -1024 }
[151] { misc: vtrace 0xc8000003 }
...
[167] { misc: vtrace 0x90000000 }
```

每个 `named_scope` 变成一对 `vtrace`，低位的 0–3 是区域的编号，开始与结束靠编号配对。区域的名字不在指令中：编译器把编号与名字的对应关系存在程序的 metadata 里，主机解析时再查回来（[研究报告 35](../../../pallas-tpu-readings-dev/research_reports/35_pallas_vtrace_to_xplane.md)）。位 27 的 `0x08000000` 来自编译器分配编号的范围，不是类型的一部分。嵌套的 `named_scope` 得到嵌套的区域，外层的结束标记排在内层之后。

`XLA TraceMe` 轨道上的四个区域，16 次调用的中位数：

| 区域 | 周期数 | 对照 |
| --- | ---: | --- |
| `load_start` | 3 | — |
| `load_wait` | 1033 | HBM → TC VMEM 512 KiB：`483.6 + 1.105 × 512` ≈ 1049 个周期 |
| `compute` | 134 | 区域内 133 个 bundle |
| `store` | 946 | TC VMEM → HBM 512 KiB：`417.5 + 1.051 × 512` ≈ 956 个周期 |

这种做法有代价：每条 `vtrace` 独占一个 bundle，区域的边界还限制了编译器跨边界重排指令（[研究报告 34](../../../pallas-tpu-readings-dev/research_reports/34_named_scope_changes_tpu_scheduling.md)）。带 region trace 测到的，是一个与正式运行不同的程序。

## 手写 vtrace

本小节实验[源码](02_tpuasm_hand_vtrace.py)、[输出](02_tpuasm_hand_vtrace.txt)。

既然事件只取决于 `vtrace` 的操作数，就不必经过 `named_scope`：用 tpuasm 在已编译的程序中直接插入一对 `vtrace`，编译器的调度完全不变。载体是上一个实验的 kernel，关闭 region trace；计算部分是 bundle 511–643，共 133 个：

```python
first = tpuasm_tools.find_bundles(serialized, 'vld.8x128')[0]
last = tpuasm_tools.find_bundles(serialized, 'vst.8x128')[-1]
patched = tpuasm_tools.insert_bundles(serialized, {first: bundle(f'misc: vtrace {start}'), last + 1: bundle(f'misc: vtrace {stop}')})
```

实验只改两个操作数：

| 插入的一对 `vtrace` | 多出的事件 |
| --- | --- |
| `0xb8000000`、`0xc8000000` | `XLA TraceMe` 轨道，名字 `$$unknown$$`，134 个周期 |
| `0xb0000005`、`0xc0000005` | `XLA TraceMe` 轨道，名字 `overlayer-overhead`，134 个周期 |
| `0x80000007`、`0x90000007` | `XLA Ops` 轨道，名字 `region.7`，134 个周期 |

三种都得到了 134 个周期的事件，与 `named_scope` 的 `compute` 相同。区别只在名字：主机用编号到程序的 metadata 中查名字，手写的编号查不到时显示 `$$unknown$$`，碰巧与 runtime 自己的编号相同时显示那个编号的名字；写成 `0x8`、`0x9` 类型时，它被当作一条 HLO 指令，名字是 `region.` 加编号。需要多个区域时，用不同的编号区分。

在同样的两个位置换成第 4 节的 LCC 读数：

```text
同样的位置换成 LCC 读数：两次读数相差 151 个周期，其中 20 个是读数自身的开销
```

扣除开销后是 131 个周期，与 XProf 的 134 相差 3 个周期。两者测的时刻略有不同：`vtrace` 记录的是它自己向量发射的时刻；LCC 读数前面有一条 `sfence`，读到的是此前的 bundle 全部向量发射之后、标量一侧的时刻（第 5 节）。

## 逐指令的事件是插值出来的

本小节实验[源码](04_pallas_instruction_events.py)、[输出](04_pallas_instruction_events.txt)。

XProf 还能显示每条指令的事件。打开编译选项 `xla_xprof_enable_custom_call_tracing`，TensorCore 的轨道多出 `VALU Instructions`、`VLD Instructions`、`VST Instructions`、`SALU Instructions` 和 `Pallas Primitives`，每个事件带有指令名和它所在的 bundle 编号（编译器内部的编号，与清单中的位置不同）。这些事件看起来像是每条指令各有一个时间戳，其实不是。

清单中多出的只有几条 `0xa` 类型的 `vtrace`（`named_scope` 的区域标记也一并打开了）：

```text
(1, '0xa0000000')、(153, '0xa000018f')、(161, '0xa0000199')
```

`0xa` 类型的操作数低位是 bundle 编号：0、399、409。编译器的配置是每 10 个 bundle 放一个这样的标记（[研究报告 38](../../../pallas-tpu-readings-dev/research_reports/38_sparse_bundle_trace_missing_dma.md)），放不进去的地方就没有。硬件只在这些标记和其他 `vtrace` 处留下时间；两个标记之间的指令事件，是主机把这段时间按 bundle 编号均匀分开得到的（[研究报告 37](../../../pallas-tpu-readings-dev/research_reports/37_loop_primitive_relations_patch_design.md)）。一次执行中，各 bundle 的事件与上一个 bundle 的起始时刻之差：

```text
bundle 编号  指令            与上一个 bundle 相差的周期数
 10         vtrace          3.0
 11         vtrace          103.9
 12         dma.done.wait   104.0
 13         vsyncadd        104.0
 ...
 19         vadd.f32        104.0
 20         vadd.f32        104.0
 21         vadd.f32        1.0
 22         vadd.f32        1.0
 ...
```

编号 10 到 20 的 bundle 每个相隔恰好 104 个周期。这一段真实的情况是：等待输入 DMA 的 `dma.done.wait` 一条指令占了约 1040 个周期，其余指令各占一个周期。主机只知道这 10 个 bundle 一共用了 1040 个周期，就给每个 bundle 分了 104 个。从编号 21 起，相邻的标记之间只有计算，每个 bundle 相隔 1 个周期，插值与实际一致。

所以逐指令的事件只能用来看“执行到了哪一段”，不能当作每条指令的耗时：等待集中在哪一条指令上，从这些事件中看不出来。两个标记的时间差太小时，主机还会跳过它们之间的全部指令事件（研究报告 38）。要知道一条指令或一小段指令的周期数，用第 4 节的 LCC。

## 区间测的是什么

`load_start` 只有 3 个周期，而输入 DMA 要一千多个周期：区域只包含发起 DMA 的那条指令，DMA 本身在区域结束之后才在后台进行。等待它的 `load_wait` 才包含了 DMA 的时间。区域的边界是 `vtrace` 指令执行的时刻，不是区域中工作完成的时刻。

`vtrace` 在 `misc` 槽，属于向量一侧（第 5 节），在向量发射时记录。由此可以判断一个区间包含什么：

- 结束标记排在一条 `vwait` 之后时，`vwait` 等到 DMA 完成才向量发射，结束标记更晚，所以区间包含这次等待。`load_wait` 和 `store` 都是这样。
- 区间只包含发起而不包含等待时，测到的只是发起。异步的 DMA、MXU 计算都要到对应的等待或取回之后才算完成（[研究报告 39](../../../pallas-tpu-readings-dev/research_reports/39_vtrace_completion_dependencies.md)、[40](../../../pallas-tpu-readings-dev/research_reports/40_vtrace_vector_issue_and_completion.md)）。
- 标量一侧的工作可以越过开始标记提前执行：开始标记被前面的向量等待拖住时，标量指令已经走到了前面，区间会少算这部分工作（[研究报告 47](../../../pallas-tpu-readings-dev/research_reports/47_vtrace_vif_timing.md)）。

另外两点限制了它的精度。记录的时间来自 GTC，第 7 节会看到 GTC 并不是每个周期均匀增加，单个区间有一两个周期的误差；在打点较稀疏的采集模式下，profiler 还会因为内部检查跳过一部分指令事件（[研究报告 38](../../../pallas-tpu-readings-dev/research_reports/38_sparse_bundle_trace_missing_dma.md)）。

## 只有 XProf 能给出的：主机一侧的事件

本小节实验[源码](03_jax_host_events.py)、[输出](03_jax_host_events.txt)。

第 1 节看到，调用一个 kernel 并等到结果，比设备上的时间多出 100 多微秒。这段时间花在主机上，LCC 看不到；XPlane 的 `/host:CPU` 中有主机一侧的事件，可以把一次调用拆开。实验对 `pl.delay(100000)` 的 kernel 连续调用 8 次，每次用 `jax.profiler.TraceAnnotation('call')` 标出整个调用，统计调用区间内各个主机事件的开始时刻与持续时间：

```python
with jax.profiler.TraceAnnotation('call'):
    compiled(x).block_until_ready()
```

8 次调用的中位数（开始时刻相对调用开始，单位 µs）：

| 事件 | 开始 | 持续 | 含义 |
| --- | ---: | ---: | --- |
| `PjitFunction(jit(wait))` | 2.3 | 84.0 | Python 一侧的 `jit` 调用（嵌套出现两次） |
| `PjRtCApiLoadedExecutable::Execute` | 12.0 | 69.2 | 交给 runtime 执行 |
| `CommonPjRtLoadedExecutable::ExecutePrepare` | 17.9 | 11.9 | 准备参数，分配输出 buffer |
| `TpuLoadedExecutable::ExecuteLaunch` | 30.8 | 44.2 | 把程序放进设备的执行队列 |
| `tpu::System::Execute`（`core_id` 0、1） | 32.2、59.3 | 25.2、14.1 | 其中每个 TensorCore 一次入队 |
| `ReadSyncFlag`（两次） | 210.1、212.3 | 26.7、27.6 | 读取设备的完成标志 |
| `CompleteCallbacks`（两次） | 237.9、240.2 | 17.6、29.0 | 完成后的回调 |
| `tpu::System::Execute=>Done`（`core_id` 1、0） | 249.2、256.4 | 3.5、12.5 | 标记每个 TensorCore 执行结束 |
| 整个调用 | 0 | 276.8 | |

按时间顺序，一次调用分成三段：

- **提交，约 75 µs。** 从调用开始到 `ExecuteLaunch` 结束（30.8 + 44.2），主机在 Python、参数处理、输出分配和入队上花掉的时间。两个 TensorCore 各入队一次。
- **等待设备，约 135 µs。** 从入队结束到主机开始读完成标志（210.1）。设备上的 module 124976 个周期（119 µs）、kernel 105875 个周期（101 µs）都在这一段之内，其余是程序启动与结束、完成标志传回主机的时间。
- **完成处理，约 67 µs。** 读完成标志、执行回调、标记结束，直到 `block_until_ready()` 返回（276.8）。

所以 1 万个周期的 kernel 调用并等到结果要 160 µs 左右，不是因为 kernel 慢：提交与完成处理这两段在主机上就占了约 140 µs，与 kernel 的长短无关。要缩短调用方的等待，只能减少调用次数，例如把多步工作合进一个程序、在 kernel 内部循环，或者让多个调用在途重叠。

## 什么时候用哪一种

| 要回答的问题 | 方法 |
| --- | --- |
| 一个 kernel 或其中一段要多少周期 | LCC（第 4 节的 `KernelClock`） |
| 一段手写的指令序列要多少周期 | LCC（第 4 节的 `LccProbe`） |
| 程序中有哪些 HLO 指令、各自大致多久，不想改程序 | XProf 的 `XLA Ops` |
| 主机在一次调用中做了什么 | XProf 的主机事件 |
