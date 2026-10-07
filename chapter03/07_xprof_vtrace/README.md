# XProf 与 vtrace

XProf 是 JAX 自带的 profiler。不改程序，它就能给出每个程序、每条 HLO 指令在设备上的起止，以及主机一侧的事件。前面几节没有用它，是因为它的设备时间并不是周期计数：**设备上的每个事件由程序中的一对 `vtrace` 指令产生，`vtrace` 记录的时间是上一节的 GTC**。所以 XProf 给出的是 GTC 时间轴上的纳秒数，分辨率是 GTC 高位的一步（1.5 个周期），而不是 LCC 那样精确到 1 个周期的计数。

本节按这条线索介绍 XProf：采集结果怎样解析，事件来自哪条指令，怎样用公开接口和手写指令标出自己的区域，这些事件的时间怎样由 GTC 换算而来，逐指令的事件为什么不能当真，一个区间到底包含了什么；最后是只有 XProf 能给出的信息：主机一侧的事件。之后各章只在 LCC 做不到时才用 XProf。

本节的实验都在只打开一颗芯片的会话中运行。上一节说明了这种会话中芯片自己就是时间源，GTC 与本地周期的比例恰好是 32 : 3。

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

事件本身只存编号，名字要到所在 plane 的 `event_metadata` 中查；属性的名字同样要到 `stat_metadata` 中查。`xprof_tools.read_xplane` 用不到一百行完成这件事：一个最小的 protobuf 解码器按字段编号读出上面这些消息，再把每个事件展开成一个字典（plane、line、name、起始时刻 `start_ps`、持续时间 `duration_ps`、属性 `stats`）。

设备上的三条轨道：

| 轨道 | 一个事件是 |
| --- | --- |
| `XLA Modules` | 一次程序的执行 |
| `XLA Ops` | 程序中一条 HLO 指令的执行，例如一个 Pallas kernel、一个 fusion |
| `XLA TraceMe` | kernel 内部标出的一个区域 |

设备上的事件有两个表示持续时间的字段：事件自身的 `duration_ps`，和属性中的 `device_duration_ps`。两者都以皮秒为单位，数值却略有不同。它们是同一对 GTC 读数按两种方式换算的结果，“事件的时间就是 GTC”一小节把它们各自换回 GTC。

## 事件来自 vtrace

设备上的每个事件都由程序中的一对 `vtrace` 指令产生。第一章第 2 节的清单中，kernel 段的第一条和最后一条指令是：

```tpuasm
{ misc: vtrace 0x80000000 }
...
{ misc: vtrace 0x90000000 }
```

`vtrace` 在 `misc` 槽中，没有任何计算效果。执行时，TensorCore 把它的 32 位操作数连同当时的 GTC 写进 trace 缓冲区；profiler 开启时，主机读回这些记录，把操作数相配的两条记录配成一个区间，再把两个 GTC 读数换算成时间。硬件不理解操作数的含义，含义是编译器与主机软件的约定：

| 操作数的高 4 位 | 低 28 位 | 配成的事件 |
| --- | --- | --- |
| `0x8` 开始、`0x9` 结束 | HLO 指令的序号；`0xfffffff` 表示整个程序 | `XLA Ops`；`0xfffffff` 是 `XLA Modules` |
| `0xb` 开始、`0xc` 结束 | 区域的编号 | `XLA TraceMe` |

配对是主机做的：遇到开始标记就把（编号，GTC）压栈，遇到结束标记就从栈顶找编号相同的一项弹出，得到一个区间（[研究报告 35](../../../pallas-tpu-readings-dev/research_reports/35_pallas_vtrace_to_xplane.md)）。所以区间可以嵌套，同一个编号在每次执行中都可以再次使用。

第 3 节的 `tpuasm_tools.hlo_ops` 找的就是 `0x8`、`0x9` 这两种标记：`KernelClock.time_ops` 在同样的位置读 LCC，测的是同一个区间，只是换了一个计数器。

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

```tpuasm
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

每个 `named_scope` 变成一对 `vtrace`，低位的 0–3 是区域的编号，开始与结束靠编号配对。区域的名字不在指令中：编译器把编号与名字的对应关系存在程序的 metadata 里，主机解析时再查回来（研究报告 35）。位 27 的 `0x08000000` 来自编译器分配编号的范围，不是类型的一部分。嵌套的 `named_scope` 得到嵌套的区域，外层的结束标记排在内层之后。

`XLA TraceMe` 轨道上的四个区域，16 次调用的中位数。ΔT 是两条 `vtrace` 记录的 GTC 高 60 位之差，时间是 ΔT / 0.7 GHz（下一小节说明这两列的来历）：

| 区域 | ΔT | 时间 | 对照（第二章第 2 节的时延公式，按 1.05 GHz 换成时间） |
| --- | ---: | ---: | --- |
| `load_start` | 1 | 1.4 ns | — |
| `load_wait` | 689 | 984.3 ns | HBM → TC VMEM 512 KiB：`483.6 + 1.105 × 512` ≈ 1049 个周期，999 ns |
| `compute` | 89 | 127.1 ns | 区域内 133 个 bundle |
| `store` | 633 | 904.3 ns | TC VMEM → HBM 512 KiB：`417.5 + 1.051 × 512` ≈ 956 个周期，910 ns |

这种做法有代价：每条 `vtrace` 独占一个 bundle，区域的边界还限制了编译器跨边界重排指令（[研究报告 34](../../../pallas-tpu-readings-dev/research_reports/34_named_scope_changes_tpu_scheduling.md)，[JAX issue #40720](https://github.com/jax-ml/jax/issues/40720)）。带 region trace 测到的，是一个与正式运行不同的程序。

## 手写 vtrace

本小节实验[源码](02_tpuasm_hand_vtrace.py)、[输出](02_tpuasm_hand_vtrace.txt)。

既然事件只取决于 `vtrace` 的操作数，就不必经过 `named_scope`：用 tpuasm 在已编译的程序中直接插入一对 `vtrace`，编译器的调度完全不变。载体是上一个实验的 kernel，关闭 region trace；计算部分是 bundle 511–643，共 133 个。`find_bundles` 找到第一条 `vld` 和最后一条 `vst`，`insert_bundles` 在前者之前、后者之后各插入一个 bundle：

```python
first = tpuasm_tools.find_bundles(serialized, 'vld.8x128')[0]
last = tpuasm_tools.find_bundles(serialized, 'vst.8x128')[-1]
patched = tpuasm_tools.insert_bundles(serialized, {first: bundle(f'misc: vtrace {start}'), last + 1: bundle(f'misc: vtrace {stop}')})
```

实验只改两个操作数：

| 插入的一对 `vtrace` | 多出的事件 |
| --- | --- |
| `0xb8000000`、`0xc8000000` | `XLA TraceMe` 轨道，名字 `$$unknown$$`，ΔT = 89（127.1 ns） |
| `0xb0000005`、`0xc0000005` | `XLA TraceMe` 轨道，名字 `overlayer-overhead`，ΔT = 90（127.9 ns） |
| `0x80000007`、`0x90000007` | `XLA Ops` 轨道，名字 `region.7`，ΔT = 90（127.9 ns） |

三种都得到了与 `named_scope` 的 `compute` 相同的事件：每次调用的 ΔT 是 89 或 90（下一小节解释为什么有两个值），表中是 16 次的中位数。区别只在名字：主机用编号到程序的 metadata 中查名字，手写的编号查不到时显示 `$$unknown$$`，碰巧与 runtime 自己的编号相同时显示那个编号的名字；写成 `0x8`、`0x9` 类型时，它被当作一条 HLO 指令，名字是 `region.` 加编号。需要多个区域时，用不同的编号区分。

## 事件的时间就是 GTC

本小节实验[源码](03_tpuasm_vtrace_gtc.py)、[输出](03_tpuasm_vtrace_gtc.txt)。

上一节看到，GTC 的读数 `G` 由高 60 位的时间计数 `T = G >> 4` 和低 4 位的本地序号组成；在时间源上，`G` 每周期按 1、15、16 循环增加，每 3 个周期合计 32。如果 `vtrace` 记录的确实是 GTC，那么事件的持续时间就应当带着这种特征。

### 从 XPlane 换回 GTC

主机把一对 GTC 读数 `S`（开始）、`E`（结束）换成两个字段（研究报告 35 从 libtpu 中恢复的算术）：

```text
duration_ps        = (E − S) × 1000 / 11.2                    完整读数之差，每个计数 1/11.2 ns
device_duration_ps = ((E >> 4) − (S >> 4)) × 1000 / 0.7       高 60 位之差 ΔT，每一步 1/0.7 ns
```

`xprof_tools.gtc_delta` 和 `xprof_tools.gtc_ticks` 把这两个字段各自换回整数：

```python
def gtc_delta(event: dict) -> int:
    return round(event['duration_ps'] * 11.2e-3)

def gtc_ticks(event: dict) -> int:
    return round(event['stats']['device_duration_ps'] * 0.7e-3)
```

实验在计算部分的前后各插入一条 `vtrace`（上一个实验的第二种操作数），调用 48 次，列出每个字段出现过的值：

```text
duration_ps：127142（3 次）、127143（15 次）、127232（17 次）、127233（3 次）、128482（9 次）、128483（1 次）
duration_ps × 11.2 / 1000，即 GTC 之差 ΔG：1424（18 次）、1425（20 次）、1439（10 次）
device_duration_ps：127143（38 次）、128571（10 次）
device_duration_ps × 0.7 / 1000，即 GTC 高 60 位之差 ΔT：89（38 次）、90（10 次）
```

换回去之后都是整数，而且只有三个 ΔG：1424、1425、1439。这正是 134 个周期在时间源上应有的结果：134 = 3 × 44 + 2，44 组完整的循环贡献 44 × 32 = 1408，剩下的 2 个周期按起点的相位贡献 1 + 15、15 + 16 或 16 + 1，即 16、31、17，合计 1424、1439、1425。高 60 位之差相应地是 89 或 90。

所以同一段代码、每次都是恰好 134 个周期，XProf 却报告两种 `device_duration_ps`：127143 ps 与 128571 ps，相差 1428.57 ps，即 `T` 的一步。**XProf 的设备时间以 1.43 ns（1.5 个周期）为一格**，一个事件落在相邻的哪一格取决于它开始时 GTC 的相位。`device_duration_ps` 永远是 1428.57 ps 的整数倍；`duration_ps` 多保留了低 4 位，数值更细，但低 4 位只是本地序号，并不代表更高的时间分辨率。

### 与直接读 GTC、读 LCC 比较

为了确认 `vtrace` 记录的就是 `srdreg.gtclo`、`srdreg.gtchi` 读到的那个计数器，在同样的两个位置改为直接读数。`KernelClock.instrument` 在指定的 bundle 之前插入读数并存进 SMEM（第 3 节），`counter='gtc'` 时读 GTC：

```python
timed = clock.instrument(compiled, [first, last + 1], counter)
```

```text
同样的两个位置改为读 GTC：两次读数之差 1601（15 次）、1615（16 次）、1616（17 次）；高 60 位之差 100（15 次）、101（33 次）
同样的两个位置改为读 LCC：两次读数之差 151（48 次）
```

读 LCC 时两次读数总是相差 151 个周期，其中 20 个是读数自身的开销（第 3 节），这段计算是 131 个周期。读 GTC 时两次读数之间同样是 151 个周期：151 = 3 × 50 + 1，50 × 32 = 1600，再加 1、15 或 16，正是测到的 1601、1615、1616。直接读出的 GTC 与 `vtrace` 记录的时间有同样的结构。

三种方法放在一起：

| 方法 | 结果 | 含义 |
| --- | --- | --- |
| LCC | 151 − 20 = 131 个周期，每次相同 | 两个读数之间本芯片的周期数 |
| 直接读 GTC | ΔG ∈ {1601, 1615, 1616} | 151 个周期在 GTC 上的增量，随相位变化 |
| `vtrace`（XProf） | ΔG ∈ {1424, 1425, 1439}，ΔT ∈ {89, 90} | 两条 `vtrace` 相隔 134 个周期；XProf 显示 127.1 或 128.6 ns |

LCC 的 131 与 `vtrace` 的 134 相差 3 个周期，是因为两者取时间的位置不同：`vtrace` 在 `misc` 槽，属于向量一侧，记录的是它自己向量发射的时刻；LCC 读数是标量指令，前面有一条 `sfence`，读到的是此前的 bundle 全部向量发射之后、标量一侧的时刻（第 4 节）。

### 设备事件怎样放到主机的时间线上

GTC 只是一个计数，要把设备事件与主机事件画在同一条时间线上，还要知道某个 GTC 读数对应主机的什么时刻。采集时 runtime 会取若干个关联样本：读一次主机时间，读一次设备的 GTC，再读一次主机时间，把 GTC 读数与两次主机时间的中点配成一对。事件的 `start_ps` 就是以这样的样本为锚点、按 11.2 个计数 / ns 推算出来的（研究报告 35）。设备事件的持续时间不受锚点影响，但它在主机时间线上的位置带有这次关联的误差（读一次寄存器的时间，微秒量级），所以不能用 XProf 中设备事件与主机事件的相对位置去推断微秒以下的先后。

### 由此得到的限制

- **单个事件有一格的不确定。** 同样的 134 个周期，显示为 127.1 ns 或 128.6 ns。要比较两种写法相差几个周期，XProf 的分辨率不够，用 LCC。
- **把 XProf 的时间折算成周期是有前提的。** `xprof_tools.device_cycles` 按 ΔT × 1.5 折算（等价于按 1.05 GHz 换算）。这只在本芯片是时间源时准确。多芯片会话中，时间源之外的芯片是跟随者：它的 `T` 被不断校正到时间源上，与本地周期的比例偏离 3 : 2 百万分之几，而且一次校正可以让 `T` 停住或多走十几步（上一节）。落在校正上的短事件会被显示得更短或更长。
- **多芯片时，XProf 的时间线是 GTC 的时间线。** 这也是它的用处：不同芯片上的事件画在同一条时间轴上，可以直接比较先后，精度是上一节给出的微秒量级。LCC 做不到这一点。

## 逐指令的事件是插值出来的

本小节实验[源码](04_pallas_instruction_events.py)、[输出](04_pallas_instruction_events.txt)。

XProf 还能显示每条指令的事件。打开编译选项 `xla_xprof_enable_custom_call_tracing`，TensorCore 的轨道多出 `VALU Instructions`、`VLD Instructions`、`VST Instructions`、`SALU Instructions` 和 `Pallas Primitives`，每个事件带有指令名和它所在的 bundle 编号（编译器内部的编号，与清单中的位置不同）。这些事件看起来像是每条指令各有一个时间戳，其实不是：硬件只在执行 `vtrace` 时记录 GTC，其余指令没有任何记录。

清单中多出的只有几条 `0xa` 类型的 `vtrace`（`named_scope` 的区域标记也一并打开了）：

```text
(1, '0xa0000000')、(153, '0xa000018f')、(161, '0xa0000199')
```

`0xa` 类型的操作数低位是 bundle 编号：0、399、409。编译器的配置是每 10 个 bundle 放一个这样的标记（[研究报告 38](../../../pallas-tpu-readings-dev/research_reports/38_sparse_bundle_trace_missing_dma.md)），放不进去的地方就没有。硬件只在这些标记和其他 `vtrace` 处留下时间；两个标记之间的指令事件，是主机把这段时间按 bundle 编号均匀分开得到的（[研究报告 37](../../../pallas-tpu-readings-dev/research_reports/37_loop_primitive_relations_patch_design.md)）。一次执行中，各 bundle 的事件与上一个 bundle 的起始时刻之差：

```text
bundle 编号  指令            与上一个 bundle 相差（ns）
 10         vtrace          2.86
 11         vtrace          126.25
 12         dma.done.wait   126.25
 13         vsyncadd        126.34
 ...
 19         vadd.f32        126.25
 20         vadd.f32        126.34
 21         vadd.f32        1.07
 22         vadd.f32        1.16
 ...
```

编号 11 到 20 的 bundle 每个相隔约 126.3 ns。这一段真实的情况是：等待输入 DMA 的 `dma.done.wait` 一条指令占了约 1260 ns，其余指令各占约 1 ns（一个周期）。主机只知道这 10 个 bundle 一共用了 1263 ns，就给每个 bundle 分了十分之一。从编号 21 起，相邻的标记之间只有计算，每个 bundle 相隔约 1 ns，插值与实际一致。

所以逐指令的事件只能用来看“执行到了哪一段”，不能当作每条指令的耗时：等待集中在哪一条指令上，从这些事件中看不出来。两个标记的时间差太小时，主机还会跳过它们之间的全部指令事件（研究报告 38）。要知道一条指令或一小段指令的周期数，用第 3 节的 LCC。

## 区间测的是什么

第一个实验中 `load_start` 只有 1.4 ns，而输入 DMA 要约 1000 ns：区域只包含发起 DMA 的那条指令，DMA 本身在区域结束之后才在后台进行。等待它的 `load_wait` 才包含了 DMA 的时间。区域的边界是 `vtrace` 指令执行的时刻，不是区域中工作完成的时刻。

`vtrace` 在 `misc` 槽，属于向量一侧（第 4 节），在向量发射时记录。由此可以判断一个区间包含什么：

- 结束标记排在一条 `vwait` 之后时，`vwait` 等到 DMA 完成才向量发射，结束标记更晚，所以区间包含这次等待。`load_wait` 和 `store` 都是这样。
- 区间只包含发起而不包含等待时，测到的只是发起。异步的 DMA、MXU 计算都要到对应的等待或取回之后才算完成（[研究报告 39](../../../pallas-tpu-readings-dev/research_reports/39_vtrace_completion_dependencies.md)、[40](../../../pallas-tpu-readings-dev/research_reports/40_vtrace_vector_issue_and_completion.md)）。
- 标量一侧的工作可以越过开始标记提前执行：开始标记被前面的向量等待拖住时，标量指令已经走到了前面，区间会少算这部分工作（[研究报告 47](../../../pallas-tpu-readings-dev/research_reports/47_vtrace_vif_timing.md)）。

在打点较稀疏的采集模式下，profiler 还会因为内部检查跳过一部分指令事件（研究报告 38）。

## 只有 XProf 能给出的：主机一侧的事件

本小节实验[源码](05_jax_host_events.py)、[输出](05_jax_host_events.txt)。

第 1 节看到，调用一个 kernel 并等到结果，比设备上的时间多出 100 多微秒。这段时间花在主机上，设备上的任何计数器都看不到；XPlane 的 `/host:CPU` 中有主机一侧的事件，可以把一次调用拆开。实验对 `pl.delay(100000)` 的 kernel 连续调用 8 次，每次用 `jax.profiler.TraceAnnotation('call')` 标出整个调用，统计调用区间内各个主机事件的开始时刻与持续时间：

```python
with jax.profiler.TraceAnnotation('call'):
    compiled(x).block_until_ready()
```

8 次调用的中位数（开始时刻相对调用开始，单位 µs）：

| 事件 | 开始 | 持续 | 含义 |
| --- | ---: | ---: | --- |
| `PjitFunction(jit(wait))` | 2.3 | 82.8 | Python 一侧的 `jit` 调用（嵌套出现两次） |
| `PjRtCApiLoadedExecutable::Execute` | 11.5 | 68.4 | 交给 runtime 执行 |
| `CommonPjRtLoadedExecutable::ExecutePrepare` | 17.2 | 11.2 | 准备参数，分配输出 buffer |
| `TpuLoadedExecutable::ExecuteLaunch` | 28.8 | 46.0 | 把程序放进设备的执行队列 |
| `tpu::System::Execute`（`core_id` 0、1） | 30.2、57.0 | 25.8、15.5 | 其中每个 TensorCore 一次入队 |
| `ReadSyncFlag`（两次） | 208.3、210.4 | 26.4、27.3 | 读取设备的完成标志 |
| `CompleteCallbacks`（两次） | 235.3、237.9 | 15.5、25.7 | 完成后的回调 |
| `tpu::System::Execute=>Done`（`core_id` 1、0） | 246.2、251.0 | 2.7、19.7 | 标记每个 TensorCore 执行结束 |
| 整个调用 | 0 | 272.7 | |

按时间顺序，一次调用分成三段：

- **提交，约 75 µs。** 从调用开始到 `ExecuteLaunch` 结束（28.8 + 46.0），主机在 Python、参数处理、输出分配和入队上花掉的时间。两个 TensorCore 各入队一次。
- **等待设备，约 134 µs。** 从入队结束到主机开始读完成标志（208.3）。设备上的 module 119.0 µs、kernel 100.8 µs 都在这一段之内，其余是程序启动与结束、完成标志传回主机的时间。
- **完成处理，约 64 µs。** 读完成标志、执行回调、标记结束，直到 `block_until_ready()` 返回（272.7）。

所以 1 万个周期（约 10 µs）的 kernel 调用并等到结果要 160 µs 左右，不是因为 kernel 慢：提交与完成处理这两段在主机上就占了约 139 µs，与 kernel 的长短无关。要缩短调用方的等待，只能减少调用次数，例如把多步工作合进一个程序、在 kernel 内部循环，或者让多个调用在途重叠。

## 什么时候用哪一种

| 要回答的问题 | 方法 |
| --- | --- |
| 一个 kernel 或其中一段在一个 TensorCore 上要多少周期 | LCC（第 3 节的 `KernelClock`） |
| 一段手写的指令序列要多少周期 | LCC（第 3 节的 `LccProbe`） |
| 不同 TensorCore、不同芯片上的事件谁先谁后、相隔多久 | GTC（第 6 节）；不想改程序时用 XProf，它的时间线就是 GTC |
| 一段时间是多少秒 | GTC，区间要足够长；单芯片会话中也可以用 LCC 的周期数除以 1.05 GHz |
| 程序中有哪些 HLO 指令、各自大致多久，不想改程序 | XProf 的 `XLA Ops` |
| 主机在一次调用中做了什么 | XProf 的主机事件 |
