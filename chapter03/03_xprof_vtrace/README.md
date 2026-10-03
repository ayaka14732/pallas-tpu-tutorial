# XProf 与 vtrace

第 1、2 节用 XProf 读出了 module 与 kernel 的设备时间。这些事件来自程序中的 `vtrace` 指令。本节看 `vtrace` 的编码、怎样用它在 kernel 内部标出区域，以及为什么它标出的区间不一定是想测的那一段。

## vtrace 的编码

第一章第 2 节的清单中，kernel 段的第一条和最后一条指令是：

```text
{ misc: vtrace 0x80000000 }
...
{ misc: vtrace 0x90000000 }
```

`vtrace` 在 `misc` 槽中，执行时让 TensorCore 把一个 32 位的操作数连同当时的时间写进 trace 缓冲区；profiler 开启时，主机读回这些记录，配对成区间，再换算成主机时间。操作数的高 4 位是记录的类型：`0x8` 与 `0x9` 是一条 HLO 指令的开始与结束，XProf 的 `XLA Ops` 事件就由这一对构成。

kernel 内部的区域用另一对类型。Pallas 把 `jax.named_scope` 的边界降低为 `trace_start` 与 `trace_stop`，但默认不生成指令；要用编译选项 `xla_enable_custom_call_region_trace` 打开：

```python
compiled = tpuasm_tools.compile(affine, x, mesh=mesh, compiler_options={'xla_enable_custom_call_region_trace': flag})
```

## 实验：四个区域

本小节实验[源码](01_pallas_trace_regions.py)、[输出](01_pallas_trace_regions.txt)。

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

实验只改编译选项。关闭时 kernel 段 159 个 bundle，只有开头和结尾两条 `vtrace`；打开时 168 个 bundle，10 条 `vtrace`。打开时的清单（连续 6 个以上不含 `vtrace` 的 bundle 折叠成一行）：

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

逐段读：

- `[0]` 与 `[167]` 是 kernel 本身的开始与结束（类型 `0x8`、`0x9`），对应 XProf 的 `XLA Ops` 事件。`[1]`–`[3]` 是 `num_cores=1` 时让 TensorCore 1 跳过主体的分支（第一章第 1 节）。
- 每个 `named_scope` 变成一对 `vtrace`：类型 `0xb` 开始、`0xc` 结束，低位的 0–3 是区域的编号，开始与结束靠编号配对。`load_start`（编号 0）只包住发起 DMA 的 `[5]`、`[6]`；`load_wait`（编号 1）包住 `vwait` 与信号量清零；`compute`（编号 2）包住 `[13]`–`[145]` 共 133 个 bundle 的读、算、写；`store`（编号 3）包住输出 DMA 的发起与等待。
- 区域的名字不在指令中：编译器把编号与名字的对应关系存在程序的 metadata 里，主机解析时再查回来（[研究报告 35](../../../pallas-tpu-readings-dev/research_reports/35_pallas_vtrace_to_xplane.md)）。位 27 的 `0x08000000` 来自编译器分配编号的范围，不是类型的一部分。
- 每条 `vtrace` 独占一个 bundle，8 条就多出 8 个 bundle；除此之外还多出 1 个 bundle。区域的边界还限制了编译器跨边界重排指令（[研究报告 34](../../../pallas-tpu-readings-dev/research_reports/34_named_scope_changes_tpu_scheduling.md)）；本例的数据依赖本来就不允许跨区域重排，影响很小，在可以交错的程序中影响会更大。所以带 region trace 测到的时间，属于一个与正式运行略有不同的程序。本例中 kernel 的时间从 2012 ns 变为 2039 ns。

XProf 中 TensorCore 0 的 `XLA TraceMe` 轨道上出现四个区域，16 次调用的中位数：

| 区域 | 时间 | 对照 |
| --- | ---: | --- |
| `load_start` | 1 ns | — |
| `load_wait` | 985 ns | HBM → TC VMEM 512 KiB：`483.6 + 1.105 × 512` ≈ 1049 周期 ≈ 999 ns |
| `compute` | 127 ns | 133 个 bundle，按每周期一个约 127 ns |
| `store` | 903 ns | TC VMEM → HBM 512 KiB：`417.5 + 1.051 × 512` ≈ 956 周期 ≈ 910 ns |

`compute` 区域的 127 ns 与区域内的 133 个 bundle 按 1.05 GHz 换算的 126.7 ns 一致：这一段没有任何等待，每周期发射一个 bundle（第 4 节）。

### 只改结构：嵌套的区域

`named_scope` 可以嵌套。把 `compute` 和 `store` 放进一个外层区域：

```python
with jax.named_scope('compute_and_store'):
    with jax.named_scope('compute'):
        x_vmem[...] = x_vmem[...] * 2.0 + 1.0
    with jax.named_scope('store'):
        pltpu.async_copy(x_vmem, o_hbm, sem).wait()
```

清单中多了一对编号为 2 的 `vtrace`，内层的两个区域编号变为 3、4，外层的结束标记排在内层之后：

```text
...；vtrace 0xb8000002；vtrace 0xb8000003；vtrace 0xc8000003；vtrace 0xb8000004；vtrace 0xc8000004；vtrace 0xc8000002；vtrace 0x90000000
```

XProf 中外层区域 1034 ns，内层 `compute` 127 ns、`store` 903 ns，合计 1030 ns：外层的时间就是内层之和加上两条 `vtrace` 之间的几个周期。嵌套的区域适合先粗后细地定位：先看外层占多少，再看是哪个内层。

## 区间测的是什么

`load_start` 只有 1 ns，而输入 DMA 要约 1 µs：区域只包含发起 DMA 的那条指令，DMA 本身在区域结束之后才在后台进行。等待它的 `load_wait` 才包含了 DMA 的时间。区域的边界是 `vtrace` 指令执行的时刻，不是区域中工作完成的时刻。

`vtrace` 在 `misc` 槽，属于向量一侧（第 5 节），在向量发射时记录。由此可以判断一个区间包含什么：

- 结束标记排在一条 `vwait` 之后时，`vwait` 等到 DMA 完成才向量发射，结束标记更晚，所以区间包含这次等待。`load_wait` 和 `store` 都是这样。
- 区间只包含发起而不包含等待时，测到的只是发起。异步的 DMA、MXU 计算都要到对应的等待或取回之后才算完成（[研究报告 39](../../../pallas-tpu-readings-dev/research_reports/39_vtrace_completion_dependencies.md)、[40](../../../pallas-tpu-readings-dev/research_reports/40_vtrace_vector_issue_and_completion.md)）。
- 标量一侧的工作可以越过开始标记提前执行：开始标记被前面的向量等待拖住时，标量指令已经走到了前面，区间会少算这部分工作（[研究报告 47](../../../pallas-tpu-readings-dev/research_reports/47_vtrace_vif_timing.md)）。这与第 5 节中 LCC 读数看不到向量一侧的问题方向相反。

另外两点限制了它的精度。记录的时间来自 GTC，第 7 节会看到 GTC 并不是每个周期均匀增加，单个区间的分辨率在纳秒量级；XProf 显示的时间还经过了 GTC 到主机时间的换算。在打点较稀疏的采集模式下，profiler 还会因为内部检查跳过一部分指令事件（[研究报告 38](../../../pallas-tpu-readings-dev/research_reports/38_sparse_bundle_trace_missing_dma.md)）。

所以 XProf 的区域适合回答“一个 kernel 中，微秒量级的几段工作各占多少时间”，并且要按上面的规则确认每个区间的两端；要精确到周期，或者测量不改变调度的原始程序，用第 4 节的 LCC。
