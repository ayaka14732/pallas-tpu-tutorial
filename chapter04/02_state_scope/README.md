# 状态的作用域与生命周期

第 1 节在一个 kernel 内部装入状态并生成。实际使用时还要知道：状态属于谁，kernel 结束后是否保留，两个 TensorCore 是否共用一份，程序开头是否有人改过它。这些决定了随机数能否复现、不同 TensorCore 的序列是否会重复。

## 编译器加入的前导

本小节实验[源码](01_tpuasm_state_lifetime.py)、[输出](01_tpuasm_state_lifetime.txt)。

完整清单中，runtime 代码里有两条与随机数有关的指令：

```text
bundle 455：va0: setrngseed v13；bundle 456：va0: vrng.8x128.u32 v17
```

它们之前约 300 个 bundle 在计算 `v13`。计算的输入是：

```text
{ s1: sld s6, [smem:0x3ffe0] }                      # runtime 写入的一个 32 位值
{ s0: seq.s32 p0, s6, 0 }
{ s0: @p0 sbr.rel L_01c9 ; s1: @!p0 sld s7, [smem:0x0] ; ... }   # 为 0 时跳过整段
{ s0: @!p0 sxor.u32 s11, 0xae5a53a9, s6 ; s1: @!p0 sld s8, [smem:0x1] ; ... }
...
{ s0: sshll.u32 s9, s7, 0x1 ; ... }
{ s0: smul.u32 s15, -905840163, s14 ; s1: sadd.s32 s10, s9, s8 }   # 芯片信息 × 2 + 本核编号
```

`[smem:0x1]` 是本核编号（第二章第 1 节），`[smem:0x0]` 是芯片信息；`vlaneseq` 提供每个 lane 的编号。所以程序每次启动时，前导用 runtime 写在 `[smem:0x3ffe0]` 的值、芯片、TensorCore 和 lane 的编号算出一个状态装入生成器，再空转一次 `vrng`。实验读出连续 4 次调用中这个值：

```text
连续 4 次调用中 [smem:0x3ffe0] 的值：0xc650c698、0x0cfb5ae9、0x4d9456c9、0x9d6f3188
连续两次调用读出的状态相同：False
同一次调用中两个 TensorCore 读出的状态相同：False
```

每次启动的值都不同，两个 TensorCore 的状态也不同。也就是说，在 kernel 开始之前，生成器已经处于一个每次启动都变、每个 TensorCore 都不同的状态。不设定种子就使用 `vrng`，得到的随机数不可复现。

## 去掉前导：状态跨调用保留

要看硬件本身的行为，先用 tpuasm 把前导的这两条指令换成 `vnop`：

```python
probe.serialized = tpuasm_tools.edit_bundles(probe.serialized, edits)   # setrngseed、vrng → vnop
```

`find_bundles(..., whole_program=True)` 在整个程序（包括 runtime 代码）中查找这两条指令。然后依次调用三种不同的 executable，每个都只做一件事：

| 第几次 | executable | 与模型一致 |
| ---: | --- | --- |
| 1 | 装入种子并生成 | True |
| 2 | 读状态 | True |
| 3 | 生成 | True |
| 4 | 读状态 | True |
| 5 | 生成 | True |
| 6 | 读状态 | True |
| 7 | 装入第 4 次读出的状态并生成 | 与第 5 次的结果相同：True |

模型从第 1 次装入的种子出发，按调用顺序推进：第 3 次的“生成”继续了第 1 次留下的状态。由此得到三条结论：

- 状态属于物理的 TensorCore，在 kernel 结束后保留，下一个 executable 从这里继续。
- `getrngseed` 只读不改：第 2、4、6 次读状态之后，下一次生成仍与模型一致。
- `getrngseed` 读出的状态可以原样用 `setrngseed` 装回，精确重放之后的序列。这就是随机数的保存与恢复。

但在未修改的程序中，前导每次启动都会覆盖状态，所以“状态跨调用保留”对普通的 Pallas 程序没有意义：要跨调用延续一个序列，只能自己保存状态（`getrngseed`）、下次再装回。

## 两个 TensorCore

去掉前导后，两个 TensorCore 执行同一段片段：

```text
装入相同状态：两个 TensorCore 的输出相同 True
TensorCore 0 装入“输入 ^ 0”：与模型一致 True
TensorCore 1 装入“输入 ^ 1”：与模型一致 True
```

每个 TensorCore 有自己的一份状态。硬件不会把核编号混进状态：装入相同的状态，两个 TensorCore 产生完全相同的序列。要让它们产生不同的序列，必须装入不同的状态，这里用 `sld` 读出核编号，异或进种子：

```python
differ = bundle('s1: sld s29, [smem:0x1]') + bundle('s0: sfence') + bundle('va0: vmov.8x128 v15, s29') + GAP + bundle('va0: vxor.8x128.u32 v16, v10, v15') + GAP
```

（`sld` 之后的 `sfence` 不是必需的，这里只是保证 `vmov` 读到的是新值。）

## 对使用随机数的设计意味着什么

- 需要复现的随机数，必须显式设定种子；不能依赖前导装入的状态。
- 多个 TensorCore、多颗芯片要产生不同的序列，必须由程序把各自的编号混进种子；用同一个种子会得到完全相同的序列。
- 一个序列要跨 kernel 调用延续，就把状态读出来，作为数据保存，下次再装回。状态跟随数据，而不是跟随某个物理 TensorCore。

[研究报告 57](../../../pallas-tpu-readings-dev/research_reports/57_tpu_v4_rng.md) 在本 host 的四颗芯片、八个 TensorCore 上验证了同样的状态行为。
