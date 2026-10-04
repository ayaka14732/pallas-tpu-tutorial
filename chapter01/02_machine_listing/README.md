# TensorCore 与机器清单

本章的每一节都要问同一个问题：编译器最后让 TensorCore 执行了哪些指令。本节建立回答这个问题的工具：取得一个 kernel 的机器清单（listing），并读懂它的结构。

本节实验：最小 Pallas kernel [源码](01_pallas_f32_8x128.py)、[输出](01_pallas_f32_8x128.txt)；同一运算的原生 XLA baseline [源码](02_jax_f32_8x128.py)、[输出](02_jax_f32_8x128.txt)。

## 机器清单从哪里来

一个 Pallas kernel 先由 Mosaic 降低，再由闭源的 libtpu 编译成 TensorCore 程序映像，最后装进 executable。我们读取的就是这份程序映像本身：[tpuasm](../../../tpuasm) 从 `compiled.runtime_executable().serialize()` 中取出机器字节，用 libtpu 自己的编解码器反汇编，再把编译时保存的源码位置写成注释。

本教程不再读编译器的 final bundles 或 LLO dump。那些文本是编译器内部的表示，其中混有不会执行的伪指令，没有标出指令落在哪个物理发射槽，同一段文本有时对应不止一种机器编码，而且无法改写后重新装载。tpuasm 清单则逐字节对应实际执行的程序，可以修改、重新汇编，并在真机上运行。本章第 3 节就会用这种方法，让硬件执行一条 Mosaic 拒绝编译的 DMA。

仓库根目录的 [`tpuasm_tools.py`](../../tpuasm_tools.py) 把常用步骤封装成几个函数：

| 函数 | 作用 |
| --- | --- |
| `compile(function, *args, mesh=...)` | `jax.jit(function).lower(*args).compile()`，并保留源码位置 |
| `listing_outline(compiled)` | 完整清单的结构概览：各段从第几个 bundle 开始 |
| `kernel_listing(compiled)` | 只保留编译器归属到 HLO 指令的代码，省略 runtime 代码 |
| `count_mnemonics(listing)` | 按助记符统计指令条数 |

## 实验的 kernel

实验沿用上一节的最小 kernel，源码没有改动：把一个 `f32[8,128]` 从 HBM 搬到 TC VMEM，乘以 2，再搬回 HBM。脚本只在编译之后多打印两样东西：

```python
print(tpuasm_tools.listing_outline(compiled))  # 完整清单的结构概览
print(tpuasm_tools.kernel_listing(compiled))   # 只含 kernel 的清单
```

本章 Pallas 实验的指令全部由 TensorCore 0 执行。XLA baseline 则可能把工作分给两个 TensorCore；遇到这种情况时，清单中的计数是每个 TensorCore 各执行一份，正文会单独说明。

## 完整清单的结构

`listing_outline` 列出完整清单中各段的起点：

```text
bundle    0: prelude-prefix-start
bundle    9: routine:wait-for-active-dma
bundle   16: routine:program-exit
bundle   19: routine:allocation-table-find
...
bundle  128: trampoline:program-start
bundle  503: function jit(scale)/scale/mpmd_map [symbol 4, scale.1]; image bundles [503, 507)
bundle  504: entry bundle: %scale.1 = custom-call(%x_hbm.1)
bundle  508: function ...; image bundles [508, 517)
bundle  517: exit bundle: %scale.1 = custom-call(%x_hbm.1)
bundle  518: function ...; image bundles [518, 534)
bundle  650: 清单结束（共 650 个 bundle）
```

整个程序有 650 个 bundle，其中 kernel 只占约 30 个。前 503 个 bundle 是 runtime 的入口、公共例程和启动代码，534 号以后是结束代码。这些代码与 kernel 无关，每个程序都有。`function` 注释标出编译器归属到 Pallas kernel 的区间；`entry bundle`/`exit bundle` 标出 HLO 指令 `custom-call` 的边界。`kernel_listing` 只保留这些区间，区间之间被跳过的 runtime 代码记为一行省略注释。

## bundle 与发射槽

清单中的每对花括号是一个 bundle（指令包），对应一条 VLIW 长指令字，TensorCore 按顺序逐个执行。一个 bundle 最多含 12 条指令，每条指令属于一个物理发射槽，用 `槽名:` 标出。一个槽在一个 bundle 中最多一条指令：

| 槽 | 主要用途（本章会逐一遇到） |
| --- | --- |
| `s0`、`s1` | 标量运算、比较、分支。DMA 发起只在 `s0`；SMEM 读写（`sld`/`sst`）只在 `s1` |
| `va0`、`va1` | 向量 ALU：逐元素运算、比较、选择、类型转换、超越函数 |
| `vld`、`vst` | TC VMEM 与 TC VREG 之间的 load/store |
| `cld` | 从 Megacore Shared CMEM 读取（第二章第 3 节） |
| `vx0`、`vx1` | 送往跨通道单元（XLU）和矩阵单元（MXU）的指令 |
| `vr0`、`vr1` | 从这些单元的结果 FIFO 取回结果（`vpop`） |
| `misc` | 同步与杂项：等待、计数器加减、trace 标记 |

清单中常见的写法：

- `{ s0: simm.s32 s7, 0 }`：目的操作数在前。`s7` 是标量寄存器，`v0` 是 TC VREG，`p0` 是谓词寄存器。
- `@p0`、`@!p0`：谓词前缀，`p0` 为真（或为假）时才执行这条指令。
- `[vmem:0x0]`、`[hbm:s0]`、`[smem:0x1]`、`[sflag:52]`：带地址空间的地址。地址单位由指令决定，不一律是字节。
- `# 文件:行:列-行:列 [primitive; LLO n]`：这条指令来自哪一行 Pallas 源码、由哪个 primitive 降低而来。
- `.encoding { ... }`：tpuasm 为逐字节还原原机器编码而记下的编码选择。阅读时可以忽略；手写清单时通常也不需要写。

## 逐行读最小 kernel

kernel 段分为三个区间。第一个区间判断由谁执行：

```text
{ misc: vtrace 0x80000000 }
{ s1: sld s6, [smem:0x1] }
{ s0: sne.s32 p0, s6, 0 }
{ s0: @p0 sbr.rel L_0205 }
```

`sld` 从 SMEM 地址 `0x1` 读出当前 TensorCore 的编号，编号不为 0 时跳到 kernel 末尾。`vtrace` 是 profiler 使用的 trace 标记，不影响计算。

> 暂且可以理解为：`vtrace` 向 profiler 报告“HLO 指令开始/结束”，不产生任何计算效果。第三章第 3 节详细介绍它的编码，以及它为什么不是精确的计时手段。

第二个区间是 kernel 主体，与源码一一对应：

```text
{ s0: dma.simple [vmem:s7], [hbm:s0], length=8, dst_flag=[sflag:52] }
{ misc: vwait.ge [sflag:52], 8 }
{ vld: vld.8x128 v0, [vmem:0x0] ;
  misc: vsyncadd.s32 [sflag:52], -8 }
{ va0: vmul.8x128.f32 v1, 2.0, v0 }
{ vst: vst.8x128 [vmem:0x0], v1 }
{ s0: dma.simple [hbm:s1], [vmem:s7], length=8, dst_flag=[sflag:52] }
{ misc: vwait.ge [sflag:52], 8 }
{ misc: vsyncadd.s32 [sflag:52], -8 }
```

- `dma.simple` 按“目的、源”的顺序发起一次 DMA，`dst_flag` 是完成时累加的同步标志（Pallas 的 DMA semaphore）。
- `vwait.ge` 等到这个标志不小于 8，随后 `vsyncadd ... -8` 把它减回 0。
- `vld` 把 TC VMEM 中的一个 `8×128` tile 读进 TC VREG `v0`，`vmul` 乘以立即数 2.0，`vst` 写回 TC VMEM。

本章第 3 节逐项解释 DMA 的这些操作数。这里先注意两点：源码中的一次 `x * 2` 对应恰好一条 `vmul.8x128.f32`；`vld`、`vmul`、`vst` 分别落在 `vld`、`va0`、`vst` 三个不同的槽。

第三个区间（`image bundles [518, 534)`）在 kernel 主体之后执行，两个 TensorCore 都会经过：

```text
{ misc: vsyncadd.remote.s32 [sflag:s2], 1 }
{ misc: vwait.ge [sflag:45], 1 }
{ misc: vsyncadd.s32 [sflag:45], -1 }
{ misc: cfence }
{ misc: vtrace 0x90000000 }
```

> 暂且可以理解为：每个 TensorCore 给另一个 TensorCore 的同步标志加 1，再等待自己的标志被对方加 1，两个 TensorCore 在 kernel 结束时汇合。第二章第 1 节详细介绍这种跨核同步及其 ISA 形式。

## 一个 bundle 需要多少时间

清单本身不带时间。读本章的计数时，暂且采用下面的近似：

> 暂且可以理解为：TensorCore 每个周期最多发射一个 bundle；一条指令的操作数尚未算出、要用的单元尚未空闲时，发射会停下来等待，`vwait.ge` 这类等待指令则停到条件满足为止。所以 bundle 数只是周期数的下限。第三章第 4–6 节用真机计数器测出这些等待，并给出向量指令何时真正执行的完整发射模型。

因此本章比较两个实现时，看的是指令条数和 bundle 数，不对运行时间下结论。

## 对照：原生 XLA 的同一运算

XLA baseline 的 fusion 段有 38 个 bundle（Pallas kernel 主体区间是 9 个），计算核心同样是 1 条 `vld`、1 条 `vmul.8x128.f32`、1 条 `vst` 和 2 条 `dma.simple`。它开头也读取 TensorCore 编号，并让编号非 0 的 TensorCore 跳过 fusion：这个只有一个 tile 的运算，XLA 也只交给 TensorCore 0 执行。

多出来的代码主要有两类：

- 每次 DMA 之前，都有一串 `sadd/slt/sne/por` 和一条带谓词的 `shalt`。这是越界检查：地址超出 buffer 时让 TensorCore 停机。Pallas 实验用 `disable_bounds_checks=True` 关掉了这类检查。
- `vsyncpa.u1` 在进入和离开 fusion 时设置 DMA 同步标志的状态。

XLA 的 fusion 没有 Pallas kernel 末尾的跨核汇合区间。这正是本教程的出发点：读 XLA 生成的清单，弄清它实际做了什么，再决定手写版本在哪里可以做得不同。
