# LCC：指令级计时

主机计时（第 1、2 节）只能看到整个程序的时间。XProf 的区域（第 3 节）能标出 kernel 内部的几段，但它改变了调度，分辨率也只到纳秒。要知道 kernel 中某几个 bundle 花了多少周期，需要在 kernel 内部读一个周期计数器。TensorCore 有一个本地周期计数器 LCC（local cycle counter），每个周期加 1，标量单元可以用一条指令读出它的值。本节写出读取 LCC 的方法，并用它验证第一章第 2 节的说法：没有等待时，TensorCore 每个周期发射一个 bundle。

## 读 LCC 的指令

LCC 是 64 位的，标量寄存器是 32 位的，所以要分两半读：

```text
{ s0: srdreg.lcclo s20 ; s1: srdreg.lcchi s25 }
```

`srdreg` 读一个特殊寄存器，`lcclo` 和 `lcchi` 是 LCC 的低、高 32 位。两条指令放在同一个 bundle 的两个标量槽中，同时执行，得到的两半属于同一个时刻，拼起来就是一致的 64 位读数。如果分在两个 bundle 读，两次读之间低位可能恰好进位，拼出的值会错 2³²。

Pallas 的公开接口没有读 LCC 的函数，所以本章用 tpuasm 把这样的 bundle 直接插进机器程序。

## 载体：LccProbe

[`tpuasm_tools.LccProbe`](../../tpuasm_tools.py) 把插入的过程包装起来。它先用 Pallas 编译一个载体 kernel：

```python
def kernel(x_hbm, out_hbm, data, sem) -> None:
    pltpu.async_copy(x_hbm, data, sem).wait()
    # 带唯一立即数的 vxor 标出插入位置。
    data[:8, :] = data[:8, :] ^ jnp.uint32(0x13579BDF)
    pltpu.async_copy(data.at[:56, :], out_hbm, sem).wait()
```

`0x13579BDF` 在整个程序中只出现一次，`find_bundles` 由它找到插入位置。`run(body)` 在这个位置插入三段清单：

1. 前缀：把输入读进 `v10` 供片段当作数据使用，再用一条 `sfence` 让此前的工作全部发射完（第 5 节解释为什么需要这一条）。
2. 片段 `body`：由调用者写出，其中用 `read_lcc(20)`、`read_lcc(21)`、`read_lcc(22)` 三次读 LCC，记作 R0、R1、R2。
3. 后缀：把六个半字读数各广播成一个 TC VREG，存进 TC VMEM 的第 1–6 个 tile，随载体的输出 DMA 返回主机。

主机把半字拼成 64 位，返回每次运行的 `(R1 − R0, R2 − R0)`。插入后的程序不再经过编译器，bundle 的内容和顺序就是设备实际执行的内容。

片段用两个小函数写出：`bundle(text)` 生成一个 bundle 的清单文本，`read_lcc(low)` 生成上面那个读 LCC 的 bundle。

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

四种 bundle 的结果完全相同，8 次运行的读数也完全相同：

| N | 0 | 1 | 4 | 16 | 64 |
| --- | ---: | ---: | ---: | ---: | ---: |
| R1 − R0 | 1 | 2 | 5 | 17 | 65 |
| R2 − R0 | 13 | 14 | 17 | 29 | 77 |

从中读出三条规则：

- **R1 − R0 = N + 1。** 两个相邻的读数相差 1，中间每多一个 bundle，差值加 1：这些 bundle 都是一个周期发射一个。标量加法链中每条依赖前一条的结果，也没有变慢：标量运算的结果在下一个周期就可用。
- **向量指令不改变 R1。** 向量加法和 `vld` 与空 bundle 的读数相同。这不说明向量指令都只要一个周期，而是说明 R1 只反映标量一侧走到了哪里。第 5 节会看到，向量一侧可以远远落在后面，R1 却看不到。
- **`sfence` 加 11 个周期。** R2 − R1 总是 12：一个是相邻读数之间的 1，另外 11 来自 `sfence`。第 5 节解释它的含义。

第一章第 2 节说“没有等待时，TensorCore 大约每个周期发射一个 bundle”，这里给出了确切的版本：每个周期一个 bundle 指的是标量发射。对只含标量指令或与前后无依赖的向量指令的序列，清单中的 bundle 数就是周期数。

## LCC 的适用范围

LCC 只在同一个 TensorCore 上可比：两次读数相减才有意义。不同 TensorCore、不同芯片的 LCC 起点不同，不能直接相减；第 7 节介绍怎样借助全局时钟 GTC 比较不同 TensorCore 上的时间，以及怎样由此标定 LCC 的频率（约 1.05 GHz）。

[tpu-v4-latency-numbers](../../../tpu-v4-latency-numbers/README.md) 用同样的载体与读法测量了第二章引用的全部 DMA 代价。[研究报告 48](../../../pallas-tpu-readings-dev/research_reports/48_scalar_cycle_lowering.md) 则走了另一条路：为读 LCC 增加一个 Pallas primitive 和它的 lowering，由编译器生成读数指令。两条路得到的读法相同。
