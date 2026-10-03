# GTC 与多个 TensorCore 的计时

LCC 只能在同一个 TensorCore 上相减（第 4 节）。要比较两个 TensorCore、两颗芯片上发生的事件，或者把周期换算成秒，需要另一个计数器：全局时钟 GTC（global time counter）。本节测出 LCC 与 GTC 的频率和比例，再用 GTC 界定同一颗芯片两个 TensorCore 之间的时间偏移。

## 读 GTC

GTC 与 LCC 的读法相同，也是同一个 bundle 中读低、高两半：

```text
{ s0: srdreg.gtclo s21 ; s1: srdreg.gtchi s26 }
```

[`tpuasm_tools`](../../tpuasm_tools.py) 中的 `read_gtc(low)` 生成这个 bundle。GTC 与 LCC 的读数都在标量一侧执行，两者之间不需要 `sfence`。

为了在一段片段中读 4 次，`LccProbe` 增加了 `run_raw(body, reads)`：它返回每次运行的全部 64 位读数，而不只是差值；`LccProbe(num_cores=2)` 让一颗芯片的两个 TensorCore 执行同一段片段，各自返回读数。

## 频率与比例

本小节实验[源码](01_tpuasm_gtc_rate.py)、[输出](01_tpuasm_gtc_rate.txt)。

片段用 LCC 包住 GTC，中间用 `vdelay` 让向量一侧停住 H 个周期，再用 `sfence` 让后面的读数等它结束：

```python
setup = bundle(f's0: simm.s32 s24, {delay}')
body = read_lcc(20) + read_gtc(21) + bundle('misc: vdelay s24') + bundle('s0: sfence') + read_gtc(22) + read_lcc(23)
```

实验只改 H，从 10³ 到 10⁹：

| H | ΔL | ΔG |
| ---: | ---: | --- |
| 10³ | 1015 | 10800、10801 |
| 10⁶ | 1000015 | 10666801、10666815 |
| 10⁹ | 1000000015 | 10666666800、10666666815 |

ΔL 总是 H + 15，完全确定。GTC 的两个读数之间隔了 ΔL − 2 个周期（外侧的 LCC 两端各多一个相邻读数），ΔG / (ΔL − 2) 随 H 增大收敛到 10.666667，即 **32/3**：LCC 每走 3 个周期，GTC 走 32。

再把 H = 10⁸ 与 10⁹ 两次运行的主机时间相减（装载程序等固定开销在差值中抵消），得到以主机时钟为准的频率：

```text
LCC 频率 1.0499 GHz，GTC 频率 11.199 GHz
```

这就是前两章换算时使用的“约 1.05 GHz”。GTC 名义上是 11.2 GHz，[研究报告 53](../../../pallas-tpu-readings-dev/research_reports/53_v4_gtc_fixed_timebase.md) 在约 33 秒、覆盖空闲、标量、DMA、向量和 MXU 负载的时间轴上测得它与名义值相差约 4 ppm，且不随负载变化；LCC 与 GTC 的比例在这些负载下也保持 32/3。

## GTC 不是每周期均匀增加

32/3 不是整数，GTC 每个周期不可能都加同样的数。连续 4 个 bundle 读 GTC：

```text
32 次运行出现的 (G1 − G0, G2 − G1, G3 − G2)：[(1, 15, 16), (15, 16, 1), (16, 1, 15)]
G3 − G0：[32]
```

每个周期的增量按 1、15、16 循环，每 3 个周期合计 32。所以 GTC 的单个计数不是 1/11.2 ns 的均匀刻度：相邻两个周期的 GTC 差可能是 1，也可能是 16。只有跨越较长的区间，ΔG / 11.2 才是可靠的纳秒数。研究报告 53 还观察到少数偏离这个循环的序列，甚至连续几个周期 GTC 不变。

由此得到两个计数器的分工：

- 同一个 TensorCore 上，优化一段代码用 LCC：它直接给出周期数，精确到 1 个周期。
- 比较不同 TensorCore、不同芯片上的事件，或者把时间换算成秒，用 GTC，并且只在足够长的区间上使用。

## 两个 TensorCore 的 GTC 偏移

本小节实验[源码](02_tpuasm_two_core_gtc.py)、[输出](02_tpuasm_two_core_gtc.txt)。

两个 TensorCore 的 GTC 读数能否直接相减，取决于它们之间的偏移有多大。偏移不能直接测：只有一方读数、另一方读数，无法知道两次读取是否同时发生。可以利用因果关系给出上下界。每个 TensorCore 执行同一段片段：

```python
BODY = (
    read_gtc(20)                                                 # G0
    + bundle('misc: vsyncadd.remote.s32 [sflag:s29], 1')         # 给对方发信号
    + bundle('misc: vwait.ge [sflag:45], 1')                     # 等对方的信号
    + bundle('misc: vsyncadd.s32 [sflag:45], -1')
    + bundle('s0: sfence')
    + read_gtc(21)                                               # G1
    + read_lcc(22)
)
```

`s29` 是对方 TensorCore 的 sflag 45 的地址，在 `setup` 中按编译器生成入口汇合的同样方式拼出（第二章第 1 节）。sflag 45 正是入口汇合用的同步标志，片段中加 1、减 1 各一次，不影响之后的出口汇合。

TensorCore 0 先读 G0、再发信号；TensorCore 1 收到信号、经过 `sfence` 之后才读 G1。所以 TensorCore 0 的 G0 在真实时间上早于 TensorCore 1 的 G1，反方向同理。记 o 为 TensorCore 1 的 GTC 减去 TensorCore 0 的 GTC 的偏移，两条因果关系给出：

```text
G0[1] − G1[0] ≤ o ≤ G1[1] − G0[0]
```

一次运行给出的区间宽约 960–1850 个 GTC 计数，主要是信号在两个 TensorCore 之间往返的时间。32 次运行的区间取交集：

```text
32 个区间的交集：[-481, 481] 个 GTC 计数，即 [-42.9, 42.9] ns
```

所以同一颗芯片两个 TensorCore 的 GTC 偏移不超过约 43 ns，与研究报告 53 用 1344 次交换得到的 ±496 个计数一致。这是上下界，不是测得的偏移：零偏移与数据相容，但数据只能证明偏移落在这个区间内。在这个精度内，两个 TensorCore 的 GTC 读数可以直接相减。例如在本实验中，两个 TensorCore 读 G0 的时刻相差 −304 到 1184 个计数，即 −27 到 106 ns：它们都在入口汇合之后、输入 DMA 之后到达同一位置，到达的先后和差距每次运行都不同。第二章第 1 节所说的“谁先到、差多少”，就可以在汇合前后各插一次这样的读数测出。

同时读的 LCC 则不能这样用：

```text
紧随 G1 的 LCC 读数之差（TensorCore 1 − TensorCore 0）：-51329–-51271
```

两个 TensorCore 的 LCC 相差约 51300 个周期，这是两个计数器起点不同造成的，与事件的先后无关。它们的速率相同，所以差值大致固定；但硬件不保证这个起点差，要比较两个 TensorCore 的 LCC，只能先借助 GTC 这样的公共时钟标定它。

## 跨芯片与跨 host

GTC 在整个切片的芯片之间是同步的。研究报告 53 在 TPU v4 (2x2x2) 的两个 host、八颗芯片、十六个 TensorCore 上用同样的因果交换验证：所有交换都满足因果顺序，零偏移与全部数据相容；能证明的偏移上界在本 host 其他芯片约为 0.6–1 µs，另一个 host 的芯片最大约 1.5 µs。这些上界主要来自跨芯片信号的往返时间，而不是 GTC 本身的误差。

所以跨芯片比较时间时，可以用 GTC 读数直接相减，但要记住这个精度只有微秒量级：它足以比较两颗芯片上一次 all-reduce 的开始和结束（第二章第 7 节，几到几十微秒），不足以比较单条 remote DMA 的到达时刻（约 2 µs）。第二章第 5 节提到的跨 host 比较，就要按这个精度解读。
