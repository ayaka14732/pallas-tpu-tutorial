# 由硬件推导集合通信

前面几节给出了跨芯片通信的全部基本操作和它们的代价。本节不介绍某个“标准”的集合通信算法，而是从这些代价出发，推导 all-reduce（每颗芯片的数组求和，结果分发回每颗芯片）应当怎样做，再用实验检验推导。目标不是背下一种算法，而是学会用硬件参数比较不同的设计。

## 推导用到的硬件事实

全部来自第 4–6 节（K 为 KiB 数）：

| 事实 | 数值 |
| --- | --- |
| 相邻芯片的 remote DMA | `1941 + 24.77K` 个周期 |
| 对角线（两跳）的 remote DMA | `2908 + 24.77K` 个周期 |
| 本 host 的四颗芯片排成 2×2，每颗有两条链路 | 0–1、0–2、1–3、2–3 相邻 |
| 不同链路、或同一链路的两个方向上的传输互不影响 | 每条链路每个方向约 43 GB/s |
| 一次 remote DMA 的固定开销 | 约 1.8 µs |

设每颗芯片的数组为 M。三种设计：

**一次交换**：每颗芯片把整份数据同时发给其余三颗，再把收到的三份相加。只付一次固定开销，但每颗芯片发出 3M，其中发往对角线的那份还要经过中间芯片转发。

**环形**：四颗芯片按物理相邻排成环（第 5 节的 0→1→3→2），把数组分成 4 块。先 reduce-scatter：每一步每颗芯片把一块发给右边、收到左边的一块并累加，3 步后每颗芯片持有一块完整的和；再 all-gather：每一步把完整的一块传给右边，3 步后每颗芯片都有全部 4 块。共 6 步，每步只传 M/4，只用相邻链路，但要付 6 次固定开销。预计约 `6 × (1941 + 24.77 × M/4)` 个周期。

**双向环形**：环形只用了每颗芯片向右的那条链路。把数组分成两半，一半沿环向右、一半沿环向左，同时进行。每步仍是 6 步，每步传 M/8，两条链路都用上。预计约 `6 × (1941 + 24.77 × M/8)` 个周期。

可以预见：M 小时固定开销占主导，一次交换最好；M 大时带宽占主导，双向环形最好。

## 写法

本小节实验[源码](01_pallas_all_reduce.py)、[输出](01_pallas_all_reduce.txt)。

三种设计都是一个 Pallas kernel，在四颗芯片的 TensorCore 0 上运行，mesh 按物理环排列。它们共用的部分是一个四方汇合：

```python
def barrier() -> None:
    ready = pltpu.get_barrier_semaphore()
    for rank in range(CHIPS):
        pl.semaphore_signal(ready, 1, device_id={'device': rank, 'tc': 0})
    pl.semaphore_wait(ready, CHIPS)
```

一次交换同时发出三个 remote DMA，写进对方 `inbox` 的不同格子，然后相加：

```python
copies = [pltpu.make_async_remote_copy(acc, inbox.at[k], send_sems.at[k], recv_sems.at[k], device_id={'device': (me + k + 1) % CHIPS, 'tc': 0}, device_id_type=pl.DeviceIdType.MESH) for k in range(CHIPS - 1)]
for copy in copies:
    copy.start()
for copy in copies:
    copy.wait_send()
    copy.wait_recv()
inbox[0] = acc[...] + inbox[0] + inbox[1] + inbox[2]
```

环形的每一步是“等下游许可 → 发送 → 等发出和收到 → 累加 → 给上游许可”：

```python
if not first:
    pl.semaphore_wait(credits.at[d], 1)          # 下游已用完上一次收到的数据
transfer = pltpu.make_async_remote_copy(source, destination, ..., device_id={'device': (me + sign) % CHIPS, 'tc': 0}, ...)
transfer.start()
transfer.wait_send()
transfer.wait_recv()
part(index)[...] = part(index)[...] + inbox[d]   # reduce-scatter 阶段累加
pl.semaphore_signal(credits.at[d], 1, device_id={'device': (me - sign) % CHIPS, 'tc': 0})
```

这里的许可（credit）是一个普通信号量，它代替了每一步的四方汇合：只有相邻的两颗芯片需要互相等待，不必等所有芯片。第 4 节说过，remote DMA 直接写对方的 buffer，必须先确认对方已经用完上一次的数据，许可就是这个确认。`sign` 为 1 时向右、为 −1 时向左，双向环形同时执行两个方向。

all-gather 阶段把完整的块直接写进下游 `acc` 的同一位置，不经过 `inbox`，省去一次复制。

## 结果

计时方法与第 2 节相同，在 kernel 内部把 all-reduce 重复 64 次和 32 次，取主机计时之差除以 32；三种设计的数值都与求和结果逐元素一致。

| 每颗芯片 | 一次交换 | 环形 | 双向环形 | 环形的预计 | 双向环形的预计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 KiB | 3.5 µs | 13.0 µs | 13.3 µs | 11.7 µs | 11.4 µs |
| 128 KiB | 9.9 µs | 17.7 µs | 14.8 µs | 15.6 µs | 13.4 µs |
| 1 MiB | 74.7 µs | 50.6 µs | 32.6 µs | 47.3 µs | 29.2 µs |

预计值按 1.05 GHz 由上面的公式算出，不含汇合与累加；实测比预计多 1.4–3.4 µs，正是这两部分。推导的结论都得到了验证：

- 16 KiB 时，环形的 6 次固定开销（约 11 µs）远大于数据量，一次交换快近 4 倍。
- 1 MiB 时，数据量占主导，一次交换让每颗芯片发出 3 MiB，最慢；双向环形用上了两条链路，比单向环形快约 35%。
- 双向环形在 16 KiB 时没有收益：每步的数据减半，但固定开销不变。

## 对照：原生 XLA

本小节实验[源码](02_jax_psum.py)、[输出](02_jax_psum.txt)。

在 `shard_map` 中调用 `jax.lax.psum`，XLA 生成一个 `psum` 段，其中有 16 条 `dma.general`。用 `fori_loop` 循环 64 次与 32 次计时：

| 每颗芯片 | XLA `psum` | 本节最快的设计 |
| --- | ---: | ---: |
| 16 KiB | 12.5 µs | 3.5 µs（一次交换） |
| 128 KiB | 14.7 µs | 9.9 µs（一次交换） |
| 1 MiB | 31.6 µs | 32.6 µs（双向环形） |

XLA 对所有大小使用同一种算法，它在大数组上与双向环形相当，在小数组上慢 1.5–3.5 倍：后者正是固定开销占主导、一次交换更合适的区域。两种计时方式不完全相同（XLA 的循环中数组可能被放进 CMEM，第 2 节），但小数组时差距主要来自算法本身的固定开销次数。

这个对照说明了手写的价值所在：XLA 必须选一种对所有情况都过得去的算法；知道硬件参数和自己的数据大小，就能选出更合适的那一种。本节的设计还有继续改进的余地，例如让每颗芯片的两个 TensorCore 各负责一半（第 1 节），或把一次交换中走对角线的那份改为两步转发。在两个 host 的 TPU v4 (2x2x2) 上，芯片多了 z 方向的链路，[研究报告 32](../../../pallas-tpu-readings-dev/research_reports/32_tpu_v4_16_allreduce.md) 按同样的思路推导了三维的 all-reduce。
