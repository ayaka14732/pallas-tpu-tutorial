# TensorCore 之间的 remote DMA 与同步

上一节借用了编译器插入的入口汇合来保证顺序。本节在 Pallas 中显式表达两个 TensorCore 之间的协作：用信号量通知对方，用 remote DMA 把数据直接写进对方的 TC VMEM。这两件事在跨芯片时形式完全相同（第 5 节），这里先在一颗芯片内看清它们。

## 写法：信号量与 remote DMA

本小节实验[源码](01_pallas_core_exchange.py)、[输出](01_pallas_core_exchange.txt)。

两个 TensorCore 交换数据：每个 TensorCore 先把输入 `x[core]` 读进自己的 `send_vmem`，再写进对方的 `recv_vmem`。相对于第 1 节的两 TensorCore kernel，`CompilerParams` 多了 `collective_id=1`，kernel 主体为：

```python
core = jax.lax.axis_index('tc')
pltpu.async_copy(x_hbm.at[core], send_vmem, sems.at[0]).wait()

ready = pltpu.get_barrier_semaphore()
pl.semaphore_signal(ready, 1, device_id={'tc': 1 - core})
pl.semaphore_wait(ready, 1)

transfer = pltpu.make_async_remote_copy(send_vmem, recv_vmem, sems.at[1], sems.at[2], device_id={'tc': 1 - core}, device_id_type=pl.DeviceIdType.MESH)
transfer.start()
transfer.wait_send()
transfer.wait_recv()
pltpu.async_copy(recv_vmem, o_hbm.at[core], sems.at[3]).wait()
```

逐项说明：

- `pltpu.get_barrier_semaphore()` 取得一个专门用于汇合的信号量。使用它必须在 `CompilerParams` 中给出 `collective_id`，同一程序中不同的集合操作用不同的编号。
- `pl.semaphore_signal(sem, n, device_id=...)` 给指定对象的信号量加 n。`device_id={'tc': 1 - core}` 指定另一个 TensorCore；省略 `device_id` 时给自己的信号量加。
- `pl.semaphore_wait(sem, n)` 等到自己的信号量不小于 n，再减去 n。
- `pltpu.make_async_remote_copy(源, 目的, 发送信号量, 接收信号量, device_id=...)` 创建一次 remote DMA：源是自己的 Ref，目的 Ref 指的是对方 TensorCore 上同一个 scratch buffer。`wait_send()` 等到数据已经从源发出，源 buffer 可以复用；`wait_recv()` 等到对方写给自己的数据已经到达。

为什么要先汇合再发送？remote DMA 直接写对方的 `recv_vmem`，并给对方的接收信号量加数。若对方还没有开始执行这个 kernel，它的 TC VMEM 中可能还是上一个 kernel 的数据，信号量也可能还没有初始化；先等到双方都进入 kernel，写入才是安全的。结果正确：输出的第 0 块来自 TensorCore 1，第 1 块来自 TensorCore 0。

## 清单中的同步与 remote DMA

汇合在清单中是：

```text
{ s0: ssub.s32 s23, 1, s6 }                       # 对方编号 1 − core
{ s0: sand.u32 s24, 0x3, s23 }
{ s0: sshll.u32 s25, s24, 0x10 }
{ s0: sadd.s32 s26, 131072, s25 }
{ s0: sshrl.u32 s27, s26, 0x2 }                   # (2 + 对方编号) << 14
{ s0: sor.u32 s28, 0x8, s27 }                     # 低位是信号量的编号 8
{ misc: vsyncadd.remote.s32 [sflag:s28], 1 }      # semaphore_signal：给对方的 sflag 8 加 1
{ misc: vsyncadd.s32 [sflag:8], -1 }              # semaphore_wait：先减 1
{ misc: vwait.ge [sflag:8], 0 }                   #                再等它回到 0 以上
```

barrier 信号量就是 `sflag:8`。地址的拼法与第 1 节编译器插入的汇合相同，只是 `device_id` 没有指定芯片，芯片字段留作 0。`semaphore_wait(ready, 1)` 的写法是先减 1、再等它不小于 0：对方的信号先到时，计数从 1 减到 0，立即通过；后到时，计数先变成 −1，等对方加 1。这与第 1 节编译器插入的汇合是同一组指令。

remote DMA 是一条 `dma.general`：

```text
{ s0: sshll.u32 s29, s23, 0x1a ; s1: sshll.u32 s30, s23, 0xd }     # 对方编号 << 26；对方编号 << 13
{ s0: sadd.s32 s2, 134217728, s29 ; s1: sadd.s32 s3, 16384, s30 }  # (2 + 对方编号) << 26；(2 + 对方编号) << 13
{ s0: sor.u32 s4, 0x80008000, s2 ; s1: sor.u32 s0, 0x36, s3 }      # ici_dest；dst_flag = … | 54
{ s0: sand.u32 s5, 0xfffff000, s4 ; s1: simm.s32 s7, 8 }
{ s0: simm.s32 s8, 53 }
{ s0: dma.general [vmem:s7], [vmem:s22], length=8, stride_descriptor=[smem:0x0], stride_count=0,
      src_flag=[sflag:s8], dst_flag=[sflag:s0], ici_dest=s5 }
{ misc: vwait.ge [sflag:53], 8 }                  # wait_send：等自己的发送信号量
{ misc: vwait.ge [sflag:54], 8 }                  # wait_recv：等自己的接收信号量
```

源和目的都是 `[vmem:...]`，目的地址指的是对方 TensorCore 的 TC VMEM。`src_flag` 是发送信号量（`sflag:53`），数据发出后它在本方加 8；`dst_flag` 是对方的接收信号量 `(2 + 对方编号) << 13 | 54`，数据到达后它在对方加 8。`ici_dest` 是 `0x80008000 | (2 + 对方编号) << 26`，低 12 位留给芯片编号，这里为 0。三处都用“2 + TensorCore 编号”表示目标 TensorCore，只是所在的位不同。双方执行同样的指令，所以每方的接收信号量 `sflag:54` 由对方的 DMA 填满。第一章第 3 节 HBM → HBM 复制时出现过的 `dma.general` 与 `ici_dest`，在这里有了用途：它们描述的是目的地在哪个 TensorCore、哪颗芯片。

> 暂且可以理解为：`ici_dest` 编码了目的 TensorCore 所在的芯片与编号，同一颗芯片内与跨芯片使用同样的格式。第 5、6 节会看到跨芯片的情况，并用 tpuasm 改写其中的字段。

## 开销

[tpu-v4-latency-numbers](../../../tpu-v4-latency-numbers/results/04_remote.md) 测得，同一颗芯片内 TC VMEM → 另一 TensorCore 的 TC VMEM（K 为 KiB 数，周期数从发起到确认完成）：

| 情况 | 周期数 |
| --- | --- |
| 单向，一个 DMA | `414.8 + 1.014K` |
| 双向同时传输，每个方向 | `414.3 + 1.998K` |
| 往返（去程到达后对方再发回，见 [results/08_rtt.md](../../../tpu-v4-latency-numbers/results/08_rtt.md)） | `752.2 + 2.039K` |

单向时的带宽与 HBM → TC VMEM 相当（每 KiB 约 1 个周期），固定开销稍小。两个方向同时传输时，每个方向的每 KiB 开销翻倍，两个方向合计的带宽与单向相同。

## 显式的生产者—消费者同步

本小节实验[源码](02_pallas_producer_consumer.py)、[输出](02_pallas_producer_consumer.txt)。

不用 remote DMA，两个 TensorCore 也可以通过共享的 HBM 交换数据，但顺序要自己保证。TensorCore 0 计算 `2x` 写进 HBM，TensorCore 1 读它、加 1、写回：

```python
scratch_types=(pltpu.VMEM((8, 128), x.dtype), pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.REGULAR),
...
@pl.when(core == 0)
def _() -> None:
    pltpu.async_copy(x_hbm, vmem, sem).wait()
    vmem[...] = vmem[...] * 2.0
    pltpu.async_copy(vmem, o_hbm.at[0], sem).wait()
    pl.semaphore_signal(ready, 1, device_id={'tc': 1})

@pl.when(core == 1)
def _() -> None:
    pl.semaphore_wait(ready, 1)
    pltpu.async_copy(o_hbm.at[0], vmem, sem).wait()
    vmem[...] = vmem[...] + 1.0
    pltpu.async_copy(vmem, o_hbm.at[1], sem).wait()
```

`pltpu.SemaphoreType.REGULAR` 是普通的信号量，与 DMA 信号量一样每个 TensorCore 各有一份，`semaphore_signal` 可以给对方的那一份加数。TensorCore 0 在写 HBM 的 DMA 等到之后才发信号，保证 TensorCore 1 收到信号时数据已经在 HBM 中。

| 写法 | TensorCore 1 的结果出错（200 次中） |
| --- | ---: |
| 等待信号 | 0 次 |
| 不等待（删掉 `semaphore_wait`） | 200 次 |

不等待时，TensorCore 1 在 TensorCore 0 写入之前就读了 HBM，200 次全错。这与上一节 CMEM 的实验是同一条规则：一方写、另一方读，必须有一次“写完之后才发出、收到之后才读”的同步。

## 小结

两个 TensorCore 之间有三种协作通路：

| 通路 | 数据经过 | 同步方式 |
| --- | --- | --- |
| 共享 HBM | HBM | 信号量（本节） |
| 共享 Megacore Shared CMEM | CMEM | 信号量或汇合（第 3 节） |
| remote DMA | 直接写进对方 TC VMEM | DMA 自带发送与接收信号量，开始前需汇合（本节） |

选择哪一种，取决于数据要给谁、给几次：只给对方一次，remote DMA 最直接；两个 TensorCore 都要反复读的数据，放进 CMEM。
