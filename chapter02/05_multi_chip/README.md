# 多芯片：shard_map、拓扑与 ICI

从这一节开始，程序跨越多颗芯片。芯片之间通过 ICI（芯片间互联）直接相连，上一节的 remote DMA 和信号量换一个 `device_id` 就能跨芯片使用。本节先看芯片在 JAX 中怎样编号、怎样排进 mesh，再看跨芯片 remote DMA 的代价，以及为什么 mesh 中的顺序不等于物理上的相邻关系。

## 本 host 的四颗芯片

本小节实验[源码](01_devices_and_mesh.py)、[输出](01_devices_and_mesh.txt)。

实验在 `import jax` 之前调用 `tpu_init.initialise_local_chips()`，它与 `initialise_one_chip()` 的区别只在环境变量：

```python
os.environ['TPU_CHIPS_PER_PROCESS_BOUNDS'] = '2,2,1'
os.environ['TPU_PROCESS_BOUNDS'] = '1,1,1'
os.environ['TPU_VISIBLE_CHIPS'] = '0,1,2,3'
```

本机是 TPU v4 (2x2x2) 中的一个 host，它直连四颗芯片。这样设置后，即使整个切片有两个 host，本进程也只打开本 host 的四颗，组成一个 `2x2x1` 的切片，不需要 `jax.distributed.initialize()`。

```text
id=0，coords=[0, 0, 0]
id=1，coords=[1, 0, 0]
id=2，coords=[0, 1, 0]
id=3，coords=[1, 1, 0]
jax.make_mesh((4,), ...) 中第 i 个位置的 device id： [0, 2, 1, 3]
```

`device.coords` 是芯片在 ICI 网格中的坐标。四颗芯片排成 2×2 的方格，坐标只差一个分量的两颗芯片之间有直接的链路：0–1、0–2、1–3、2–3 相邻，而 0–3、1–2 在对角线上，数据要经过一颗中间芯片转发，即两跳。

`jax.make_mesh((4,), ('device',))` 把四颗芯片排成一维 mesh，顺序是 device 0、2、1、3。这个顺序对 XLA 的分片是合理的，但它不是一个物理上的环：mesh 中相邻的 2→1、以及首尾 3→0，都是对角线。

## 写法：跨芯片的 shard_map 与 remote DMA

本小节实验[源码](02_pallas_ring_shift.py)、[输出](02_pallas_ring_shift.txt)。

四颗芯片组成环，每颗芯片的 TensorCore 0 把自己的 `f32[8,128]` 发给环上的下一颗。相对于第一章的写法，变化在 `shard_map` 和 `device_id`：

```python
@jax.shard_map(
    mesh=mesh,
    in_specs=P('device'),
    out_specs=P('device'),
    check_vma=False,
)
def shift(x):
    ...
    def kernel(x_hbm, o_hbm, send_vmem, recv_vmem, sems):
        me = jax.lax.axis_index('device')
        ...
        transfer = pltpu.make_async_remote_copy(send_vmem, recv_vmem, sems.at[1], sems.at[2], device_id={'device': (me + 1) % 4, 'tc': 0}, device_id_type=pl.DeviceIdType.MESH)
```

- `in_specs=P('device')` 把输入 `f32[4,8,128]` 沿第 0 维切成四块，每颗芯片的 kernel 收到其中一块 `f32[1,8,128]`；`out_specs` 同理拼回。
- `jax.lax.axis_index('device')` 是本芯片在 mesh 的 `'device'` 轴上的位置，不是 `device.id`。
- `device_id={'device': ..., 'tc': 0}` 用 mesh 中的位置指定目标芯片及其上的 TensorCore；`device_id_type=pl.DeviceIdType.MESH` 表示按 mesh 位置解释。同一颗芯片内的 remote DMA（第 4 节）只是 `'device'` 不变、`'tc'` 不同。

mesh 的顺序决定了“下一颗”是谁。用 `jax.make_mesh` 的顺序，环是 device 0→2→1→3→0；直接用 device 列表构造 mesh、按物理相邻排列，环是 0→1→3→2→0：

```python
mesh = jax.sharding.Mesh(np.array([devices[i] for i in [0, 1, 3, 2]]), ('device',))
```

清单中跨芯片的 remote DMA 与第 4 节同一颗芯片内的完全相同，仍是一条 `dma.general`，只是 `ici_dest` 寄存器中的目标不同：

```text
{ s0: dma.general [vmem:s3], [vmem:s13], length=8, ..., src_flag=[sflag:s4], dst_flag=[sflag:s30], ici_dest=s2 }
```

## 跨芯片的代价

[tpu-v4-latency-numbers](../../../tpu-v4-latency-numbers/results/04_remote.md) 测得，TC VMEM → 另一颗芯片的 TC VMEM（K 为 KiB 数，周期数从发起到确认完成）：

| 情况 | 周期数 |
| --- | --- |
| 相邻芯片（一跳） | `1941.1 + 24.77K` |
| 对角线芯片（两跳） | `2908.6 + 24.77K` |
| 同一颗芯片的另一个 TensorCore（第 4 节） | `414.8 + 1.014K` |

与芯片内相比，跨芯片的固定开销大约是 4–7 倍，每 KiB 开销是 24 倍：一条链路单向约 43 GB/s，而 HBM 约 1 TB/s。两跳比一跳多约 970 个周期的固定开销，每 KiB 开销相同。

[研究报告 31](../../../pallas-tpu-readings-dev/research_reports/31_tpu_v4_remote_dma_topology.md) 进一步测得：两条流同向经过同一对芯片时共享一条链路，合计只有约 43 GB/s；经过不同链路或反向的两条流各自满速，合计约 84 GB/s。所以链路的分配与芯片的计算一样，是需要设计的资源。

## 只改 mesh 顺序：环的每一步是一跳还是两跳

实验在 kernel 内部把“四颗芯片汇合、每颗发出 4 KiB、等待收到”重复 32 轮和 64 轮，用主机计时之差除以 32 得到每轮的时间：

| mesh 顺序 | 环上每一步的跳数 | 每轮 |
| --- | --- | ---: |
| `jax.make_mesh`：0, 2, 1, 3 | 1, 2, 1, 2 | 约 2.99 µs |
| 物理环：0, 1, 3, 2 | 1, 1, 1, 1 | 约 2.43 µs |

两种顺序的数值都正确，但按 `jax.make_mesh` 的顺序成环时，有两步要走对角线，每轮慢约 0.56 µs。每一轮中的汇合是四颗芯片两两互发信号，两种顺序都相同；差别全部来自发送那一步的跳数。

设计跨芯片的通信时，应当按 `device.coords` 安排谁和谁通信，而不是按 mesh 中的序号。

## 两个 host：TPU v4 (2x2x2)

本小节实验[源码](03_devices_2x2x2.py)、[输出](03_devices_2x2x2.txt)。

整个切片有两个 host、八颗芯片。程序要在每个 host 上各运行一个进程，用 `podrun` 启动：

```sh
podrun -- /srv/workspace/venv/bin/python chapter02/05_multi_chip/03_devices_2x2x2.py
```

每个进程在访问设备之前都要调用 `jax.distributed.initialize()`，进程之间才能互相发现：

```text
进程数 2，全部 device 8 个，本进程的 device 4 个
id=4，coords=[0, 0, 1]，process_index=1
...
jax.make_mesh((8,), ...) 的 device id： [0, 4, 2, 6, 1, 5, 3, 7]
```

另一个 host 的四颗芯片坐标的第三个分量为 1，与本 host 的四颗沿 z 方向一一相邻。`jax.make_mesh` 的八元顺序先沿 z，再沿 y、x 排列。跨 host 的芯片之间同样是 ICI 直连，remote DMA 的写法不变，只是两个进程各自负责自己 host 上的芯片。

> 暂且可以理解为：两个 host 上的 TensorCore 计数器互不同步，跨 host 比较时间不能直接相减。第三章第 7 节介绍怎样用全局时钟 GTC 对齐不同芯片上的时间。
