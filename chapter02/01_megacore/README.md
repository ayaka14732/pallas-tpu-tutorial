# Megacore：一颗芯片两个 TensorCore

第一章的 kernel 都只用一个 TensorCore，另一个读到自己的编号后跳过 kernel 主体。本节让两个 TensorCore 一起工作：看清 Pallas 怎样把工作分给它们，机器清单中多了哪些同步，哪些同步可以去掉；最后看另一种组织方式，让 runtime 把两个 TensorCore 当作两个 device。

## 一颗芯片，一个 device，两个 TensorCore

TPU v4 的一颗芯片有两个 TensorCore。默认情况下，runtime 把一颗芯片作为一个 device 交给 JAX，`device.num_cores` 为 2；这种组织方式称为 Megacore。两个 TensorCore 执行同一份程序，各有自己的 TC VMEM、TC VREG、SMEM 和标量寄存器，共享这颗芯片的 HBM。第一章第 2 节的清单中，`sld s6, [smem:0x1]` 读出的就是当前 TensorCore 的编号 0 或 1。

原生 XLA 会自动利用这一点。第一章的 XLA baseline 为了只用一个 TensorCore，特意换了一种 runtime 模式（第一章第 1 节）；回到默认模式，同样的 `f32[64,128] × 2`（[源码](04_jax_f32_64x128.py)、[输出](04_jax_f32_64x128.txt)）编译出的 fusion 中，DMA 的长度从 64 变成 32，`vld`、`vmul`、`vst` 从各 8 条变成各 4 条，开头多了两条指令：

```text
{ s1: sld s6, [smem:0x1] }
{ s0: sshll.u32 s7, s6, 0x5 }
```

XLA 读出 TensorCore 编号后乘以 32（左移 5 位），作为 HBM 地址的 granule 偏移：TensorCore 0 处理第 0–31 行，TensorCore 1 处理第 32–63 行。清单中的计数是每个 TensorCore 各执行一份。

## 写法：num_cores=2 与 axis_index

本小节实验[源码](01_pallas_two_cores.py)、[输出](01_pallas_two_cores.txt)。

相对于第一章第 1 节的最小 kernel，输入改为 `f32[64,128]`，TensorCore mesh 改为两个，每个 TensorCore 只处理自己的 32 行：

```python
tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=2)
...
rows = x.shape[0] // num_cores
scratch_types=(pltpu.VMEM((rows, x.shape[1]), x.dtype), pltpu.SemaphoreType.DMA),
...
def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, sem: Ref) -> None:
    core = jax.lax.axis_index('tc')
    window = pl.ds(core * rows, rows)
    pltpu.async_copy(x_hbm.at[window], x_vmem, sem).wait()
    x_vmem[...] = x_vmem[...] * 2.0
    pltpu.async_copy(x_vmem, o_hbm.at[window], sem).wait()
```

- `jax.lax.axis_index('tc')` 返回当前 TensorCore 在 `TensorCoreMesh` 中的编号，是一个运行时的标量。两个 TensorCore 执行同一个 kernel 函数，只有这个值不同。
- `scratch_types` 中的 buffer 每个 TensorCore 各有一份，位于各自的 TC VMEM。`pltpu.VMEM((32, 128), ...)` 在两个 TensorCore 上都分配到地址 0，但它们是两块不同的内存，互不可见。
- 输入输出 Ref 指向共享的 HBM，两个 TensorCore 按编号读写其中不重叠的窗口。

| `num_cores` | kernel 段 bundle 数 | 每个 TensorCore 的 `vld`/`vmul`/`vst` | `vsyncadd.remote` |
| ---: | ---: | --- | ---: |
| 1 | 37 | 8 / 8 / 8 | 1 |
| 2 | 43 | 4 / 4 / 4 | 2 |

每个 TensorCore 的向量指令减半，与 XLA 的分工相同。窗口起点就是 `axis_index` 乘以 32：

```text
{ s1: sld s6, [smem:0x1] }                  # TensorCore 编号
{ s0: sshll.u32 s18, s6, 0x5 }              # 编号 × 32
{ s0: sadd.s32 s21, s18, s0 }               # HBM 基址 + 偏移
{ s0: dma.simple [vmem:s22], [hbm:s21], length=32, dst_flag=[sflag:52] }
```

## 跨核汇合

两个 TensorCore 时，kernel 主体前后各有一段同样的同步：

```text
{ s1: sld s6, [smem:0x1] }                       # 自己的编号
{ s1: sld s7, [smem:0x0] }                       # 芯片的编号
{ s0: sadd.s32 s8, 1, s6 }
{ s0: sand.u32 s9, 0x1, s8 ; s1: sand.u32 s10, 0xfff, s7 }   # 对方的编号：(自己 + 1) mod 2
{ s0: sand.u32 s11, 0x3, s9 ; s1: sshll.u32 s13, s10, 0x12 } # 芯片编号 << 18
{ s0: sshll.u32 s12, s11, 0x10 }
{ s0: sadd.s32 s14, 131072, s12 }
{ s0: sshrl.u32 s15, s14, 0x2 }                  # (2 + 对方的编号) << 14
{ s0: sor.u32 s16, 0x2d, s15 }                   # 低位是同步标志的编号 45
{ s0: sor.u32 s17, s16, s13 }
{ misc: vsyncadd.remote.s32 [sflag:s17], 1 }     # 给对方的 sflag 45 加 1
{ misc: vwait.ge [sflag:45], 1 }                 # 等自己的 sflag 45 被对方加到 1
{ misc: vsyncadd.s32 [sflag:45], -1 }            # 减回 0
```

`vsyncadd.remote` 修改的不是自己的同步标志，而是另一个 TensorCore 的。它的地址操作数由三部分拼成：`芯片编号 << 18`、`(2 + TensorCore 编号) << 14` 和同步标志的编号。`131072` 是 `2 << 16`，加上 `对方的编号 << 16` 再右移 2 位，就是中间那一项；TensorCore 0 时它是 `0x8000`，TensorCore 1 时是 `0xc000`。这里目标就在本芯片上，芯片编号取自己的；第 5 节换成别的芯片的编号，同一条指令就能给另一颗芯片上的 TensorCore 发信号。两个 TensorCore 各给对方加 1，再各等自己的标志变成 1：先到的一方停在 `vwait.ge`，直到后到的一方也执行了 `vsyncadd.remote`。这是一个两方的汇合（barrier）：执行完这一段时，两个 TensorCore 都已经到达这里。

`num_cores=1` 时，kernel 出口也有这样一段汇合，只是入口没有，因为 TensorCore 1 什么都不做。

> 暂且可以理解为：同步标志（sflag）是每个 TensorCore 上的一组计数器，DMA 完成时会给它们加数，`vwait` 等它们达到某个值；`vsyncadd.remote` 让一个 TensorCore 修改另一个 TensorCore 的计数器。第 4 节详细介绍怎样在 Pallas 中显式使用这类同步。

## 去掉入口汇合

本小节实验[源码](02_tpuasm_remove_entry_barrier.py)、[输出](02_tpuasm_remove_entry_barrier.txt)。

本例中两个 TensorCore 读写 HBM 中互不重叠的窗口，谁也不读对方写的数据，入口的汇合保护不了任何东西。用 tpuasm 把入口的三条同步指令换成 `vnop`，保留出口的汇合，连续运行 100 次，结果全部正确。

这次改写用 [`tpuasm_tools`](../../tpuasm_tools.py) 中按 bundle 编号定位的两个函数，因为同样的三条指令在 kernel 出口和 runtime 代码中也出现了，不能按全文第一次出现来替换：

```python
pc = tpuasm_tools.find_bundles(serialized, 'vsyncadd.remote.s32')[0]   # kernel 中第一次出现的 bundle
edits[pc] = ('misc: vsyncadd.remote.s32 [sflag:s17], 1', 'misc: vnop')
...
patched = tpuasm_tools.edit_bundles(serialized, edits)
```

`find_bundles` 只在编译器归属到 Pallas kernel 的 bundle 中查找；`edit_bundles` 只替换指定 bundle 内的文本，同一 bundle 其他槽的指令保持不变。第三条 `vsyncadd.s32 [sflag:45], -1` 与 `sshll`、`simm` 共用一个 bundle，正需要这样只改一条。

去掉入口汇合省去了一次等待对方到达的时间。两个 TensorCore 几乎同时开始时，这段等待很短；但只要一方因为别的原因晚到，另一方就要空等。

> 暂且可以理解为：跨核汇合的代价取决于两个 TensorCore 谁先到、差多少；第三章第 6 节介绍如何在两个 TensorCore 上同时计时，测出这种差距。

反过来，如果一个 TensorCore 要读另一个写入 HBM 的数据，就必须在读之前确认对方已经写完。删掉同步的前提是证明两方之间没有这样的依赖。[研究报告 20](../../../pallas-tpu-readings-dev/research_reports/20_pallas_tpu_v4_megacore_entry_barrier_patch.md) 用一个生产者—消费者的反例说明了这一点：删掉 XLA 插入的全部隐式汇合后，消费者会读到尚未写入的零。需要这种顺序时，应在 Pallas 中显式写出同步（第 4 节），而不是依赖编译器插入的汇合。

## 另一种组织方式：split-chip

本小节实验[源码](03_pallas_split_chip.py)、[输出](03_pallas_split_chip.txt)。

第一章的 XLA baseline 已经用过 runtime 的另一种模式：在 TPU 初始化之前，给 libtpu 传入参数 `--deepsea_chip_config_name=legacy`，一颗芯片就作为两个 device 出现，每个 device 只有一个 TensorCore。那里只用了第 0 个 device；这里把两个都用上：

```python
tpu_init.initialise_one_chip()
os.environ['LIBTPU_INIT_ARGS'] = f"{os.environ.get('LIBTPU_INIT_ARGS', '')} --deepsea_chip_config_name=legacy".strip()
```

```text
devices： [(0, [0, 0, 0], 0, 1), (1, [0, 0, 0], 1, 1)]     # (id, 芯片坐标, core_on_chip, num_cores)
```

两个 device 的芯片坐标相同，`core_on_chip` 分别为 0 和 1。此时 `jax.make_mesh` 拒绝创建 mesh（`Creating meshes for TPU >v3 requires one device per chip`），要直接用 device 列表构造：

```python
mesh = jax.sharding.Mesh(np.array(devices), ('device',))
```

工作的划分交给 `shard_map`：`in_specs=P('device')` 把输入沿行切成两半，每个 device 拿到 `f32[32,128]`，kernel 本身与第一章的单 TensorCore 版本完全相同，`TensorCoreMesh` 用 `num_cores=1`。清单中 kernel 段只有 16 个 bundle：没有读取 TensorCore 编号，也没有任何跨核汇合，因为在 runtime 看来，这两个 TensorCore 是两个独立的 device。

两种方式各有用处：

- Megacore：一个 kernel 内部可以让两个 TensorCore 分工，并且在第 3、4 节中看到，它们可以共享 Megacore Shared CMEM、相互发起 DMA。代价是编译器插入的跨核汇合。
- split-chip：每个 TensorCore 的程序最简单，适合两个 TensorCore 完全独立的工作；它们之间的协作要像两个 device 一样，通过集合通信完成（第 5–7 节）。

这个参数是 libtpu 的运行时模式，不是公开稳定的接口，换 libtpu 版本后需要重新确认 `device.num_cores`。除第一章的 XLA baseline 外，本教程都使用默认的 Megacore 模式。
