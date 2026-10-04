# 跨芯片的 Megacore Shared CMEM

第 5 节的 remote DMA 是 TC VMEM → 另一颗芯片的 TC VMEM。DMA 引擎还能把一颗芯片 Megacore Shared CMEM 中的数据直接写进另一颗芯片的 CMEM，不经过任何 TensorCore 的 TC VMEM。公开的 Pallas 接口不能表达这条通路（第 3 节：Mosaic 不允许在 TPU v4 上分配 CMEM buffer），本节用 tpuasm 改写一个 TC VMEM 版本的 kernel 来实现它。这个例子还说明：改写 DMA 时，除了指令上看得见的端点，还要弄清寄存器中编码的目标。

## 载体

本小节实验[源码](01_tpuasm_cmem_to_cmem.py)、[输出](01_tpuasm_cmem_to_cmem.txt)。

载体是两颗相邻芯片（device 0 与 1）交换数据的 kernel，写法与第 5 节的环相同，只是环上只有两颗芯片：

```python
me = jax.lax.axis_index('device')
pltpu.async_copy(x_hbm.at[0], send_vmem, sems.at[0]).wait()
ready = pltpu.get_barrier_semaphore()
pl.semaphore_signal(ready, 1, device_id={'device': 1 - me, 'tc': 0})
pl.semaphore_wait(ready, 1)
transfer = pltpu.make_async_remote_copy(send_vmem, recv_vmem, sems.at[1], sems.at[2], device_id={'device': 1 - me, 'tc': 0}, device_id_type=pl.DeviceIdType.MESH)
transfer.start()
transfer.wait_send()
transfer.wait_recv()
pltpu.async_copy(recv_vmem, o_hbm.at[0], sems.at[0]).wait()
```

清单中与 remote DMA 有关的两条指令：

```text
{ s0: sor.u32 s21, 0x88008000, s17 }
{ s0: dma.general [vmem:s23], [vmem:s9], length=8, ..., src_flag=[sflag:s24], dst_flag=[sflag:s0], ici_dest=s21 }
```

`s17` 是目标芯片的编号，`ici_dest` 寄存器 `s21` 由它与常数 `0x88008000` 拼成。

## 目标写在两个地方

只把 `dma.general` 的两端从 `[vmem:...]` 改成 `[cmem:...]` 是不够的。[tpu-v4-latency-numbers](../../../tpu-v4-latency-numbers/README.md#44-megacore-shared-cmem-的远端目标不能沿用-tc-override) 对 libtpu `0.0.49` 的逆向表明，编译器为 TC VMEM 目标生成 `ici_dest` 时，在其 bits 28:26 写入 `2 + TensorCore 编号`，指定由目标芯片上的哪个 TensorCore 接收；目标是 CMEM 时，这个字段保持为 0。只改指令端点而保留这个字段，接收芯片会异常停机（`RuntimeUnexpectedCoreHalt`）。

`0x88008000` 中的 `0x08000000` 正是 bits 28:26 = 2，即 TensorCore 0。所以改写有两处：

```text
{ s0: sor.u32 s21, 0x80008000, s17 }                                           # 清除 TensorCore 字段
{ s0: dma.general [cmem:s26], [cmem:s25], length=8, ..., ici_dest=s21 }       # 两端改为 CMEM
```

此外，发送前要把数据从 TC VMEM 搬进本芯片的 CMEM，收到后再从 CMEM 搬回 TC VMEM，以便沿原来的路径写回 HBM。这两段插在汇合之前和输出 DMA 之前：

```text
# 汇合之前：本芯片 CMEM 地址 0 放要发送的数据，地址 64 作为接收区
{ s0: simm.s32 s25, 0 ; s1: simm.s32 s26, 64 }
{ s0: dma.simple [cmem:s25], [vmem:s9], length=8, dst_flag=[sflag:52] }
{ misc: vwait.ge [sflag:52], 8 }
{ misc: vsyncadd.s32 [sflag:52], -8 }
...
# 输出之前：接收区搬回 TC VMEM 的接收 buffer
{ s0: dma.simple [vmem:s23], [cmem:s26], length=8, dst_flag=[sflag:52] }
{ misc: vwait.ge [sflag:52], 8 }
{ misc: vsyncadd.s32 [sflag:52], -8 }
```

改写后，数据经 TC VMEM → 本芯片 CMEM → 对方 CMEM → 对方 TC VMEM，20 组不同输入的结果都正确。接收信号量 `dst_flag` 不用改：数据写进对方的 CMEM 后，对方 TensorCore 0 的接收信号量照样增加，`wait_recv()` 正常完成。

## 用 edit_bundles 只改需要改的部分

改写用 [`tpuasm_tools.edit_bundles`](../../tpuasm_tools.py)：

```python
patched = tpuasm_tools.edit_bundles(serialized, {
    remote_pc: ('dma.general [vmem:s23], [vmem:s9]', 'dma.general [cmem:s26], [cmem:s25]'),
    ici_pc: ('sor.u32 s21, 0x88008000, s17', 'sor.u32 s21, 0x80008000, s17'),
})
patched = tpuasm_tools.insert_bundles(patched, {barrier_pc: STAGE, output_pc: UNSTAGE})
```

`edit_bundles` 在逐字节精确的清单上工作：只有被修改的 bundle 重新编码，其余 bundle 与编译器生成的完全相同。

本小节实验[源码](02_tpuasm_canonical_reencode.py)、[输出](02_tpuasm_canonical_reencode.txt)。

为什么要强调“其余完全相同”？同一条指令往往有不止一种机器编码。例如一个常数可以放进 bundle 的几个立即数槽中的任意一个，`vwait.ge [sflag:52], 8` 的阈值可以直接写在指令里，也可以引用立即数槽。清单的文本只写出指令，写不出这些选择。tpuasm 的清单有两种格式：`encoding='exact'` 在需要时附上 `.encoding` 约束，记录编译器的选择，重新汇编后逐字节还原，例如

```text
{ misc: vwait.ge [sflag:52], 8 ; .encoding { misc.value = imm2 } }
```

`encoding='canonical'` 不带约束，由汇编器为每个 bundle 重新选择编码。实验把载体的清单不做任何修改，分别以两种格式重新汇编后运行：

```text
encoding='exact'：结果正确：True
encoding='canonical'：canonical 编码改变了 99 / 970 个 bundle 的机器字节；结果正确：True
```

canonical 格式改变了 99 个 bundle 的字节，结果仍然正确：改变的只是编码的选择，指令的含义不变。

但这依赖汇编器对每一条指令的理解都完整、正确。只要有一条指令的某个字段没有体现在清单文本里，canonical 格式就会在重新汇编时把它丢掉，而且出错的可能是与改写毫无关系的 runtime 代码。本教程写作时就遇到过一次：tpuasm 曾把 mask 寄存器之间的移动（例如 `vm4 ← vm7`）显示成 `misc: vnop`，目的和源寄存器只留在 `.encoding` 约束里。canonical 格式丢掉约束后，runtime 代码中的一次 mask 移动变成了真正的空操作，这个实验因此挂起，直到在被改变的一百来个 bundle 中对半排查，才定位到这一个 bundle。逐字节精确的清单没有这个风险：没有被修改的 bundle 根本不经过重新编码。所以改写已编译的程序时，默认用精确格式，只改动必须改动的 bundle，其余部分保持编译器生成的原样；`edit_bundles` 和 `insert_bundles` 都是这样做的。

## 开销

[tpu-v4-latency-numbers](../../../tpu-v4-latency-numbers/results/04_cmem.md) 测得，相邻芯片之间 CMEM → CMEM 的 remote DMA 为 `1840.7 + 24.77K` 个周期（K 为 KiB 数），比 TC VMEM → TC VMEM（`1941.1 + 24.77K`）的固定开销少约 100 个周期，每 KiB 开销相同：带宽由 ICI 链路决定，与两端是哪种内存无关。

这条通路的价值不在于更快，而在于数据不必经过 TensorCore：两颗芯片可以直接在共享的片上内存之间交换数据，两颗芯片上的两个 TensorCore 都能从各自的 CMEM 读到它（第 3 节），而 TC VMEM 的容量留给计算。
