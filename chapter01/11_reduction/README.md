# 归约

归约把一个轴上的元素合并成一个：求和、求最大值。在 TC VREG 中，沿 lane 和沿 sublane 是两种完全不同的操作，前者由 XLU 完成，后者由 sublane 循环移位与逐元素运算拼成。本节逐一看这两个方向，再看跨多个 tile 的归约，以及归约结果怎样从向量单元进入标量单元。

## 写法：输出的 shape 由运算推出

本小节实验[源码](01_pallas_reduction.py)、[输出](01_pallas_reduction.txt)；kernel 的构造在 [reduction_common.py](reduction_common.py) 中。

相对于第 1 节的最小 kernel，计算换成一个归约函数 `f`，输出的 shape 用 `jax.eval_shape` 从 `f` 推出：

```python
out = jax.eval_shape(f, jax.ShapeDtypeStruct(x.shape, x.dtype))
@pl.kernel(
    out_type=pltpu.HBM(out.shape, out.dtype),
    ...
)
def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, o_vmem: Ref, sem: Ref) -> None:
    pltpu.async_copy(x_hbm, x_vmem, sem).wait()
    o_vmem[...] = f(x_vmem[...])
    pltpu.async_copy(o_vmem, o_hbm, sem).wait()
```

这里的 `out_type` 不是 `jax.ShapeDtypeStruct`，而是 `pltpu.HBM(shape, dtype)`：它明确要求输出放在 HBM。第 1 节说过，kernel 输入输出的默认内存空间是 `ANY`，由 XLA 决定放在哪里。对于 `f32[8,1]` 这样又窄又小的输出，XLA 会把它放进 Megacore Shared CMEM，kernel 中的 DMA 目的地随之变成 `[cmem:...]`（实验[源码](03_pallas_output_memory_space.py)、[输出](03_pallas_output_memory_space.txt)）。

> 暂且可以理解为：Megacore Shared CMEM 是芯片上一块两个 TensorCore 共享的片上内存，容量比 TC VMEM 大，比 HBM 近。第二章第 3 节详细介绍它。

本章只研究单个 TensorCore 与 HBM、TC VMEM 之间的操作，所以这里固定为 HBM。另外，`f32[8,1]` 在 HBM 中的 layout 与 kernel 写出的不同，XLA 会在 kernel 之后追加一段重排；本节的统计只看 kernel 本身（`kernel_listing(compiled, pallas_only=True)`）。

## 沿 lane：XLU 的跨 lane 归约

`jnp.sum(x, axis=1, keepdims=True)` 把 `f32[8,128]` 的每一行加成一个数：

```text
{ vx0: vadd.xlane.0.8x128.f32 trf0, v0 }
{ vr0: vpop.8x128 v3, trf0 }
{ vst: vst.msk.8x128 [vmem:0x8], vm0, v3 }
```

一条 `vadd.xlane` 把一个 TC VREG 送入 XLU，XLU 对每个 sublane 的 128 个 lane 求和，结果经 `trf0` 取回。随后的掩码 store 只写每行的第 0 列（掩码 `vm0` 由 `vlaneseq`、`vand`、`veq` 生成，表示 lane == 0）。`max(x, axis=1)` 相同，指令换成 `vmax.xlane`。

bf16 输入（`bf16[16,128]`，先 `astype(jnp.float32)`）先 unpack 成两个 f32 TC VREG，各做一次 `vadd.xlane`。

### 结果在每个 lane 中

本小节实验[源码](04_tpuasm_xlane_layout.py)、[输出](04_tpuasm_xlane_layout.txt)。

上面的掩码 store 只写第 0 列，那么 `vpop` 取回的 TC VREG 中，各行的和放在哪里？用 tpuasm 直接执行一条 `vadd.xlane`，把取回的整个 TC VREG 写回主机（第三章第 4 节的 `LccProbe` 载体，`run_tiles` 取回片段写入 TC VMEM 的 tile）：

```python
body = bundle('vx0: vadd.xlane.0.8x128.f32 trf0, v10') + GAP + bundle('vr0: vpop.8x128 v11, trf0') + GAP + bundle('vst: vst.8x128 [vmem:0x8], v11') + GAP
```

输入第 s 行第 l 列是 `(128s + l) mod 7`，再在第 37 列加上 10s，使各行的和与最大值都不同：

```text
## vadd.xlane
  每行的期望值：[379.0, 393.0, 407.0, 414.0, 421.0, 435.0, 449.0, 449.0]
  取回的 TC VREG 第 0、1、127 列：[379.0, 393.0, ...]、[379.0, 393.0, ...]、[379.0, 393.0, ...]
  每行 128 个 lane 都等于该行的结果：True
```

`vmax.xlane` 相同。XLU 把每一行的结果广播到这一行的全部 128 个 lane。所以 `keepdims=True` 的 `f32[8,1]` 只是取第 0 列；如果下一步要用这个结果对同一行的每个元素做运算（例如减去最大值、除以和），它已经在每个 lane 中，不需要再广播。

### 时间

> 暂且可以理解为：`vadd.xlane` 是提交—取回的形式，从提交到能取回要等约 80 个周期，但连续的 `vadd.xlane` 每 8 个周期可以发射一条，多个 TC VREG 一起归约时等待被重叠。第三章第 6 节测得这两个数分别是 79 和 8 个周期。

## 沿 sublane：循环移位与逐元素运算

`jnp.sum(x, axis=0, keepdims=True)` 把 8 行加成一行。XLU 没有跨 sublane 的归约，编译器用第 8 节介绍的 `vrot.slane.down` 拼出一棵二叉树：

```text
4 × vrot.slane.down，vadd     # 与循环移动 4 行的自己相加
2 × vrot.slane.down，vadd     # 再与移动 2 行的相加
1 × vrot.slane.down，vadd     # 再与移动 1 行的相加
```

共 7 条 `vrot.slane.down`、3 条 `vadd.8x128.f32`，每个 sublane 都得到 8 行之和。第 8 节已经看到，`vrot.slane.down` 每次只移动一个 sublane，没有位移量操作数；所以树虽然只有 3 层，移位却要 4 + 2 + 1 = 7 条。

`vrot.slane.down` 与 `vadd` 一样在向量 ALU 槽（`va0`、`va1`）中执行，不经过 XLU，也不需要取回；第三章第 6 节测得它的结果 2 个周期后可用。树的每一层都依赖上一层，7 条移位与 3 条加法串成一条链，一个 TC VREG 约 17 个周期，比沿 lane 的 XLU 归约等待的时间短得多。

`jnp.max(x, axis=0)` 却没有用树：清单中是 7 条 `vrot.slane.down` 和 7 条 `vmax.8x128.f32`，每移动一行就取一次最大值。移位数相同，逐元素运算多了 4 条，而且 7 次 `vmax` 前后依赖，串成一条长链。

可以自己写出树形的版本：

```python
def f(x):
    for shift in shifts:
        x = jnp.maximum(x, pltpu.roll(x, shift, axis=0))
    return x[0:1, :]
```

| `shifts` | `vmax` | `vrot.slane.down` |
| --- | ---: | ---: |
| 编译器生成 | 7 | 7 |
| `(4, 2, 1)` | 3 | 17 |
| `(4, 6, 7)` | 3 | 7 |

位移取 `(4, 2, 1)` 时，`vmax` 少了，移位却多到 17 条。原因是 `pltpu.roll(x, k, axis=0)` 被降低成 `(8 - k) % 8` 条 `vrot.slane.down`（第 8 节中位移 3 是 5 条，位移 7 是 1 条），位移 2 和 1 分别要 6 条和 7 条。对求最大值来说，向上或向下移动并不影响结果，于是改用 `(4, 6, 7)`，得到 3 条 `vmax` 加 7 条移位，与编译器为求和生成的树一样。这个例子说明，即使写的是同一个算法，也要对照清单，确认每个高层操作落成了几条指令。

## 跨多个 tile：先逐元素，再 sublane

`f32[64,128]` 沿 axis=0 求和：8 个 tile 先用 7 条 `vadd` 逐元素加成一个 TC VREG，再在这个 TC VREG 内做上面的 sublane 树（3 条 `vadd`、7 条移位），共 10 条 `vadd`。跨 tile 的归约就是普通的逐元素运算，代价随 tile 数线性增长；只有最后一个 TC VREG 内部的归约需要移位。

## 归约到标量：从向量单元到标量单元

`jnp.sum(x, keepdims=True)` 把 `f32[8,128]` 加成一个数。先 `vadd.xlane` 沿 lane 求和，再用 sublane 树合并 8 行，之后出现了两条新指令：

```text
{ vst: vpush v2sf, v11 }      # 把 TC VREG 送进向量到标量的队列 v2sf
{ s0: spop s0, v2sf }         # 标量单元从 v2sf 取出一个数
{ va0: vmov.8x128 v15, s0 }   # 再广播回向量，写入输出
```

第 7 节说过，标量进入向量单元只需一条 `vmov`；反方向要经过队列 `v2sf`：向量单元用 `vpush` 送入，标量单元用 `spop` 取出。这条通路让向量的计算结果可以决定标量单元的行为，例如作为循环次数或分支条件。这里结果最终又要写回 TC VMEM，编译器仍然把它送进了标量单元，再广播回来。

> 暂且可以理解为：`vpush` 与 `spop` 之间要经过一段延迟，标量单元在数据到达之前停止发射。第三章第 6 节测得 `spop` 最早在 `vpush` 之后 43 个周期执行。

## 对照：原生 XLA

本小节实验[源码](02_jax_reduction.py)、[输出](02_jax_reduction.txt)。

XLA 的五个归约都先把输入预取到 Megacore Shared CMEM，再用 `cld` 读入 TC VREG，这是 XLA 的默认选择。计算部分：

| 归约 | XLA | Pallas |
| --- | --- | --- |
| `sum(x, axis=1)` | 1 次 `vxpose` 转置、24 条 `vadd`、17 条 `vpop`、7 条移位 | 1 条 `vadd.xlane` |
| `sum(x, axis=0)` | 8 条 `vadd`、7 条移位 | 3 条 `vadd`、7 条移位 |
| `max(x, axis=0)` | 8 条 `vmax`、7 条移位 | 7 条 `vmax`、7 条移位 |
| `sum(x)` | 1 条 `vadd.xlane`、8 条 `vadd`、7 条移位 | 1 条 `vadd.xlane`、3 条 `vadd`、7 条移位 |

XLA 的 sublane 归约用了 8 条逐元素运算，沿 lane 的求和则先转置、再沿 sublane 归约，没有用 `vadd.xlane`。归约是最常见的操作之一，读清单、选对硬件通路，就能比 XLA 的通用写法少很多指令。
