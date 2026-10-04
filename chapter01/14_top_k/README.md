# Top-k

Top-k 从每行中选出最大的 k 个值及其下标。它把第 11 节的跨 lane 归约推进了一步：不仅要知道最大值是多少，还要知道它在哪里，并且要重复 k 次。本节从 argmax 出发，看 XLU 怎样同时给出最大值和下标、并列时选哪一个，再看 k = 8 时的重复选取与它的时间、跨两个 lane tile 的情况，以及当前 Pallas 的 argmax 与 top-k 的限制和两个与 JAX 语义不符的结果。

## 写法：两个输出

本小节实验[源码](01_pallas_top_k.py)、[输出](01_pallas_top_k.txt)。

top-k 有两个输出：值与下标。相对于第 1 节的最小 kernel，`out_type` 是两个输出组成的 tuple，scratch 中多两个 TC VMEM buffer，计算一次写两个 Ref：

```python
out_type=(pltpu.HBM(values.shape, values.dtype), pltpu.HBM(indices.shape, indices.dtype)),
...
def kernel(x_hbm: Ref, v_hbm: Ref, i_hbm: Ref, x_vmem: Ref, v_vmem: Ref, i_vmem: Ref, sem: Ref) -> None:
    pltpu.async_copy(x_hbm, x_vmem, sem).wait()
    v_vmem[...], i_vmem[...] = f(x_vmem[...])
    pltpu.async_copy(v_vmem, v_hbm, sem).wait()
    pltpu.async_copy(i_vmem, i_hbm, sem).wait()
```

输出 Ref 按 `out_type` 的顺序排在输入之后。输出明确放在 HBM 的原因见第 11 节。

## 基本形式：argmax 是一次 XLU 操作

`jnp.max(x, axis=1, keepdims=True)` 与 `jnp.argmax(x, axis=1, keepdims=True)` 一起，输入 DMA 之后的清单是：

```text
{ va0: vlaneseq.8x128.u32 v1 ; vld: vld.8x128 v0, [vmem:0x0] }
{ va0: vand.8x128.u32 v2, 0x7f, v1 ; vx0: vmax.xlane.2.8x128.f32 trf0, v0 }   # v2 = lane 号；提交求最大值
{ va0: veq.8x128.s32 vm0, v2, 0 }                                            # vm0 = 第 0 个 lane
{ vx0: vmax.index.xlane.0.8x128.f32 trf0, v0 }                               # 提交求最大值的下标
{ vr0: vpop.8x128 v3, trf0 }                                                 # 取回最大值
{ vst: vst.msk.8x128 [vmem:0x8], vm0, v3 }                                   # 只写第 0 个 lane
{ vr0: vpop.8x128 v4, trf0 }                                                 # 取回下标
{ vst: vst.msk.8x128 [vmem:0x10], vm0, v4 }
```

XLU 有专门给出最大值下标的指令 `vmax.index.xlane`，不需要先求最大值、再比较、再找位置。两次提交进入同一个队列 `trf0`，按提交的顺序取回。与第 11 节的 `vadd.xlane` 一样，结果广播到整行的 128 个 lane；输出是 `[8,1]`，只占每行的第 0 个 lane，所以编译器用 `vlaneseq` 与 `0x7f` 相与得到 lane 号，比较出“第 0 个 lane”的掩码，再用带掩码的 `vst.msk` 只写这一个 lane。下标直接作为 int32 写出：`vmax.index.xlane` 助记符中的 `f32` 指输入的类型，结果是整数下标。值与下标都与 CPU 一致。

### vmax.index.xlane 的规则

本小节实验[源码](03_tpuasm_max_index.py)、[输出](03_tpuasm_max_index.txt)。

上面的输入每行互不相同。并列、NaN 时选哪个下标，要用 tpuasm 直接构造输入来看。实验用第三章第 3 节的 `LccProbe` 执行手写片段：先用 `vlaneseq`、`vand`、`vcvt`、`veq`、`vsel` 在 `v13` 中构造一种输入（每个 sublane 相同），再依次提交 `vmax.index.xlane`、`vmax.xlane` 和 `vmin.index.xlane`，取回写出。实验只改构造 `v13` 的那几条指令：

| 输入 | `vmax.index` | `vmax` | `vmin.index` |
| --- | ---: | ---: | ---: |
| 全为 0 | 127 | 0 | 127 |
| 全为 −inf | 127 | −inf | 127 |
| lane & 15（15 在 lane 15、31、…、127，0 在 lane 0、16、…、112） | 127 | 15 | 112 |
| −lane（最大值在 lane 0） | 0 | 0 | 127 |
| lane 5 与 lane 70 为 1，其余为 0 | 70 | 1 | 127 |
| lane 3 为 NaN，其余为 lane | 3 | NaN | 3 |
| lane 9 为 −0，其余为 +0 | 127 | +0 | 127 |

由此得到三条规则：

- **并列时取 lane 号最大的一个。** 最大值或最小值出现在多个 lane 时，`vmax.index` 与 `vmin.index` 都返回其中最大的 lane 号：lane & 15 的最大值 15 返回 127，最小值 0 返回 112。全部相等时返回 127。这与 `jnp.argmax` 的约定（取第一个）相反，而 Mosaic 直接用这条指令实现 `jnp.argmax`，没有做任何修正。第一个实验加了一组每行最大值在 lane 15、31、…、127 并列的输入：

  ```text
  ## max 与 argmax，每行的最大值在 lane 15、31、…、127 并列
    值与 CPU 结果一致：True；下标与 CPU 结果一致：False
    第 0 行下标：[127]；CPU：[15]
  ```

  所以在 Pallas kernel 中，`jnp.argmax` 遇到并列时返回最后一个下标，与 JAX 的语义不同。需要第一个下标时，可以用下文手写 top-k 的办法：先求最大值，再在等于它的位置中用 `vmin.xlane` 取最小的下标。
- **NaN 优先。** 有 NaN 时，最大值与最小值都是 NaN，两个下标都指向它。
- **−0 与 +0 相等。** 它们按并列处理。

还测了从提交到取回的延迟：`vmax.xlane`、`vmax.index.xlane`、`vmin.xlane` 都是 79 个周期，换用 `vx1` 槽与 `trf1` 队列也一样，与第三章第 5 节测得的 `vadd.xlane` 相同。

## 写法的限制：is_stable 与 bf16

`jax.lax.top_k(x, k)` 在 kernel 中无法编译：

```text
is_stable=True is not supported in Pallas top_k. For efficiency, only is_stable=False is supported
```

`lax.top_k` 默认要求稳定：相等的值按原来的顺序排列。Pallas 只支持 `jax.lax.top_k(x, k, is_stable=False)`，即相等值之间的顺序不确定。实验的每行都是互不相同的值，两者结果相同。

bf16 输入无法编译：`bfloat16 top_k is not supported on TPUv5 or older`。需要时先转换成 f32。

## 只改 k：重复选取 8 次

`jax.lax.top_k(x, 8, is_stable=False)` 的值和下标都正确。清单中是 8 轮选取：

| 指令 | 条数 |
| --- | ---: |
| `vmax.xlane` | 8 |
| `vmax.index.xlane` | 8 |
| `vpop` | 16 |
| `veq`、`vlt`、`vsel` | 8、7、21 |

清单的前两轮（只保留计算指令）：

```text
{ va0: vand.8x128.u32 v2, 0x7f, v1 ; vx0: vmax.index.xlane.2.8x128.f32 trf0, v0 }   # 第 1 轮：原始输入的下标
{ va0: veq.8x128.s32 vm6, v2, 0 ; va1: vlt.8x128.s32 vm7, v2, 2 }                    # 拼结果用的掩码
{ vr0: vpop.8x128 v3, trf0 }                                                         # 第 1 个下标
{ va0: veq.8x128.s32 vm0, v2, v3 }                                                   # 选中的位置
{ va0: vsel.8x128 v4, vm0, 0xff800000, v0 ; va1: vlt.8x128.s32 vm0, v2, 3 }          # 把它改成 −inf
{ vx0: vmax.index.xlane.0.8x128.f32 trf0, v4 }                                       # 第 2 轮
{ vr0: vpop.8x128 v5, trf0 }
{ va0: veq.8x128.s32 vm1, v2, v5 ; va1: vsel.8x128 v15, vm6, v3, v5 }                # lane 0 放第 1 个下标，其余放第 2 个
...
```

每一轮是一条依赖链：提交 `vmax.index.xlane`，等 79 个周期取回下标，`veq` 比较出选中的 lane，`vsel` 把它改成 `0xff800000`（−inf），下一轮对改过的数组再求下标。下标链在关键路径上；最大值则不在：编译器在后面对第 1–8 轮的数组 `v0`、`v4`、`v6`… 各提交一次 `vmax.xlane`，它们之间没有依赖，每 8 个周期可以发射一条。

8 个结果要拼成一行的第 0–7 个 lane。编译器事先算出“lane < 2”“lane < 3”…“lane < 8”的掩码（7 条 `vlt`，加上“lane = 0”的一条 `veq`），每取回一个结果就做一次 `vsel`：小于 j 的 lane 保留之前的结果，其余放第 j 个结果。值和下标各拼一次。

助记符中的 `.0`–`.3` 与队列的对应是固定的：`.0`、`.2` 的结果进入 `trf0`，`.1`、`.3` 进入 `trf1`（与第 9 节 `vxpose` 的变体相同），编译器轮流使用它们，并同时用 `vx0`、`vx1` 两个槽提交。

**开销。** 用 LCC 实测这段计算（把清单中输入 DMA 之后、输出 DMA 之前的向量指令原样插进载体执行），并用第三章第 5 节的发射模型预测：

| kernel | bundle 数 | XLU 归约 | 实测 R2 − R0 | 模型 |
| --- | ---: | ---: | ---: | ---: |
| argmax | 9 | 2 | 103 | 103 |
| `lax.top_k`，k = 8 | 54 | 16 | 671 | 671 |

R2 − R0 包含 13 个周期的读数与排空开销（第三章第 3 节）。argmax 的 90 个周期几乎全是一次 79 个周期的归约；top-8 的 658 个周期是 8 轮，每轮约 82 个周期，正是 79 个周期的延迟加上取回、比较、选择各一个周期。所以 k 增大时，时间按每轮约 82 个周期线性增长，增加的不是指令数，而是这条串行链的长度。

本实验每行 8 个、只有一个 TC VREG。行数更多时，各个 TC VREG 的链彼此独立，XLU 的提交指令每 8 个周期可以发射一条（第三章第 5 节），10 条左右的链可以交错执行；这是由模型参数推出的，本节没有实测。

## 只改宽度：跨两个 lane tile

`f32[8,256]` 每行跨两个 TC VREG，结果仍然正确。清单多出了 8 条 `vmax.8x128.f32`、8 条 `vge.8x128.f32` 和 8 条 `vperm`：每一轮先在两个 TC VREG 之间逐元素比较、取较大者，再对较大者做跨 lane 的选取，下标也要换算回 0–255 的范围。行宽每多一个 TC VREG，每一轮就要多一层这样的合并。

## 一个边界错误：有效值不足 k 个

输入的每行只有 3 个有限值，其余都是 `-inf`，求 top-8：

```text
第 0 行下标：[0, 2, 1, 127, 127, 127, 127, 127]
CPU：        [0, 2, 1, 3, 4, 5, 6, 7]
```

值是对的，但后 5 个下标全都是 127，重复了。`is_stable=False` 允许相等值以任意顺序出现，但不允许同一个下标出现多次。上一小节的规则解释了它：前 3 轮选完有限值后，整行都是 −inf，已选位置被改成的 −inf 与原有的 −inf 无法区分；全部并列时 `vmax.index.xlane` 返回 lane 127，第 4 轮选中它、把它改成 −inf（值不变），第 5 轮又全部并列，再次返回 127。

这类“有效值少于 k”的情况在实际中并不少见，例如对已经做过掩码的分数求 top-k。

## 只改写法：用掩码排除已选位置

问题出在“把已选位置的值改成 `-inf`”：改完之后，已选位置与原本就是 `-inf` 的位置无法区分。硬件有独立的掩码寄存器（第 6 节），可以把“是否已选”与值分开记录：

```python
def f(x):
    lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1).astype(jnp.float32)
    taken = jnp.zeros(x.shape, jnp.bool_)
    for _ in range(k):
        best = jnp.max(jnp.where(taken, -jnp.inf, x), axis=1, keepdims=True)
        index = jnp.min(jnp.where(~taken & (x == best), lane, float(x.shape[1])), axis=1, keepdims=True)
        taken = taken | (lane == index)
        ...
```

每一轮先求未选位置中的最大值，再在“未选且等于最大值”的位置中取最小的下标，最后把这个下标标为已选。下标用 f32 表示，以便用 `vmin.xlane` 在 lane 之间取最小。两组输入的结果都与 CPU 完全一致，包括只有 3 个有限值的一组：

```text
## 手写 top-8，用掩码排除已选位置，每行只有 3 个有限值，其余为 -inf
  值与 CPU 结果一致：True；下标与 CPU 结果一致：True
```

“相等时取最小的下标”恰好就是稳定排序的顺序，所以这个写法还给出了 `is_stable=True` 的结果。指令数与 `lax.top_k` 相近：每轮一次 `vmax.xlane` 和一次 `vmin.xlane`（共 16 次 XLU 归约、16 次 `vpop`），`taken` 的更新是掩码寄存器上的 `vmor`、`vmxor`，另有 36 条 `vsel`。但时间差了一倍：

| kernel | bundle 数 | XLU 归约 | 实测 R2 − R0 | 模型 |
| --- | ---: | ---: | ---: | ---: |
| `lax.top_k`，k = 8 | 54 | 16 | 671 | 671 |
| 手写 top-8 | 88 | 16 | 1337 | 1336 |

区别在依赖关系。`lax.top_k` 每轮的两次归约中，只有求下标的一次在链上，求值的一次可以放到后面；手写版本每轮先求最大值，再用它求下标，两次 79 个周期的归约都在链上，每轮约 165 个周期。这是为“下标不重复、相等时取最小下标”付出的代价。如果输入保证没有 −inf（例如掩码用一个有限的极小值），`lax.top_k` 的写法就不会出错，可以继续用它。

## 对照：原生 XLA

本小节实验[源码](02_jax_top_k.py)、[输出](02_jax_top_k.txt)。

| 运算 | XLA 的实现 | HLO 段 bundle 数 |
| --- | --- | ---: |
| top-1 | `custom-call`（TopK） | 1100 |
| top-8 | `sort`（连同下标 iota 一起排序） | 558 |

XLA 对 top-1 调用一个通用的 TopK 实现，对 top-8 则把整行连同下标一起排序再取前 8 个。两者的结果都正确，包括 `-inf` 的边界情况（XLA 的 `top_k` 默认 `is_stable=True`）。但对于“每行 128 个数中取最大的几个”，Pallas 中一次 `vmax.xlane` 加一次 `vmax.index.xlane` 的 argmax，比 XLA 的上千个 bundle 少得多；只是要自己处理上一小节的边界情况。
