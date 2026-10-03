# Top-k

Top-k 从每行中选出最大的 k 个值及其下标。它把第 11 节的跨 lane 归约推进了一步：不仅要知道最大值是多少，还要知道它在哪里，并且要重复 k 次。本节从 argmax 出发，看 XLU 怎样同时给出最大值和下标，再看 k = 8 时的重复选取、跨两个 lane tile 的情况，以及当前 Pallas top-k 的限制和一个边界错误。

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

`jnp.max(x, axis=1)` 与 `jnp.argmax(x, axis=1)` 一起，清单中的计算指令是：

```text
vmax.xlane.2.8x128.f32 ...        # 每行的最大值
vmax.index.xlane.0.8x128.f32 ...  # 每行最大值所在的 lane
2 × vpop
```

XLU 有专门给出最大值下标的指令 `vmax.index.xlane`，不需要先求最大值、再比较、再找位置。值与下标都与 CPU 一致。

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

每一轮用 XLU 求出当前最大值及其下标，再用比较和选择把选中的位置从候选中去掉，进入下一轮。8 轮的 XLU 操作分散在助记符中 `.0`–`.3` 四种变体上。

k 增大时，轮数线性增长；每一轮都要等前一轮的结果才能去掉选中的元素，所以这是一条串行的依赖链。

## 只改宽度：跨两个 lane tile

`f32[8,256]` 每行跨两个 TC VREG，结果仍然正确。清单多出了 8 条 `vmax.8x128.f32`、8 条 `vge.8x128.f32` 和 8 条 `vperm`：每一轮先在两个 TC VREG 之间逐元素比较、取较大者，再对较大者做跨 lane 的选取，下标也要换算回 0–255 的范围。行宽每多一个 TC VREG，每一轮就要多一层这样的合并。

## 一个边界错误：有效值不足 k 个

输入的每行只有 3 个有限值，其余都是 `-inf`，求 top-8：

```text
第 0 行下标：[0, 2, 1, 127, 127, 127, 127, 127]
CPU：        [0, 2, 1, 3, 4, 5, 6, 7]
```

值是对的，但后 5 个下标全都是 127，重复了。`is_stable=False` 允许相等值以任意顺序出现，但不允许同一个下标出现多次。结果说明，去掉已选位置之后，它与原有的 `-inf` 无法区分，后面几轮反复选中了同一个位置。

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

“相等时取最小的下标”恰好就是稳定排序的顺序，所以这个写法还给出了 `is_stable=True` 的结果。代价与 `lax.top_k` 相当：每轮一次 `vmax.xlane` 和一次 `vmin.xlane`（共 16 次 XLU 归约、16 次 `vpop`），`taken` 的更新是掩码寄存器上的 `vmor`、`vmxor`，另有 36 条 `vsel`；`lax.top_k` 是每轮一次 `vmax.xlane` 和一次 `vmax.index.xlane`。

## 对照：原生 XLA

本小节实验[源码](02_jax_top_k.py)、[输出](02_jax_top_k.txt)。

| 运算 | XLA 的实现 | HLO 段 bundle 数 |
| --- | --- | ---: |
| top-1 | `custom-call`（TopK） | 1131 |
| top-8 | `sort`（连同下标 iota 一起排序） | 582 |

XLA 对 top-1 调用一个通用的 TopK 实现，对 top-8 则把整行连同下标一起排序再取前 8 个。两者的结果都正确，包括 `-inf` 的边界情况（XLA 的 `top_k` 默认 `is_stable=True`）。但对于“每行 128 个数中取最大的几个”，Pallas 中一次 `vmax.xlane` 加一次 `vmax.index.xlane` 的 argmax，比 XLA 的上千个 bundle 少得多；只是要自己处理上一小节的边界情况。
