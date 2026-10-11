# 前缀扫描

前缀和把每个位置换成它和它之前所有元素之和：`y[i] = x[0] + … + x[i]`。归约只要最后一个数，前缀和要保留每一步的中间结果，所以不能直接套用上一节的 XLU 归约或二叉树。本节先看 Pallas 能否直接写 `jnp.cumsum`，再分别沿通道和沿子通道设计实现，并说明为什么同一个算法在两个方向上的开销不同。

## jnp.cumsum 不能用

本小节实验[源码](01_pallas_prefix_scan.py)、[输出](01_pallas_prefix_scan.txt)；kernel 的构造在 [scan_common.py](scan_common.py) 中，与上一节相同。

在 kernel 中写 `jnp.cumsum(x, axis=1)` 或 `axis=0`，编译失败：

```text
Unimplemented primitive in Pallas TPU lowering for tc: cumsum.
```

Mosaic 没有为前缀和提供降低规则，只能用已有的操作自己拼。下面是两种基本思路。

## 沿通道：Hillis–Steele 扫描

Hillis–Steele 扫描分 log₂(n) 轮：第 d 轮，每个位置加上它左边第 2^d 个位置的值（左边不够的不加）。128 个通道需要 7 轮：

```python
def hillis_steele_lanes(x):
    lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    for d in range(7):
        shift = 1 << d
        x = x + jnp.where(lane >= shift, pltpu.roll(x, shift, axis=1), 0)
    return x
```

`pltpu.roll(x, shift, axis=1)` 把每一行循环右移 `shift` 位，`jnp.where(lane >= shift, ..., 0)` 把循环绕回来的部分置零。结果正确，每一轮是一次 XLU 循环移位（`vrot` 加 `vpop`）、一次比较（`vlt`）、一次选择（`vsel`）和一次加法，7 轮共 37 条计算指令（含生成 lane 序号的 2 条）。8 行在一个 TC VREG 中同时完成。

bf16 输入先 unpack 成 f32，其余相同。

## 沿子通道：方向决定开销

同样的算法改为沿子通道，3 轮即可（8 = 2³）：

```python
for d in range(3):
    shift = 1 << d
    x = x + jnp.where(sublane >= shift, pltpu.roll(x, shift, axis=0), 0)
```

结果正确，但用了 17 条 `vrot.slane.down`。上一节已经看到，`pltpu.roll(x, k, axis=0)` 降低成 `(8 - k) % 8` 条 `vrot.slane.down`：位移 1、2、4 分别要 7、6、4 条。

`vrot.slane.down` 只朝一个方向移动，把它的方向用作前缀和要走的方向，开销最低。实际上，后缀和 `y[i] = x[i] + … + x[7]` 正好需要 roll 位移为 7、6、4，即每轮 1、2、4 条 `vrot.slane.down`：

```python
for d in range(3):
    shift = 1 << d
    x = x + jnp.where(sublane < 8 - shift, pltpu.roll(x, 8 - shift, axis=0), 0)
```

结果正确，只用 7 条移位。前缀和与后缀和在数学上对称，在 TPU v4 上开销却差一倍多。需要沿子通道扫描时，可以先把数据的行序颠倒存放，再做后缀和。

## 沿子通道的另一种做法：逐行广播

本小节实验[源码](03_pallas_row_serial_scan.py)、[输出](03_pallas_row_serial_scan.txt)。

8 行很少，也可以逐行串行地累加：维护一个 8 个子通道都相同的累加值，每次把下一行广播到 8 个子通道再加上：

```python
total = jnp.zeros(x_vmem.shape, x_vmem.dtype)
for k in range(8):
    row = x_vmem[pl.ds(k, 8, stride=0), :]
    total = total + row
    o_vmem[pl.ds(k, 1), :] = total[0:1, :]
```

`pl.ds(start, size, stride)` 的第三个参数是行距：从第 k 行开始读 8 行，每读一行前进 `stride` 行。`stride=0` 时 8 行都是第 k 行，即把一行广播到整个 TC VREG。清单中它是一条带 `ss=0` 的 load：

```tpuasm
vld: vld.8x128 v0, [vmem:0x0, ss=0]       # 第 0 行广播到 8 个 sublane
vld: vld.8x128 v1, [vmem:0x1, ss=0]       # 第 1 行
vst: vst.8x128 [vmem:0x8, sm=1], v0       # 输出第 0 行
va1: vadd.8x128.f32 v2, v1, v0            # 前两行之和
vst: vst.8x128 [vmem:0x9, sm=1], v2       # 输出第 1 行
```

`ss` 是 `vld` 的子通道行距，与第 5 节的子通道掩码 `sm` 一样是 load/store 地址的一部分。共 8 条 load、7 条加法、8 条只写一个子通道的 store，没有任何移位。

同样的广播若写成 `jnp.broadcast_to(x_vmem[pl.ds(k, 1), :], x_vmem.shape)`，结果也正确，但编译器先把 8 行逐行搬到一块临时的 TC VMEM，再从那里做 `ss=0` 的 load，load 和 store 各多出 8 条。直接在 Ref 上写出行距，就能直接用上 load 指令的这项能力。

## 指令少不等于快：各种写法的周期数

前面的比较只数了指令。实验[源码](05_tpuasm_scan_timing.py)、[输出](05_tpuasm_scan_timing.txt)把四种写法中编译器生成的计算部分（输入 DMA 之后、输出 DMA 之前的向量指令）原样插进第 3 节的 `LccProbe` 载体，读 R2 − R0，并与发射模型的预测比较：

```python
section = tpuasm_tools.compute_section(listing)      # 清单中两次 DMA 等待之间的 bundle，去掉标量指令
_, measured = probe.time_section(section)            # 原样插进载体执行
reads = issue_model.replay(issue_model.parse(tpuasm_tools.section_program(section)))
```

| 写法 | bundle 数 | 实测 R2 − R0 | 模型 | 关键路径 |
| --- | ---: | ---: | ---: | --- |
| 沿通道，Hillis–Steele | 35 | 526 | 526 | 7 轮，每轮一次 XLU 循环移位（提交后 69 个周期才能取回）加比较、选择、加法 |
| 沿子通道，前缀和 | 27 | 58 | 58 | 17 条 `vrot.slane.down` 与 3 条加法串行，每条的结果 2 个周期后可用 |
| 沿子通道，后缀和 | 17 | 38 | 38 | 7 条 `vrot.slane.down` 与 3 条加法 |
| 沿子通道，逐行广播 | 14 | 30 | 30 | 7 条加法串行，每条 2 个周期 |

R2 − R0 包含 13 个周期的固定开销（第 3 节），扣除之后四种写法分别是 513、45、25、17 个周期。

> 暂且可以理解为：模型按每条指令的结果延迟和各单元的发射间隔，逐个 bundle 推算它最早何时可以发射。用到的参数就是本章各节测出的那些：f32 加法和 `vrot.slane.down` 2 个周期，`vrot` 往返 69 个周期、发射间隔 8 个周期。第三章第 4 节给出完整的规则。

四种写法的模型与实测完全一致：这几段代码的周期数可以只凭清单和这几个参数算出来。

沿通道的 Hillis–Steele 每轮的指令最少，单个 TC VREG 的等待却最长，是逐行广播的 30 倍：每一轮都要等上一轮的 XLU 结果取回，才能发出下一轮的移位，7 轮就是 7 × 69 = 483 个周期。这是第 9 节所说的第一种极限：结果前后依赖，时间是往返的次数乘延迟。XLU 的提交指令每 8 个周期可以发射一条，所以同时扫描许多个 TC VREG 时，各个 TC VREG 的轮次可以交错，平均到每个 TC VREG 的时间会小得多；只扫描一个 TC VREG 时，这段等待无法掩盖。

## 只改硬件单元：沿通道的前缀和交给 MXU

本小节实验[源码](04_pallas_scan_by_matmul.py)、[输出](04_pallas_scan_by_matmul.txt)。

沿通道的前缀和是一个线性变换：`y[s, l] = Σ_{k ≤ l} x[s, k]`，即 `y = x @ U`，`U` 是上三角全 1 的 `128×128` 矩阵。所以它也可以交给 MXU（第 11 节），一次 `vmatmul` 完成 8 行：

```python
row = jax.lax.broadcasted_iota(jnp.int32, (128, 128), 0)
column = jax.lax.broadcasted_iota(jnp.int32, (128, 128), 1)
upper = (row <= column).astype(jnp.float32)
return jnp.dot(x, upper, precision=precision, preferred_element_type=jnp.float32)
```

实验只改 `precision`：

| `precision` | −100 到 100 的整数 | `[0, 1)` 的随机浮点数（最大绝对误差） | MXU 指令 |
| --- | --- | ---: | --- |
| 默认 | 精确 | 0.0174 | 8 次 `vmatpush`、1 次 `vmatmul` |
| `HIGHEST` | 精确 | 3.22 × 10⁻⁶ | 96 次 `vmatpush`、6 次 `vmatmul` |

第 11 节说过，f32 的 `jnp.dot` 默认把输入舍入成 bf16 再乘。`U` 只含 0 和 1，舍入不改变它；x 若是 bf16 能精确表示的数（例如绝对值不超过 256 的整数），结果精确，否则每个元素先带上 bf16 的舍入误差，前缀和的误差逐步累积到 0.0174。`HIGHEST` 把 x 和 `U` 拆成高、低两部分分别相乘（第 11 节的多遍乘法），误差降到 f32 的量级，代价是 6 次 `vmatmul` 和 96 次 `vmatpush`。

清单中构造 `U` 还用了约 50 条向量指令（`vle`、`vsel`、`vpackc` 等）。`U` 是常数，实际使用时应当只构造一次，或作为输入传入，并且一次装入 MXU 后用于许多个 TC VREG：每个 TC VREG 只需一次 `vmatmul`（默认精度），第一个结果 83 个周期后可以取回，之后每 8 个周期一个（第 11 节）。对需要扫描很多行的情况，这比 7 轮 XLU 循环移位便宜得多。这是“先看硬件有哪些单元，再决定算法落在哪个单元上”的一个例子：前缀和在数学上是矩阵乘法，TPU 上最强的单元正是做矩阵乘法的。

## 对照：原生 XLA

本小节实验[源码](02_jax_prefix_scan.py)、[输出](02_jax_prefix_scan.txt)。

XLA 把 `jnp.cumsum` 降低为 `reduce-window`：

- 沿子通道（axis=0）：48 个 bundle，做法与上一小节完全相同，即 `ss=0` 广播 load、逐行累加、单 sublane 的 store。
- 沿通道（axis=1）：234 个 bundle。XLA 先把 `8×128` 转置，在子通道方向逐行累加 128 次（129 条 `vadd`），再转置回来；没有用 Hillis–Steele 的 7 轮循环移位。

### 周期数

本小节实验[源码](06_jax_pallas_cycles.py)、[输出](06_jax_pallas_cycles.txt)。

把 `f32[8,128]` 的前缀和放进第 3 节的 TC VMEM 边界，XLA 的 `jnp.cumsum` 与本节的几种写法各测一次，结果逐元素相同：

| 方向 | 写法 | 区间的 bundle 数 | 周期数 |
| --- | --- | ---: | ---: |
| 沿通道 | XLA：转置、逐行累加、转置回来 | 230 | 430 |
| 沿通道 | Pallas：Hillis–Steele | 56 | 531 |
| 沿通道 | Pallas：交给 MXU（默认精度） | 65 | 173 |
| 沿子通道 | XLA：逐行广播 | 43 | 59 |
| 沿子通道 | Pallas：Hillis–Steele | 49 | 65 |
| 沿子通道 | Pallas：逐行广播 | 36 | 52 |

沿通道的三种写法，周期数的次序与指令数正好相反。Hillis–Steele 的 bundle 最少，却最慢：7 次 XLU 往返串在一条链上。XLA 的 bundle 是它的四倍，反而快 100 个周期：它付的是两次转置（一次取回 16 个 TC VREG，一次提交 16 个，各约 130 个周期）和中间 129 条串行的加法（每条 2 个周期），三段还有一部分相互重叠。交给 MXU 最快，173 个周期中包含了现场构造 `U` 并装入 MXU 的指令，以及一次 83 个周期的往返；`U` 事先装好时，每多一个 TC VREG 只多 8 个周期。

沿子通道三者相近，逐行广播比 XLA 少 7 个周期，Hillis–Steele 的 17 条移位使它最慢。

所以这一节的结论要用周期数来下：沿通道的前缀和，“按算法的 log 轮数设计”的 Hillis–Steele 指令最少，但在只有一两个 TC VREG 时不如 XLA 的写法，更不如 MXU；只有同时扫描很多 TC VREG、XLU 的往返可以相互重叠时，它才划算。XLA 的选择在这里是一个不容易越过的底线。
