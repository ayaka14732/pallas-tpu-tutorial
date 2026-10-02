# 前缀扫描

前缀和把每个位置换成它和它之前所有元素之和：`y[i] = x[0] + … + x[i]`。归约只要最后一个数，前缀和要保留每一步的中间结果，所以不能直接套用上一节的 XLU 归约或二叉树。本节先看 Pallas 能否直接写 `jnp.cumsum`，再分别沿 lane 和沿 sublane 设计实现，并说明为什么同一个算法在两个方向上的代价不同。

## jnp.cumsum 不能用

本小节实验[源码](01_pallas_prefix_scan.py)、[输出](01_pallas_prefix_scan.txt)；kernel 的构造在 [scan_common.py](scan_common.py) 中，与上一节相同。

在 kernel 中写 `jnp.cumsum(x, axis=1)` 或 `axis=0`，编译失败：

```text
Unimplemented primitive in Pallas TPU lowering for tc: cumsum.
```

Mosaic 没有为前缀和提供降低规则，只能用已有的操作自己拼。下面是两种基本思路。

## 沿 lane：Hillis–Steele 扫描

Hillis–Steele 扫描分 log₂(n) 轮：第 d 轮，每个位置加上它左边第 2^d 个位置的值（左边不够的不加）。128 个 lane 需要 7 轮：

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

## 沿 sublane：方向决定代价

同样的算法改为沿 sublane，3 轮即可（8 = 2³）：

```python
for d in range(3):
    shift = 1 << d
    x = x + jnp.where(sublane >= shift, pltpu.roll(x, shift, axis=0), 0)
```

结果正确，但用了 17 条 `vrot.slane.down`。上一节已经看到，`pltpu.roll(x, k, axis=0)` 降低成 `(8 - k) % 8` 条 `vrot.slane.down`：位移 1、2、4 分别要 7、6、4 条。

`vrot.slane.down` 只朝一个方向移动，把它的方向用作前缀和要走的方向，代价最低。实际上，后缀和 `y[i] = x[i] + … + x[7]` 正好需要 roll 位移为 7、6、4，即每轮 1、2、4 条 `vrot.slane.down`：

```python
for d in range(3):
    shift = 1 << d
    x = x + jnp.where(sublane < 8 - shift, pltpu.roll(x, 8 - shift, axis=0), 0)
```

结果正确，只用 7 条移位。前缀和与后缀和在数学上对称，在 TPU v4 上代价却差一倍多。需要沿 sublane 扫描时，可以先把数据的行序颠倒存放，再做后缀和。

## 沿 sublane 的另一种做法：逐行广播

本小节实验[源码](03_pallas_row_serial_scan.py)、[输出](03_pallas_row_serial_scan.txt)。

8 行很少，也可以逐行串行地累加：维护一个 8 个 sublane 都相同的累加值，每次把下一行广播到 8 个 sublane 再加上：

```python
total = jnp.zeros(x_vmem.shape, x_vmem.dtype)
for k in range(8):
    row = x_vmem[pl.ds(k, 8, stride=0), :]
    total = total + row
    o_vmem[pl.ds(k, 1), :] = total[0:1, :]
```

`pl.ds(start, size, stride)` 的第三个参数是行距：从第 k 行开始读 8 行，每读一行前进 `stride` 行。`stride=0` 时 8 行都是第 k 行，即把一行广播到整个 TC VREG。清单中它是一条带 `ss=0` 的 load：

```text
vld: vld.8x128 v0, [vmem:0x0, ss=0]       # 第 0 行广播到 8 个 sublane
vld: vld.8x128 v1, [vmem:0x1, ss=0]       # 第 1 行
vst: vst.8x128 [vmem:0x8, sm=1], v0       # 输出第 0 行
va1: vadd.8x128.f32 v2, v1, v0            # 前两行之和
vst: vst.8x128 [vmem:0x9, sm=1], v2       # 输出第 1 行
```

`ss` 是 `vld` 的 sublane 行距，与第 4 节的 sublane 掩码 `sm` 一样是 load/store 地址的一部分。共 8 条 load、7 条加法、8 条只写一个 sublane 的 store，没有任何移位。

同样的广播若写成 `jnp.broadcast_to(x_vmem[pl.ds(k, 1), :], x_vmem.shape)`，结果也正确，但编译器先把 8 行逐行搬到一块临时的 TC VMEM，再从那里做 `ss=0` 的 load，load 和 store 各多出 8 条。直接在 Ref 上写出行距，就能直接用上 load 指令的这项能力。

## 对照：原生 XLA

本小节实验[源码](02_jax_prefix_scan.py)、[输出](02_jax_prefix_scan.txt)。

XLA 把 `jnp.cumsum` 降低为 `reduce-window`：

- 沿 sublane（axis=0）：51 个 bundle，做法与上一小节完全相同，即 `ss=0` 广播 load、逐行累加、单 sublane 的 store。
- 沿 lane（axis=1）：237 个 bundle。XLA 先把 `8×128` 转置，在 sublane 方向逐行累加 128 次（129 条 `vadd`），再转置回来；没有用 Hillis–Steele 的 7 轮循环移位。

这里 XLA 的选择有一半值得借鉴：沿 sublane 的逐行广播是很好的做法；沿 lane 的方向，按算法的 log 轮数设计、利用 XLU 的循环移位，指令数少得多。
