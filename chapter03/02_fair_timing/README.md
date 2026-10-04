# 公平计时

比较两个实现谁更快，首先要保证两边测的是同一件事：同样的输入从同样的地方出发，结果写到同样的地方，计时覆盖同样的范围。本节用第二章第 2 节的例子，把原生 XLA 与 Pallas kernel 放在一起，用五种方法计时，看每种方法在哪里出错，以及怎样从编译结果判断两边实际做了什么。

## 比较的对象

本小节实验[源码](01_jax_pallas_timing_methods.py)、[输出](01_jax_pallas_timing_methods.txt)。

运算是 `f32[32768,128]`（16 MiB）的 `y = 2x + 1`，有三个候选：

```python
functions = {
    'XLA': lambda x: x * 2.0 + 1.0,
    'Pallas（out_type=pltpu.HBM）': pallas_hbm,
    'Pallas（out_type=ShapeDtypeStruct）': pallas_any,
}
```

两个 Pallas 候选都是第二章第 2 节输入输出双缓冲、`f32[2048,128]` tile 的 kernel，只改了一行：

```python
out_type=pltpu.HBM(x.shape, x.dtype) if out_hbm else jax.ShapeDtypeStruct(x.shape, x.dtype),
```

## 先看编译结果

计时之前，先检查编译后的 HLO 中三件与计时边界有关的事：有没有跨程序预取（`cross_program_prefetch`，把输入提前搬进片上内存）、有没有数组被放进 Megacore Shared CMEM（layout 中的 `S(3)`）、程序的结果（`ROOT`）是哪条指令：

```text
XLA：cross_program_prefetch 无，S(3) 0 处，ROOT 是 fusion
Pallas（out_type=pltpu.HBM）：cross_program_prefetch 无，S(3) 0 处，ROOT 是 copy
Pallas（out_type=ShapeDtypeStruct）：cross_program_prefetch 无，S(3) 0 处，ROOT 是 custom-call
```

`out_type=pltpu.HBM` 的版本在 kernel 之后多了一次 copy：kernel 写出的数组被要求留在 HBM 内存空间，而程序的结果使用默认内存空间，XLA 只好再复制一遍。这次 16 MiB 的复制在 XProf 中占 36211 个周期，与 kernel 本身相当。第二章第 2 节用 `pltpu.HBM` 是为了在 kernel 内部重复计时时固定数据通路，那里测的是 kernel 的每遍时间，不受这次复制影响；但作为一个完整的程序与 XLA 比较时，应当用 `ShapeDtypeStruct`，让 kernel 直接写程序的结果。

## 五种方法

| 方法 | XLA | Pallas，`pltpu.HBM` | Pallas，`ShapeDtypeStruct` |
| --- | ---: | ---: | ---: |
| 1. 调用并等到结果，固定同一个输入 | 190.3 µs | 233.1 µs | 198.8 µs |
| 2. 调用并等到结果，每次一个新输入 | 192.4 µs | 231.6 µs | 197.0 µs |
| 3. XProf，kernel（op） | 36205 个周期 | 44004 + 36211 个周期 | 43915 个周期 |
| 3. XProf，TensorCore 1 的 module | 36309 个周期 | 80408 个周期 | 44018 个周期 |
| 4. `fori_loop` 重复 16 与 32 次之差 | 9.8 µs | 43.1 µs | 30.9 µs |

方法 1、2、4 用主机时钟，单位是微秒；方法 3 是设备上的周期数（36205 个周期是 34.5 µs）。

**方法 1、2：主机计时。** 第 1 节说过，等到结果的时间包含 100 多微秒的固定开销（这里约 160 µs）。XLA 与 `ShapeDtypeStruct` 版本在设备上差 7710 个周期（7.3 µs），主机计时的差是 4.6 µs，而各轮之间的波动就有近 10 µs（方法 5）。主机计时只适合差别远大于开销波动的比较。

方法 1 与方法 2 的区别是输入是否每次都是新数组。这里两者相同，因为 XLA 没有做跨程序预取；在第二章第 9 节的矩阵乘法中，XLA 会把输入预取进 CMEM，固定输入时下一次调用可能直接用上片上的副本。正式比较应当每次用一个新的输入数组（[研究报告 28](../../../pallas-tpu-readings-dev/research_reports/28_xla_pallas_kernel_fair_timing.md) 称为 fresh-HBM），并且在预热之后计时。

**方法 3：XProf。** XProf 直接给出设备上的时间，差别清楚：XLA 的 fusion 36205 个周期，Pallas kernel 43915 个周期。比较完整程序时应看 module：它包含 kernel 之外的 copy 等其他指令。TensorCore 0 的 module 还包含一段 kernel 之外的 runtime 代码，这里约 3 万个周期，第 1 节的采集中约 1.9 万个周期，每次采集都不一样；两个候选都有这一段，用 TensorCore 1 的 module 比较更稳定。

**方法 4：在 jit 中循环。** 把调用放进 `jax.lax.fori_loop`，用循环 32 次与 16 次之差消去固定开销：

```python
looped = tpuasm_tools.compile(lambda x: jax.lax.fori_loop(0, count, lambda i, y: function(y), x), x, mesh=meshes[name])
```

这看起来与第二章第 2 节“在 kernel 内部重复”的思路相同，结果却完全不同：XLA 每次只要 9.8 µs（约 1 万个周期），比 XProf 中的 36205 个周期快了近 4 倍。原因在编译结果中：

```text
XLA：循环版本的 HLO 中 cross_program_prefetch 无，S(3) 13 处，ROOT 是 copy-done
Pallas（out_type=ShapeDtypeStruct）：循环版本的 HLO 中 cross_program_prefetch 无，S(3) 12 处，ROOT 是 copy-done
```

循环中传递的数组被 XLA 放进了 Megacore Shared CMEM，每次迭代读写的都是片上内存，不再经过 HBM。测到的是另一个程序。`pltpu.HBM` 版本的输出被固定在 HBM，所以循环中的时间（43.1 µs，约 45000 个周期）仍接近 kernel 本身。改变程序结构的计时方法，都要重新检查编译结果。

**方法 5：轮次与顺序。** 方法 2 按正序、倒序交替做 4 轮：

```text
第 1 轮：XLA 191.5 µs，Pallas（pltpu.HBM）232.1 µs，Pallas（ShapeDtypeStruct）195.3 µs
第 2 轮：XLA 189.1 µs，Pallas（pltpu.HBM）230.8 µs，Pallas（ShapeDtypeStruct）197.2 µs
第 3 轮：XLA 182.9 µs，Pallas（pltpu.HBM）224.2 µs，Pallas（ShapeDtypeStruct）194.9 µs
第 4 轮：XLA 182.0 µs，Pallas（pltpu.HBM）226.6 µs，Pallas（ShapeDtypeStruct）189.9 µs
```

同一个候选在不同轮次之间相差可达 9.5 µs，比 XLA 与 Pallas 的设备时间之差（7.3 µs）还大；只跑一轮，结论可能取决于运气。交替顺序可以抵消“先跑的吃亏”或“后跑的吃亏”这类与顺序有关的漂移。研究报告 28 的做法是：每个候选在自己的连续块中运行，块内取中位数，多轮之间正反交替，最后取各轮中位数的中位数；需要彻底清除片上状态时，每轮用一个新进程。

## 公平比较的清单

1. 写明比较的边界：输入从哪里出发（HBM，每次新数组）、结果写到哪里（程序的结果，在 HBM）、用几个 TensorCore。
2. 检查两边的编译结果：有没有额外的 copy、跨程序预取、放进 CMEM 的数组。不同就说明两边做的不是同一件事。
3. 设备时间用 XProf 或 kernel 内部的计数器；主机时间只用于差别远大于开销波动的场合。
4. 多轮、交替顺序，报告中位数，并保留每轮的结果。

按这张清单，本例的结论是：在同样从 HBM 读、写回 HBM 的边界下，XLA 的 fusion 36205 个周期，单个 TensorCore 的 Pallas kernel 43915 个周期。XLA 用了两个 TensorCore（第二章第 2 节），这是两个实现之间真实的差别，而不是计时的问题；第二章第 1 节的方法可以让 Pallas kernel 也用上两个 TensorCore。
