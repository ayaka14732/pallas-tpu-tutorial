# 逐元素运算

逐元素运算是 TensorCore 向量单元最基本的工作：一条指令对一个 TC VREG 的 8×128 个位置同时做同一种运算。本节要建立的是一张“价目表”：哪些运算是一条硬件指令，哪些要用多条指令拼出来，拼出来的结果精度如何。设计算法时，这张表决定了哪些写法便宜、哪些写法昂贵。

## 写法：同一个 kernel 框架，只换中间一行

本节的实验都用同一个两输入 kernel，放在 [elementwise_common.py](elementwise_common.py) 中。相对于第 1 节的最小 kernel，它多一个输入 y，输出的 shape 和 dtype 由运算本身推出，计算换成任意函数 `f`：

```python
out = jax.eval_shape(f, jax.ShapeDtypeStruct(x.shape, x.dtype), jax.ShapeDtypeStruct(y.shape, y.dtype))
...
def kernel(x_hbm: Ref, y_hbm: Ref, o_hbm: Ref, x_vmem: Ref, y_vmem: Ref, o_vmem: Ref, sem: Ref) -> None:
    pltpu.async_copy(x_hbm, x_vmem, sem).wait()
    pltpu.async_copy(y_hbm, y_vmem, sem).wait()
    o_vmem[...] = f(x_vmem[...], y_vmem[...])
    pltpu.async_copy(o_vmem, o_hbm, sem).wait()
```

`jax.eval_shape` 只推导函数输出的 shape 和 dtype，不实际计算。各实验脚本把一组 `f` 依次交给这个框架，例如 `lambda x, y: jnp.exp(x)`，每个 `f` 单独编译一个 kernel，统计清单中的计算指令（不计 load/store、DMA 和标量指令）。

kernel 内可以直接用 `jnp` 和 `jax.lax` 的逐元素函数。它们由 Mosaic 降低成 TensorCore 指令，降低不了的会在编译时报错。

## f32：一条指令的运算

本小节实验[源码](01_pallas_f32_ops.py)、[输出](01_pallas_f32_ops.txt)。

输入是 `[-4, 4)` 和 `[0.25, 4)` 中的均匀随机数，误差与 float64 参考比较，以 ULP（相差几个 f32 可表示数）计：

| 运算 | 指令 | 误差 |
| --- | --- | ---: |
| `x + y` | `vadd.8x128.f32` | 0 ULP |
| `x - y` | `vsub.8x128.f32` | 0 ULP |
| `x * y` | `vmul.8x128.f32` | 0 ULP |
| `maximum(x, y)` | `vmax.8x128.f32` | 0 |
| `-x` | `vsub.8x128.f32`（0 − x） | 0 |
| `abs(x)` | `vand.8x128.u32`（清除符号位） | 0 |

加减乘的结果都是正确舍入的。`-x` 和 `abs(x)` 没有专用指令，分别用减法和按位与实现。

`x * y + 1` 是一条 `vmul` 加一条 `vadd`。TPU v4 的向量单元没有乘加融合（FMA），乘积先舍入一次，加法再舍入一次。乘积接近 −1 时发生相消，结果相对误差可达 21 ULP，而一次舍入的 FMA 不会有这个问题。依赖 FMA 精度的算法（例如某些补偿求和）在这里要重新设计。

上一节已经提到，`vmul.8x128.f32` 只能在 `va0` 槽发射，`vadd.8x128.f32` 只能在 `va1` 槽发射。一个 bundle 可以同时做一次 f32 乘法和一次 f32 加法，但不能同时做两次乘法。

## f32：超越函数与 EUP

`exp(x)` 的清单：

```text
{ va0: vmul.8x128.f32 v1, 1.4426950216293335, v0 }   # x × log2(e)
{ va0: vpow2.8x128.f32 erf, v1 }                     # 2 的幂，结果送入 erf
{ vr0: vpop.8x128 v2, erf }                          # 从 erf 取回结果
```

超越函数由独立的超越函数单元 EUP 计算。`vpow2` 在 `va0` 槽发射，但它的目的操作数不是 TC VREG，而是结果队列 `erf`；结果要另用一条 `vpop` 从 `vr0` 或 `vr1` 槽取回。

> 暂且可以理解为：EUP 的计算有延迟，提交后要过若干周期结果才进入 `erf`，`vpop` 在结果到达前会等待。第三章第 5 节测得：五种 EUP 函数都是提交后 7 个周期可以取回，连续提交每 2 个周期一次，所以连续计算许多个 TC VREG 时，EUP 每 2 个周期完成一个，延迟被流水掩盖。

本节实验中出现的 EUP 指令有 `vpow2`、`vlog2`、`vrcp`、`vrsqrt`、`vtanh`，每条都配一条 `vpop`。其他函数由它们加上普通向量运算拼成：

| 运算 | 计算指令数 | 核心指令 | 误差 | 最大绝对误差 |
| --- | ---: | --- | ---: | ---: |
| `exp2(x)` | 2 | `vpow2` | 57 ULP | 5.5e-5 |
| `exp(x)` | 3 | `vmul` + `vpow2` | 47 ULP | 1.8e-4 |
| `log(y)` | 3 | `vlog2` + `vmul` | 3652 ULP | 1.0e-4 |
| `tanh(x)` | 2 | `vtanh` | 710 ULP | 4.2e-5 |
| `1 / y` | 15 | `vrcp` + 迭代修正 | 1 ULP | 2.1e-7 |
| `x / y` | 16 | `vrcp` + 迭代修正 + `vmul` | 2 ULP | 1.0e-6 |
| `rsqrt(y)` | 11 | `vrsqrt` + 迭代修正 | 2 ULP | 1.7e-7 |
| `sqrt(y)` | 13 | `vrsqrt` + 迭代修正 | 2 ULP | 2.4e-7 |
| `sigmoid(x)` | 19 | `vpow2` + `vrcp` | 58 ULP | 9.5e-7 |
| `sin(x)` | 198 | 无 EUP，纯多项式 | 2 ULP | 8.7e-8 |

这张表中有两种不同的取舍：

- 倒数、平方根倒数和除法：EUP 只给出近似值，编译器再用几次乘加做牛顿迭代，把误差修正到 1–2 ULP。多出来的 `vweird`（判断是否为无穷或 NaN，结果写进掩码）、`veq`、`vsel` 等指令处理无穷、零和 NaN：这些输入不能进入迭代，直接选用特殊值。
- `exp`、`log`、`tanh`：直接使用 EUP 的近似结果，不做修正，误差为几十到几百 ULP。`log` 的 ULP 数格外大，是因为 `y` 接近 1 时结果接近 0，约 1e-4 的绝对误差折合成很多个 ULP。

这两种取舍的依据是 EUP 指令本身的精度。实验[源码](05_tpuasm_eup_raw.py)、[输出](05_tpuasm_eup_raw.txt)用 tpuasm 直接执行五条 EUP 指令，不加任何修正：先由 1024 个随机的 `u32` 拼出 [1, 2) 中的 y 和 [−4, 4) 中的 x，再对每条指令各提交、取回一次，与 float64 的参考值比较。

| 指令 | 最大相对误差 | 相当于 | 最大绝对误差 |
| --- | ---: | ---: | ---: |
| `vrcp`（1 / y） | 1.6e-05 | 约 16 位 | 1.1e-05 |
| `vrsqrt` | 4.0e-06 | 约 18 位 | 3.9e-06 |
| `vlog2` | 2.6e-04 | 约 12 位 | 1.4e-04 |
| `vpow2` | 4.1e-06 | 约 18 位 | 4.0e-05 |
| `vtanh` | 6.4e-05 | 约 14 位 | 4.3e-05 |

f32 的尾数是 24 位，而 EUP 只给出 12–18 位有效的结果。`vrcp` 的 16 位经过一次牛顿迭代（`r + r × (1 − y × r)`，清单中的 2 条 `vmul`、1 条 `vsub`、1 条 `vadd`）翻倍，超过 24 位，所以 `1 / y` 只需要一步修正就达到 1 ULP；`vrsqrt` 同理。`vpow2`、`vlog2`、`vtanh` 没有这样便宜的修正公式，编译器直接使用原始结果，上表的误差就是 `exp2`、`log`、`tanh` 的误差。

`sin` 不使用 EUP，而是范围约化加多项式，共 198 条指令，换来 2 ULP 的精度。用它之前要先想清楚是否值得。

对照实验[源码](04_jax_f32_ops.py)、[输出](04_jax_f32_ops.txt)：原生 XLA 对 `x / y`、`exp`、`log`、`tanh`、`sin`、`x * y + 1` 生成的计算指令和误差与 Pallas 完全相同，两者用的是同一套降低规则。需要更高精度时，例如要求 `exp` 或 `log` 达到 f32 的完整精度，就必须自己在 EUP 结果上追加修正步骤，XLA 和 Pallas 都不会自动做。

## 整数：没有整数乘法器

本小节实验[源码](02_pallas_i32_ops.py)、[输出](02_pallas_i32_ops.txt)。

输入是随机的 uint32（脚本第二部分把同样的 bit 解释为 int32），结果与 CPU 上的 JAX 逐元素比较，全部一致：

| 运算 | 计算指令数 | 指令 |
| --- | ---: | --- |
| `x + y`、`x - y` | 1 | `vadd.8x128.s32`、`vsub.8x128.s32` |
| `x & y`、`x \| y`、`x ^ y` | 1 | `vand`、`vor`、`vxor` |
| `x << 3`、`x >> 3` | 1 | `vshll`、`vshrl`（逻辑）、`vshra`（算术） |
| `population_count(x)`、`clz(x)` | 1 | `vpcnt`、`vclz` |
| int32 `maximum(x, y)` | 2 | `vgt` + `vsel` |
| `x * y` | 33 | 6 次 `vcvt` 转 f32、6 次 `vmul.f32`、6 次 `vcvt` 转回，加移位和掩码 |
| uint32 `x // y` | 318 | 32 轮移位—比较—相减 |
| int32 `x // y`、`x % y` | 347、333 | 同上，另加符号处理 |

加减、位运算、移位、位计数各是一条指令。但 32 bit 整数乘法没有对应的硬件指令。编译器把两个操作数拆成若干段，每段小到可以精确地放进 f32 的 24 bit 尾数，用 6 次 f32 乘法算出部分积，再转回整数、移位相加，共 33 条指令。整数除法和取余更贵，是逐位的长除法，约 330 条指令。

这个事实直接影响算法设计。例如第四章介绍的计数器式随机数生成器 Philox 以 32 bit 整数乘法为核心，在 TPU v4 上它的每次乘法都要付出这 33 条指令的代价；而只用加法、异或和移位的生成器则便宜得多。

另外两点：

- 无符号的 `maximum` 无法编译（`failed to legalize operation 'arith.maxui'`）。有符号的 `maximum` 是一次比较加一次选择。需要无符号比较时，可以先把两数都异或 `0x80000000` 再做有符号比较。
- `x << y` 中，脚本写的是 `x << (y & 31)`：移位量来自数据时，要自己保证它在 0–31 之间。

## 比较、选择与向量掩码

本小节实验[源码](03_pallas_compare_select.py)、[输出](03_pallas_compare_select.txt)。

`jnp.where(x > y, x, y)` 是两条指令：

```text
vgt.8x128.f32 vm0, v0, v1      # 比较结果写进掩码寄存器 vm0
vsel.8x128    v2, vm0, v0, v1  # 按掩码逐元素选择
```

比较指令的目的操作数是向量掩码寄存器 `vm0`–`vm7`，每个元素一个 bit；`vsel` 按掩码在两个 TC VREG 之间逐元素选择。比较的一侧可以是立即数，`where(x > 0, x, 0.1 * x)` 就是一条 `vgt`、一条 `vmul`、一条 `vsel`。

掩码之间的逻辑运算有专用指令 `vmand`、`vmor`、`vmxor`、`vmneg`，在 `misc` 槽执行。但编译器不一定用它们：`where((x > 0) & (y > 0), x, y)` 被改写成两次嵌套的 `vsel`，`where((x > 0) | ~(y > 0), x, y)` 则用了 `vmor` 和 `vmxor`。

TPU v4 只有 8 个向量掩码寄存器。实验在函数中先算出 count 个掩码，再依次使用它们，迫使所有掩码同时活跃：

| 同时活跃的掩码 | 用到的掩码寄存器 | `vgt` | `vsel` | 额外指令 |
| ---: | --- | ---: | ---: | --- |
| 6 | `vm0`–`vm5` | 6 | 12 | 无 |
| 12 | `vm0`–`vm7` | 13 | 28 | `vimm`×4、`vsmask`×4 |

掩码超过 8 个时，编译器把一部分掩码转存到 TC VREG 中：

```text
vimm.8x128.s32 v31, 0                  # 先把一个 TC VREG 清零
vsel.8x128 v31, vm6, 0xffffffff, v31   # 掩码为 1 的位置写全 1，掩码转存为 32 bit 数据
...
vsmask vm6, v31                        # 需要时再从 TC VREG 恢复成掩码
```

每转存一个掩码要多一条 `vimm`、一条 `vsel`、一条 `vsmask`，还要占用一个 TC VREG；另有一个掩码被重新比较了一次（13 条 `vgt`）。这是第 4 节所说的掩码代价。需要大量条件的算法，应尽量让每个掩码产生后立刻用掉。[研究报告 58](../../../pallas-tpu-readings-dev/research_reports/58_tpu_v4_vmask_spill.md) 记录了这种转存的更多细节。
