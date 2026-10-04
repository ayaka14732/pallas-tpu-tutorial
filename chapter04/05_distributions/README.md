# 从比特到分布

生成器给出的是均匀分布的 32 位整数。实际需要的往往是 `[0, 1)` 中的浮点数、按概率取真假的掩码、正态分布或指数分布的噪声，以及把 f32 随机舍入成 bf16。本节逐一看这些变换：数学上怎样从比特得到分布，在 TensorCore 上用哪些指令，每个 TC VREG 要多少条，结果的精度和取值范围受到什么限制。

## 实验的写法

本节的五个实验共用 [distribution_common.py](distribution_common.py)。每个变换写成一个函数，kernel 先 `prng_seed(1)`，再调用它，得到若干个 `[512,128]` 的数组（64 个 TC VREG，65536 个随机数）写回 HBM：

```python
def kernel(*refs: Ref) -> None:
    ...
    pltpu.prng_seed(1)
    for buffer, value in zip(buffers, sample(*(buffer[...] for buffer in input_buffers)), strict=True):
        buffer[...] = value
```

需要输入数组的变换（随机舍入）通过 `inputs` 传入。各实验只改 `sample`。

统计指令时，用同样的写法编译一个只生成比特的 kernel（`sample = lambda: (bits(),)`）作为基准，把两者的计算指令相减，再除以 64：

```python
def per_tile(listing: str, baseline: str) -> str:
    difference = compute_counts(listing) - compute_counts(baseline)
    total = sum(difference.values()) / TILES
```

这样种子混合、`vrng`、DMA 与 `vst` 都被减掉，剩下的就是变换本身每个 TC VREG 的开销。`excerpt` 从清单中取出第一条 `vrng`（或 `vlog2`）开始的几个 bundle，作为变换的实际指令序列。

## 均匀分布

本小节实验[源码](01_pallas_uniform.py)、[输出](01_pallas_uniform.txt)。

### 用尾数构造

f32 由 1 位符号、8 位指数、23 位尾数组成。指数位为 `0x7F`（偏置后为 0）时，数值是 `1.m`，23 位尾数 m 均匀地取遍 `[1, 2)` 中 2²³ 个等距的值。所以把随机比特的高 23 位放进尾数、配上 `0x3F800000` 的符号与指数，就得到 `[1, 2)` 中的均匀分布，再减 1：

```python
def mantissa():
    # 高 23 位放进 [1, 2) 的尾数，减 1 得到 [0, 1) 中 2^-23 的整数倍。
    return (jax.lax.bitcast_convert_type((bits() >> 9) | jnp.uint32(0x3F800000), jnp.float32) - 1.0,)
```

`bitcast_convert_type` 只重新解释位型，不产生指令。清单中每个 TC VREG 三条：

```text
{ va0: vrng.8x128.u32 v22 }
{ va1: vshrl.8x128.s32 v23, v22, 0x9 }
{ va0: vor.8x128.u32 v24, 0x3f800000, v23 }
{ va1: vadd.8x128.f32 v25, -1.0, v24 }
```

`[1, 2)` 中的数减 1 是精确的（结果的位数不超过原数），所以结果恰好是 `k · 2^-23`，k 是比特的高 23 位：

```text
手写尾数版本等于 (比特 >> 9) × 2^-23：True
```

`pltpu.stateful_uniform` 与 `jax.random.uniform` 的实现相同，输出与手写版本逐位相同，只多一条 `vmax`：按 `[minval, maxval)` 缩放后可能因舍入略小于 `minval`，`vmax` 把它夹回来。默认的 `[0, 1)` 不需要这一条。

结果可以取到 0、取不到 1，最小的正值是 2^-23 ≈ 1.19 × 10⁻⁷。65536 个样本中有 65276 个不同的值：从 2²³ 个值中抽 65536 次，按生日问题约有 65536² / 2²⁴ ≈ 256 次重复，与 260 相符。

### 用整数转浮点

另一种写法把高 24 位当作整数转成 f32，再乘以 2^-24：

```python
def convert():
    # 高 24 位转成整数再转成 f32（24 位以内的整数在 f32 中精确），乘以 2^-24。
    return ((bits() >> 8).astype(jnp.int32).astype(jnp.float32) * 2.0**-24,)
```

```text
{ va1: vshrl.8x128.s32 v23, v22, 0x8 }
{ va0: vcvt.8x128.s32.f32 v24, v23 }
{ va0: vmul.8x128.f32 v25, 5.960464477539063e-08, v24 }
```

同样三条指令，分辨率却高了一倍：结果是 2^-24 的整数倍而不全是 2^-23 的整数倍。f32 的尾数在 `[0.5, 1)` 中的间隔本来就是 2^-24，尾数构造的方法浪费了这一位。

### 改变区间

`stateful_uniform(minval=-1, maxval=1)` 多一条 `vmul` 和一条 `vadd`，共 6 条：`u · (maxval − minval) + minval`。缩放之后间隔变成 `2 · 2^-23`，可取的值的个数不变。

## 伯努利分布

本小节实验[源码](02_pallas_bernoulli.py)、[输出](02_pallas_bernoulli.txt)。

### 两种写法

`pltpu.stateful_bernoulli(p, shape)` 先生成均匀分布，再与 p 比较，每个 TC VREG 6 条：均匀分布的 4 条、`vlt.f32`、把掩码变成整数的 `vsel`。

概率可以直接写成整数阈值，跳过浮点：

```python
def threshold(p: float) -> int:
    return round(p * 2**24)

def by_integer(p: float):
    # 右移 8 位后是非负的 int32，可以用有符号比较。
    return lambda: (((bits() >> 8).astype(jnp.int32) < threshold(p)).astype(jnp.int32),)
```

```text
{ va1: vshrl.8x128.s32 v23, v22, 0x8 }
{ va0: vlt.8x128.s32 vm0, v23, 4194304 }
{ va0: vsel.8x128 v25, vm0, 0x1, v24 }
```

3 条，少一半。右移 8 位使数值落在有符号整数的非负范围内，避免第一章第 6 节中无符号比较的问题；阈值 `4194304` 就是 0.25 · 2²⁴，作为立即数直接写在 `vlt` 中。

### 概率实际是多少

两种写法都把 p 量化了。浮点写法中，`u = k · 2^-23 < p` 成立的 k 有 `⌈p · 2²³⌉` 个；整数写法中，概率是 `round(p · 2²⁴) / 2²⁴`：

| p | `stateful_bernoulli` 的实际概率 | 整数阈值的实际概率 | 65536 个样本中为真的比例 |
| --- | --- | --- | --- |
| 0.25 | 0.25 | 0.25 | 0.2504 |
| 0.3 | 0.300000071526（+7.15 × 10⁻⁸） | 0.300000011921（+1.19 × 10⁻⁸） | 0.3013 |

两种写法用的比特相同，所以样本比例完全一样；与 p 的差别都在 10⁻⁷ 以下，远小于样本的标准误差（0.0018）。整数写法用了 24 位，量化误差更小。

### 掩码不必写成整数

比较的结果本来就在掩码寄存器 `vm0` 中（第一章第 6 节）。若随机掩码只是为了选择，例如以 1/4 的概率置 0、其余放大 4/3：

```python
def dropout(x):
    keep = (bits() >> 8).astype(jnp.int32) >= threshold(0.25)
    return (jnp.where(keep, x * (4.0 / 3.0), 0.0),)
```

```text
{ va1: vshrl.8x128.s32 v25, v24, 0x8 }
{ va0: vlt.8x128.s32 vm0, v25, 4194304 ; ... }
{ va0: vsel.8x128 v26, vm0, 0x0, v23 }
```

掩码从 `vlt` 直接进入 `vsel`，不写回 TC VREG 或内存；加上乘法，每个 TC VREG 4 条。结果只有 0 和 4，均值 2.9985，期望 3。

## 正态分布

本小节实验[源码](03_pallas_normal.py)、[输出](03_pallas_normal.txt)。

### 反误差函数

`pltpu.stateful_normal` 与 `jax.random.normal` 相同：先得到 `(−1, 1)` 中的均匀数 u，再计算 `√2 · erfinv(u)`。Pallas 中 `erfinv` 的 f32 实现（JAX 的 `jax/_src/pallas/utils.py`）是：

```python
w = -jnp.log1p(x * -x)
w_lt_5 = w < 5.0
w = jnp.where(w_lt_5, w - 2.5, jnp.sqrt(w) - 3.0)
p = jnp.where(w_lt_5, w_lt_5_constants[0], w_gt_5_constants[0])
for i in range(1, k_degree):
    c = jnp.where(w_lt_5, w_lt_5_constants[i], w_gt_5_constants[i])
    p = c + p * w
```

`w` 按是否小于 5 分两段，每段一个 9 个系数的多项式。向量单元不能让不同的 lane 走不同的分支，所以两段共用一次 Horner 求值，每一步用 `vsel` 按 lane 选出系数。清单的统计与此对应：

```text
每个正态数 TC VREG：变换指令 77.2 条，其中 EUP vlog2.8x128.f32 1，vrsqrt.8x128.f32 1
vmul.8x128.f32 21，vsel.8x128 14.4531，vadd.8x128.f32 14，vlt.8x128.f32 10.5781，...
```

`log1p` 用 `vlog2` 加乘以 ln 2 实现，参数很小时另走一条近似（清单中与 `0.00044273…` 的比较和随后的 `vsel`）；`sqrt` 用 `vrsqrt`。每个 TC VREG 两条 EUP 指令、约 77 条计算指令，是均匀分布的二十多倍。

65536 个样本中，落在 ±1σ、±2σ、±3σ 内的比例为 0.6851、0.9540、0.9974，理论值 0.6827、0.9545、0.9973。

### Box–Muller

另一种经典方法用两个均匀数得到两个独立的正态数：

```python
def box_muller():
    u1 = 1.0 - uniform()
    u2 = uniform()
    r = jnp.sqrt(-2.0 * jnp.log(u1))
    theta = 2.0 * math.pi * u2
    return r * jnp.cos(theta), r * jnp.sin(theta)
```

每两个正态数只要一次 `vlog2` 和一次 `vrsqrt`，EUP 指令少一半；但每个正态数 TC VREG 需要约 121 条计算指令，比反误差函数还多。原因是 TPU v4 的 EUP 只有 `vpow2`、`vlog2`、`vrcp`、`vrsqrt`、`vtanh`（第一章第 6 节），没有正弦和余弦：`jnp.cos`、`jnp.sin` 要先做范围归约（清单中大量的 `vcvt`、`vand`、`vshrl`、`vshll`），再用多项式近似。两个输出的相关系数 0.0018，分布同样正确。

在 TPU v4 上，正态分布的开销主要在变换，选择变换时要按“EUP 有哪些函数”来估算，而不是按教科书上的运算次数。

## 指数分布与 Gumbel 分布

本小节实验[源码](04_pallas_exponential_gumbel.py)、[输出](04_pallas_exponential_gumbel.txt)。

指数分布是 `−log u`，Gumbel 分布是 `−log(−log u)`。u 必须避开 0，这里用 `2 − [1, 2)` 直接得到 `(0, 1]`，与 `[0, 1)` 的开销相同：

```python
def open_uniform():
    # (0, 1] 中 2^-23 的整数倍：1 − [0, 1)，避免 log(0)。
    return 2.0 - jax.lax.bitcast_convert_type((bits() >> 9) | jnp.uint32(0x3F800000), jnp.float32)
```

每次 `log` 是三条指令：`vlog2` 送入 EUP、`vpop` 取回（第一章第 6 节）、`vmul` 乘以 ln 2：

| 分布 | 每个 TC VREG 的计算指令（含均匀数的 3 条） | 与同一 u 的 float64 公式相比的最大绝对误差 | 均值（理论值） |
| --- | --- | --- | --- |
| 指数 | 7（`vlog2` 1） | 1.07 × 10⁻⁴ | 0.9952（1） |
| Gumbel | 11（`vlog2` 2） | 3.58 × 10⁻⁴ | 0.5833（0.5772） |

误差来自 EUP 的 `vlog2` 近似。Gumbel 的相对误差最大达 0.24，出现在结果接近 0 的地方，所以要按绝对误差衡量。

均匀数的分辨率限制了分布的尾部。u 的最小值是 2^-23，所以 `−log u` 不会超过 23 · ln 2 ≈ 15.94；真实的指数分布超过这个值的概率是 e^-15.94 ≈ 1.2 × 10⁻⁷，即每八百多万个样本中应有一个，这里永远不会出现。用 24 位的整数转浮点写法，上限提高到 16.6。反过来，u = 1 时 Gumbel 是 `−log 0 = +∞`，概率 2^-23；用 Gumbel 噪声做选择时，要么接受它总会选中该位置，要么把 u 限制在 `(0, 1)`。

## 随机舍入

本小节实验[源码](05_pallas_stochastic_round.py)、[输出](05_pallas_stochastic_round.txt)。

把 f32 舍入成 bf16 时，按被舍去部分的大小随机决定向上还是向下：被舍去部分占相邻两个 bf16 之间距离的比例为 f，就以概率 f 向上。期望值等于原值，累加时不会像舍入到最近那样积累系统偏差。

### 公开接口与手写

`pltpu.stochastic_round` 在 TPU v4 上不能编译：

```text
pltpu.stochastic_round：编译失败：Stochastic convert not supported on TPU generations < 5.
```

硬件本身有一种随机舍入：第一章第 5 节看到，f32 → int32 的 `vcvt` 的第三个操作数是随机舍入的阈值，传入一个 TC VREG 的随机比特就是一次无偏的随机舍入。但 f32 → bf16 的打包指令 `vpackc` 没有这样的操作数，只能手写。好在这个运算只需要整数加法。bf16 就是 f32 的高 16 位，被舍去的正是低 16 位；把一个均匀的 16 位随机数加到低 16 位上，产生进位的概率正好等于低 16 位占 2¹⁶ 的比例，再截掉低 16 位：

```python
def by_hand(x):
    raw = jax.lax.bitcast_convert_type(x, jnp.uint32)
    rounded = (raw + (bits() & jnp.uint32(0xFFFF))) & jnp.uint32(0xFFFF0000)
    return (jax.lax.bitcast_convert_type(rounded, jnp.float32),)
```

```text
{ va0: vand.8x128.u32 v24, 0xffff, v23 }
{ va0: vadd.8x128.s32 v25, v24, v22 }
{ va0: vand.8x128.u32 v26, 0xffff0000, v25 ; ... }
```

每个 TC VREG 3 条。实验让每个 lane 一个输入值，512 行是对它的 512 次独立舍入。1 + 2^-10 介于 bf16 的 1 与 1 + 2^-7 之间，f = 1/8：

```text
向上的比例 0.1250，其余都是 1.0：True
```

### 是否无偏

128 个随机的数，量级 2^-20 到 2^20，正负各半，每个舍入 512 次：

```text
512 次的平均值与原值的最大相对误差 3.21e-04；舍入到最近的 bf16 的最大相对误差 3.67e-03
每个结果都是原值两侧相邻的两个 bf16 之一：True
```

单次随机舍入的误差与舍入到最近相当，但 512 次的平均值已经比舍入到最近准确十倍，并随次数继续收敛；舍入到最近的误差则是固定的。负数同样正确：f32 是“符号 + 绝对值”的表示，加法作用在绝对值上，向上进位就是绝对值变大，对正负数对称。

### 特殊输入

位型上的加法对特殊值是否安全，要逐个检查：

| 输入 | 512 次的结果 |
| --- | --- |
| ±inf | ±inf，不变 |
| NaN（`0x7FC00000`） | NaN，不变 |
| ±0 | ±0，不变 |
| f32 的最大有限值 | 全部变成 inf |
| 1e-40（非正规数） | 9.18 × 10⁻⁴¹ × 475、1.84 × 10⁻⁴⁰ × 37 |
| NaN（位型 `0x7F800001`） | 全部变成 inf |

- 无穷大和常见的 NaN 的低 16 位是 0，加上随机数后又被截掉，保持不变。
- f32 的最大有限值大于 bf16 的最大有限值，介于它与无穷大之间，且几乎紧挨着无穷大，所以几乎总是向上溢出成 inf。这与舍入的定义一致，但意味着接近上限的数据要先缩放。
- 非正规数的位型同样是连续的整数，向上的比例 37/512 ≈ 0.072，期望 0.089，在统计误差之内。
- 只有低 16 位非零的 NaN 被截成了 inf。JAX 产生的 NaN 不是这种位型，但处理外部数据时若需要保留所有 NaN，要再加一条 `jnp.where(jnp.isnan(x), x, rounded)`。

## 小结

| 变换 | 每个 TC VREG 的计算指令（不含 `vrng`） | 注意 |
| --- | ---: | --- |
| 均匀分布，尾数构造 | 3（`stateful_uniform` 为 4） | 分辨率 2^-23，含 0 不含 1 |
| 均匀分布，整数转浮点 | 3 | 分辨率 2^-24 |
| 伯努利，整数阈值 | 3（只作为 `vsel` 的条件时 2） | 概率量化到 2^-24 |
| 伯努利，`stateful_bernoulli` | 6 | 概率量化到 2^-23 |
| 指数分布 | 7，EUP 1 条 | 尾部截断在约 15.94 |
| Gumbel 分布 | 11，EUP 2 条 | u = 1 时为 +∞ |
| 正态分布，反误差函数 | 约 77，EUP 2 条 | |
| 正态分布，Box–Muller | 约 121，EUP 1 条 | 正弦、余弦没有 EUP 指令 |
| 随机舍入到 bf16 | 3 | `pltpu.stochastic_round` 在 TPU v4 上不能编译 |

生成比特本身每个 TC VREG 只要一条 `vrng`，而且每 8 个周期才能发射一条（第 6 节）。上表中 8 条以内的变换可以完全藏在 `vrng` 的发射间隔里；正态分布则让瓶颈从生成器转移到变换。
