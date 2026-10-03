# 从比特到分布

生成器给出的是均匀分布的 32 位整数。实际需要的往往是 `[0, 1)` 中的浮点数、按概率取真假的掩码、正态分布或 Gumbel 分布的噪声，以及把 f32 随机舍入成 bf16。本节看这些变换在 TensorCore 上用哪些指令完成，代价各是多少。

## 实验的写法

本小节实验[源码](01_pallas_distributions.py)、[输出](01_pallas_distributions.txt)。

每种变换是一个函数，在 `prng_seed(1)` 之后返回若干个 `[64,128]` 的数组（8 个 TC VREG），kernel 把它们写回 HBM：

```python
def kernel(*refs: Ref) -> None:
    outputs, buffers, sem = refs[:len(dtypes)], refs[len(dtypes):-1], refs[-1]
    pltpu.prng_seed(1)
    for buffer, value in zip(buffers, sample(), strict=True):
        buffer[...] = value
    ...
```

实验只改 `sample`。下面的指令数都是 8 个 TC VREG 的合计，减去所有变换共有的种子混合（约 120 条）和 8 条 `vrng`，就是变换本身的代价。

## 均匀分布

`pltpu.stateful_uniform` 与手写版本：

```python
def uniform_by_hand():
    # 高 23 位放进 [1, 2) 的尾数，减 1 得到 [0, 1) 中 2^-23 的整数倍。
    u = jax.lax.bitcast_convert_type((bits() >> 9) | jnp.uint32(0x3F800000), jnp.float32) - 1.0
```

指数位为 `0x3F8`、尾数任意的 f32 正好均匀地覆盖 `[1, 2)` 中 2²³ 个等距的值。每个 TC VREG 只要 `vshrl`、`vor`、`vadd.f32` 三条；`stateful_uniform` 与 `jax.random.uniform` 相同，多一条 `vmax`，保证结果不小于 `minval`。两者的输出完全相同：

```text
均值 0.4969，标准差 0.2875，最小 0.000141501，最大 0.999618
全部是 2^-23 的整数倍：True
```

结果是 `[0, 1)` 中 2^-23 的整数倍，可以取到 0，取不到 1。32 位中有 9 位被丢弃。

## 伯努利分布

`pltpu.stateful_bernoulli(0.25, shape)` 先生成均匀分布，再与 p 比较（`vlt.f32`、`vsel`）。概率是 2 的幂的分数时，可以直接比较整数：

```python
def bernoulli_by_integer():
    # 概率 1/4 写成比特的阈值：高 24 位小于 2^22 的概率恰好是 2^22 / 2^24。
    return (((bits() >> 8).astype(jnp.int32) < (1 << 22)).astype(jnp.int32),)
```

每个 TC VREG 是 `vshrl`、`vlt.s32`、`vsel` 三条，比浮点版本少两条。右移 8 位使数值落在有符号整数的非负范围，避免第一章第 6 节中无符号比较的问题。两种写法的 8192 个样本中真的比例都是 0.2533。

## 正态分布

`pltpu.stateful_normal` 与 `jax.random.normal` 相同：先得到 `(-1, 1)` 中的均匀分布，再计算 `√2 · erfinv(u)`。`erfinv` 由多项式近似，分两段，按 `|u|` 选择：

```text
vmul.8x128.f32 168，vadd.8x128.f32 112，vsel.8x128 112，...，EUP 指令 16
```

每个 TC VREG 约 21 次乘法、14 次加法、14 次选择，外加 2 条 EUP 指令，总计约 70 条，是均匀分布的二十多倍。结果的均值 −0.0108、标准差 0.9929。

## Gumbel 分布

`−log(−log u)` 需要两次对数：

```python
def gumbel():
    # 均匀数取 (0, 1]：1 − [0, 1)，避免 log(0)。
    u = 1.0 - uniform_by_hand()[0]
    return u, -jnp.log(-jnp.log(u))
```

每个 TC VREG 两条 `vlog2`（EUP，第一章第 6 节）加两次乘以 ln 2 等少量运算。与主机上 float64 的同一公式相比，最大绝对误差 3.47e-04，来自 EUP 的 `vlog2` 近似；8192 个样本的均值 0.5874（理论值 0.5772）。从 `[0, 1)` 取 `1 − u` 很重要：`u = 0` 时 `log(0) = −∞`，结果是 −∞。

## 随机舍入

把 f32 舍入成 bf16 时，按被舍去部分的大小随机决定向上还是向下，期望值就等于原值。`pltpu.stochastic_round` 在 TPU v4 上不能编译：

```text
pltpu.stochastic_round → bf16：编译失败：Stochastic convert not supported on TPU generations < 5.
```

更新的 TPU 有专门的转换指令；但这个运算本身只需要整数加法：

```python
def stochastic_round_by_hand():
    # bf16 是 f32 的高 16 位：把随机的 16 位加到被舍去的低 16 位上，进位的概率正好等于被舍去部分的比例，再截断。
    raw = jax.lax.bitcast_convert_type(x, jnp.uint32)
    rounded = (raw + (bits() & jnp.uint32(0xFFFF))) & jnp.uint32(0xFFFF0000)
```

每个 TC VREG 两条 `vand`、一条 `vadd`。输入 1 + 2^-10 介于 bf16 的 1 与 1 + 2^-7 之间，应以 1/8 的概率向上：

```text
向上舍入到 1 + 2^-7 的比例 0.1263（理论值 0.125），其余都是 1.0：True
```

这个写法适用于有限的数；对 NaN、无穷大以及进位使指数溢出的情况，需要另外处理。

## 小结

| 变换 | 每个 TC VREG 的指令（不含 `vrng`） |
| --- | --- |
| 均匀分布 | 3（`stateful_uniform` 为 4） |
| 伯努利，整数阈值 | 3（浮点比较为 6） |
| 随机舍入到 bf16 | 3 |
| Gumbel | 约 12，其中 EUP 2 条、取回 2 条 |
| 正态分布 | 约 70，其中 EUP 2 条 |

生成比特本身每个 TC VREG 只要一条 `vrng`，大部分分布的代价在变换上。下一节实测这些代价与生成器的吞吐。
