# 计数器式生成器

硬件生成器是顺序式的：第 n 个随机数取决于此前已经生成了多少个。计数器式生成器则把随机数写成一个函数：`随机数 = f(key, 计数器)`，计数器通常就是元素在数组中的位置。同一个 key 和位置，在任何设备、任何分块、任何执行顺序下都得到同一个随机数，不需要保存任何状态。`jax.random` 默认的 threefry2x32 就是这样的生成器。代价是每个随机数都要独立算一遍 f，而 TensorCore 没有专门的硬件，只能用向量整数运算拼出来。本节看这个代价有多大、来自哪里，以及“计数器 = 位置”在 JAX 中具体是什么意思。

## 写法：在 kernel 中使用 jax.random

本小节实验[源码](01_pallas_threefry.py)、[输出](01_pallas_threefry.txt)。

kernel 从 SMEM 读出两个整数，用 `jax.random` 的普通写法生成：

```python
pltpu.async_copy(seed_hbm, seed_smem, sem).wait()
# 与 kernel 外相同的写法：由种子得到 key，按用途 fold_in，再生成比特。
key = jax.random.fold_in(jax.random.key(seed_smem[0], impl='threefry2x32'), seed_smem[1])
bits_vmem[...] = jax.random.bits(key, (rows, 128), jnp.uint32)
```

`seed_smem[0]` 是运行时从 SMEM 读出的标量，`jax.random.key` 与 `fold_in` 都可以接受它。实验对 `rows` = 8、16、32、64、128 五种大小，结果都与 kernel 外 `jax.random.bits(jax.random.fold_in(jax.random.key(2026), 3), ...)` 逐位相同。这是计数器式生成器最重要的性质：随机数可以在 kernel 内外、不同程序之间精确对齐，用 kernel 外的 `jax.random` 就能检验 kernel 的随机数。

## threefry2x32 由什么组成

threefry2x32 对两个 32 位的字 `(x0, x1)` 做 20 轮混合（JAX 的实现在 `jax/_src/random/threefry2x32.py`）。每一轮：

```python
def apply_round(v, rot):
    v[0] = v[0] + v[1]
    v[1] = rotate_left(v[1], rot)
    v[1] = v[0] ^ v[1]
```

循环移位量按 `13, 15, 26, 6` 和 `17, 29, 16, 24` 交替；每 4 轮把 key 的一个字和轮次编号加进去。key 有两个字 `k0、k1`，再加上第三个字 `k0 ^ k1 ^ 0x1BD11BDA`。

当前 JAX 默认 `jax_threefry_partitionable = True`。这时 `jax.random.bits(key, shape)` 给每个元素一个 64 位的计数器，即它按行优先展平后的序号，拆成高、低两个 32 位字作为 `(x0, x1)`，hash 之后把两个输出字异或得到这个元素的 32 位随机数。所以每个输出元素要做一次完整的 20 轮 hash。

TensorCore 没有循环移位指令，`rotate_left(v, r)` 要拆成三条：`vshll` 左移 r 位、`vshrl` 右移 32 − r 位、`vor` 合并。于是每一轮 5 条向量指令。清单中 hash 的一段：

```text
{ va0: vor.8x128.u32 v11, v10, v8 }
{ va0: vxor.8x128.u32 v12, v11, v7 }
{ va0: vadd.8x128.s32 v13, v12, v7 ; va1: vshll.8x128.s32 v15, v12, 0x6 }
{ va0: vadd.8x128.s32 v20, v13, v4 ; va1: vshrl.8x128.s32 v16, v12, 0x1a }
{ va0: vor.8x128.u32 v9, v16, v15 }
{ va0: vxor.8x128.u32 v17, v9, v13 }
{ va0: vadd.8x128.s32 v18, v17, v14 ; va1: vadd.8x128.s32 v14, v26, v25 }
{ va0: vadd.8x128.s32 v19, 5, v18 }
```

`vshll 0x6` 与 `vshrl 0x1a`（26）组成循环左移 6 位，随后 `vor`、`vxor`；最后的 `vadd 5` 是第 5 次 key 注入中的轮次编号。每一轮都依赖上一轮的结果，只算一个 TC VREG 时，两个向量 ALU 槽大部分时间只有一个在工作。

按算法估计，每个输出 TC VREG：20 轮 × 5 条 = 100 条，加 6 次 key 注入约 15 条，加计数器与最后的异或，约 120 条。

## 只改行数：代价随大小怎样变化

| `rows` | threefry2x32 计算指令 | `vld` + `vst` | Philox 4x32 计算指令 | 硬件 `vrng` 计算指令 |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 267 | 1 | 1162 | 124 |
| 16 | 387 | 2 | 1154 | 125 |
| 32 | 627 | 15 | 1143 | 127 |
| 64 | 1107 | 270 | 2214 | 131 |
| 128 | 2191 | 1441 | 4363 | 139 |

“计算指令”是向量运算，不含访存与同步。从 64 行到 128 行，每多一个 TC VREG：

- **threefry2x32 增加 135.5 条**，与上面约 120 条的估计相符。`rows = 8` 时的 267 条中，约 130 条是固定开销：`key` 与 `fold_in` 本身也是一次 threefry hash，作用在两个标量上；清单中这部分还用到了 XLU 的 `vrot` 与 `spop`，把 key 的两个字从 lane 中取出。
- **Philox 4x32 增加 268.6 条**，是 threefry 的两倍。Philox 每轮要做两次 32 × 32 → 64 位乘法；第一章第 6 节说过，TPU v4 的向量单元没有整数乘法器，这里乘数是常数，编译器把乘法展开成移位与加减。小尺寸时（8–32 行）Philox 的指令数几乎不变：实验每次 hash 产生 4 个字，`rows` 很小时只用到一个 TC VREG 的一部分，却仍要算满整个 TC VREG。实验用的是 JAX 中 `jax.experimental.pallas.ops.tpu.random.philox` 的 `philox_4x32`，只统计指令，没有与其他实现比较数值。
- **硬件 `vrng` 增加 1 条**，就是 `vrng` 本身；约 120 条的种子混合只付一次（第 3 节）。

`vld` + `vst` 一列说明了另一个问题。threefry 对整个数组逐元素计算，Mosaic 把所有行同时展开：`rows = 128` 时 16 个输出 TC VREG 各自需要 `x0`、`x1` 和中间值，远超 TC VREG 的数量，溢出到 TC VMEM，读写多达 1441 条。所以计数器式生成器应当分块调用，每次只生成几十行；第 6 节的计时就是每次 64 行。

## 计数器是展平后的序号

计数器式生成器常被说成“按位置寻址”。在 JAX 中，“位置”的确切含义是：元素在整个数组中按行优先展平后的序号。实验在 kernel 外比较不同形状的结果：

```text
bits((16,128)) 的前 8 行等于 bits((8,128))：True
bits((8,256)) 的前 128 列等于 bits((8,128))：False
bits((8,256)) 等于 bits((2048,)) 按行排成 [8,256]：True
```

增加行数时，前面各行的序号不变，随机数也不变；改变列数时，同一个 `(行, 列)` 的序号变了，随机数也就不同。所以：

- 同一个 key 生成的数组，只要形状相同，任何分块都能按序号复现它的任何部分。
- 形状不同的两个数组，不能指望同一个 `(行, 列)` 得到相同的随机数。要按逻辑坐标对齐，应当把坐标显式地 `fold_in` 进 key，或者用第 3 节的 `sample_block`，它把全局形状和 tile 的编号都固定下来。

## 选择

| 需要 | 合适的方法 |
| --- | --- |
| 与 kernel 外的 `jax.random` 逐位一致；结果不随分块、执行顺序、设备数变化 | 计数器式（threefry2x32），分块生成以避免溢出 |
| 只需要统计上的随机性，生成量大，追求速度 | 硬件生成器，显式设定种子，并把核、芯片编号混进种子 |
| 按位置复现，但只需在 Pallas 内部一致 | Pallas key 与 `sample_block`（第 3 节） |

计数器式生成器的“计数器”可以是任何能唯一确定一个随机数的整数组合：数组位置、迭代次数、请求编号、用途编号等。用 `fold_in` 依次把它们混进 key，就得到一个由这些坐标寻址的随机数；每次 `fold_in` 是一次作用在标量上的 hash，代价只付一次。
