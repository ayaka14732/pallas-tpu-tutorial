# 计数器式生成器

硬件生成器是顺序式的：第 n 个随机数取决于前面已经生成了多少个。计数器式生成器则把随机数写成一个函数：`随机数 = f(key, 计数器)`，计数器通常是数组中的位置。同一个 key 和位置，在任何设备、任何分块、任何执行顺序下都得到同一个随机数。`jax.random` 默认的 threefry2x32 就是这样的生成器。代价是每个随机数都要独立计算一遍 f，而 TensorCore 没有专门的硬件，只能用向量整数运算拼出来。

## 写法：在 kernel 中使用 jax.random

本小节实验[源码](01_pallas_threefry.py)、[输出](01_pallas_threefry.txt)。

kernel 从 SMEM 读出两个整数，用 `jax.random` 的普通写法生成：

```python
pltpu.async_copy(seed_hbm, seed_smem, sem).wait()
# 与 kernel 外相同的写法：由种子得到 key，按用途 fold_in，再生成比特。
key = jax.random.fold_in(jax.random.key(seed_smem[0], impl='threefry2x32'), seed_smem[1])
bits_vmem[...] = jax.random.bits(key, (rows, 128), jnp.uint32)
```

结果与 kernel 外 `jax.random.bits(jax.random.fold_in(jax.random.key(2026), 3), ...)` 逐位相同。这是计数器式生成器最重要的性质：随机数可以在 kernel 内外、不同的程序之间精确对齐。

## 代价：指令数

只改 `method`，比较三种生成方法的清单：

| 方法 | `u32[8,128]` 的向量运算 | `u32[64,128]` 的向量运算 | 每个 TC VREG |
| --- | ---: | ---: | ---: |
| threefry2x32 | 267 | 1107 | 138 |
| Philox 4x32 | 1162 | 2214 | 277 |
| 硬件 `vrng` | 124 | 131 | 16（其中 `vrng` 1 条） |

- **threefry2x32** 每个 TC VREG 约 138 条向量运算，全部是 32 位加法、异或和循环移位（`vshll`、`vshrl`、`vor` 三条拼成一次循环移位）。`u32[64,128]` 时还多出 136 条 `vld`、134 条 `vst`：中间值超过 TC VREG 的数量，溢出到 TC VMEM。
- **Philox 4x32** 每轮要做两次 32 × 32 → 64 位乘法。第一章第 6 节说过，TPU v4 的向量单元没有整数乘法器；这里乘数是常数，编译器把乘法展开成移位与加减（`vshll` 705、`vadd` 637、`vsub` 438），每个 TC VREG 的代价是 threefry 的两倍。实验用的是 JAX 中 `jax.experimental.pallas.ops.tpu.random.philox` 的 `philox_4x32`，只统计指令，没有与其他实现比较数值。
- **硬件 `vrng`** 每个 TC VREG 只要一条指令；表中的其余约 120 条是 `prng_seed` 的种子混合（第 3 节），只付一次。

所以在 TPU v4 上，计数器式生成器比硬件生成器贵一到两个数量级。第 6 节实测它们的速度。

## 选择

| 需要 | 合适的方法 |
| --- | --- |
| 与 kernel 外的 `jax.random` 逐位一致；结果不随分块、执行顺序、设备数变化 | 计数器式（threefry2x32） |
| 只需要统计上的随机性，生成量大，追求速度 | 硬件生成器，显式设定种子，并把核、芯片编号混进种子 |
| 按位置复现，但能接受只在 Pallas 内部一致 | Pallas key 与 `sample_block`（第 3 节） |

计数器式生成器的“计数器”可以是任何能唯一确定一个随机数的整数组合：数组位置、迭代次数、请求编号、用途编号等。用 `fold_in` 依次把它们混进 key，就得到一个由这些坐标寻址的随机数，不需要保存任何状态。
