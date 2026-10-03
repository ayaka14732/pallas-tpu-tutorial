# Pallas 的硬件随机数接口

Pallas 提供两层接口使用第 1 节的硬件生成器：有状态的 `pltpu.prng_seed` 与 `pltpu.prng_random_bits`，以及把 key 当作参数的 Pallas key。本节看它们在清单中变成什么，与第 1、2 节的硬件行为怎样对应。

## prng_seed 与 prng_random_bits

本小节实验[源码](01_pallas_prng_seed.py)、[输出](01_pallas_prng_seed.txt)。

kernel 设定种子、生成一个 `u32[rows,128]`、写回 HBM：

```python
def kernel(o_hbm: Ref, bits_vmem: Ref, sem: Ref) -> None:
    core = jax.lax.axis_index('tc')
    pltpu.prng_seed(*seeds(core))
    bits_vmem[...] = pltpu.prng_random_bits((rows, 128)).astype(jnp.uint32)
    pltpu.async_copy(bits_vmem, o_hbm.at[pl.ds(core * rows, rows)], sem).wait()
```

`pltpu.prng_seed` 接受一个或多个整数，`pltpu.prng_random_bits(shape)` 返回该形状的 32 位随机比特（类型为 int32，这里转成 uint32）。

`prng_seed(7)` 与 `rows = 8` 的清单：

```text
kernel 段 119 个 bundle，setrngseed 之前 97 个
setrngseed 之前的指令：vadd.8x128.s32 36，vxor.8x128.u32 21，vshll.8x128.s32 20，vshrl.8x128.s32 20，vor.8x128.u32 20，...，vlaneseq.8x128.u32 1
{ va0: setrngseed v21 }；{ va0: vrng.8x128.u32 v22 }
```

`prng_seed` 不是把整数直接装进状态。编译器用 `vlaneseq` 得到每个 lane 的编号，与种子一起经过约 120 条加法、异或、移位组成的混合运算（移位与 `vor` 成对出现，是 32 位的循环移位），算出 64 个生成器各不相同的状态，再 `setrngseed`。之后每 8 行一条 `vrng`：`rows = 64` 时有 8 条 `vrng`、1 条 `setrngseed`。

为了确认输出就是第 1 节的生成器，实验把最后一条 `vrng` 换成 `getrngseed`，kernel 写出的就成了混合运算算出的状态；把这个状态交给主机模型：

```text
读回的状态经 xorshift128+ 模型得到的 tile 与 kernel 的输出一致：True
```

混合运算保证了种子 0 不会产生全零状态：

```text
prng_seed(0)：输出中 0 的个数 0 / 1024
```

### 两个 TensorCore

只改 `num_cores` 和种子：

```text
两个 TensorCore 都 prng_seed(7)：两个 TensorCore 的输出相同 True
prng_seed(7, core)：两个 TensorCore 的输出相同 False
```

与第 2 节的硬件行为一致：`prng_seed` 的混合运算只用 lane 编号和种子，不会自动加入核编号。多个 TensorCore 要得到不同的序列，必须像 `prng_seed(7, core)` 这样把 `jax.lax.axis_index('tc')` 写进种子；多颗芯片同理，加入 `jax.lax.axis_index('device')`。

`prng_seed` 之后，状态随每条 `vrng` 推进，结果取决于此前生成了多少次。在循环中生成时，只在循环之前设一次种子；每次迭代都设同一个种子，每次迭代就会得到相同的随机数。

## Pallas key

本小节实验[源码](02_pallas_key.py)、[输出](02_pallas_key.txt)。

`pltpu.to_pallas_key(jax.random.key(0))` 把 JAX key 变成一个 `u32[1,2]` 的 Pallas key。在 kernel 中，它可以像 JAX key 一样传给 `jax.random.bits`、`uniform`、`bernoulli`、`normal`；每次采样都先用 key 的两个字执行一次 `prng_seed`，再用 `vrng` 生成。

`pl.kernel` 不能直接接收这种类型的数组（Mosaic 报 `Failed to set window params for input 0`），所以实验把 key 的两个 u32 搬进 SMEM，在 kernel 中拼回 key：

```python
pltpu.async_copy(key_hbm, key_smem, sem).wait()
# SMEM 只能逐个读出标量；wrap_pallas_seed 把两个标量拼成 Pallas key（fold_in 内部也用它）。
key = tpu_primitives.wrap_pallas_seed(key_smem[0, 0], key_smem[0, 1], impl='pallas_tpu')
```

`wrap_pallas_seed` 来自 `jax._src.pallas.mosaic.primitives`，不是公开接口。结果：

```text
同一个 key 采样两次，再分别 fold_in(key, 1)、fold_in(key, 2)：setrngseed 4 条，vrng 16 条
两次采样相同：True；fold_in 1 与原 key 不同：True；fold_in 1 与 2 不同：True
split：NotImplementedError：Cannot split a Pallas key. Use fold_in instead to generate new keys.
```

四次采样各有一条 `setrngseed`：每次采样都重新设定状态，所以结果只取决于 key，与此前生成过什么无关。代价是每次都要付一次种子混合运算。Pallas key 不支持 `split`，要派生新的 key 只能用 `fold_in(key, data)`；它把 `data` 加到 key 的第二个字上，再做一轮简单的混合。

### 与分块方式无关的采样

`pltpu.sample_block` 把整个数组按固定的 `tile_size` 划分，每个 tile 用 `fold_in(key, tile 的全局编号)` 采样。于是一个块无论多大、按什么顺序生成，同一位置的随机数都相同：

```python
sample = lambda block, index: pltpu.sample_block(jax.random.bits, key, block_size=block, tile_size=TILE, total_size=SHAPE, block_index=index, dtype=jnp.uint32)
```

```text
sample_block：整块与 [8,128] 分块相同 True；整块与 [16,128] 分块相同 True
```

这把随机数从“生成器的第几步”变成了“数组中的哪个位置”：只要位置相同，随机数就相同。下一节的计数器式生成器把这一思路贯彻到底。
