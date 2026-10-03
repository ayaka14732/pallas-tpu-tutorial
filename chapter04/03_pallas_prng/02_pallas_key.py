"""Pallas key：用 pltpu.to_pallas_key 把 JAX key 变成硬件生成器的 key，在 kernel 中用 jax.random 的接口采样。检查同一个 key 的两次采样、fold_in、split，以及 sample_block 对分块方式的不变性。"""
import tpu_init
tpu_init.initialise_one_chip()

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax._src.pallas.mosaic import primitives as tpu_primitives
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

import tpuasm_tools

SHAPE = (16, 256)
TILE = (8, 128)

def build(sample):
    """sample(key) 返回 4 个 u32[16,256]；key 的两个 u32 经 DMA 进入 SMEM，在 kernel 中拼回 Pallas key。"""
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def draw(key_data: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct((4, *SHAPE), jnp.uint32),
            mesh=tc_mesh,
            scratch_types=(pltpu.SMEM((1, 2), jnp.uint32), pltpu.VMEM((4, *SHAPE), jnp.uint32), pltpu.SemaphoreType.DMA),
            name='draw',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(key_hbm: Ref, o_hbm: Ref, key_smem: Ref, bits_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(key_hbm, key_smem, sem).wait()
            # SMEM 只能逐个读出标量；wrap_pallas_seed 把两个标量拼成 Pallas key（fold_in 内部也用它）。
            key = tpu_primitives.wrap_pallas_seed(key_smem[0, 0], key_smem[0, 1], impl='pallas_tpu')
            for index, value in enumerate(sample(key)):
                bits_vmem[index] = value
            pltpu.async_copy(bits_vmem, o_hbm, sem).wait()

        return kernel(key_data)

    return mesh, draw

def repeated(key: jax.Array) -> list[jax.Array]:
    return [
        jax.random.bits(key, SHAPE, jnp.uint32),
        jax.random.bits(key, SHAPE, jnp.uint32),
        jax.random.bits(jax.random.fold_in(key, 1), SHAPE, jnp.uint32),
        jax.random.bits(jax.random.fold_in(key, 2), SHAPE, jnp.uint32),
    ]

def blocked(key: jax.Array) -> list[jax.Array]:
    sample = lambda block, index: pltpu.sample_block(jax.random.bits, key, block_size=block, tile_size=TILE, total_size=SHAPE, block_index=index, dtype=jnp.uint32)
    whole = sample(SHAPE, (0, 0))
    # 同一个 [16,256] 按 4 个 [8,128] 分块生成，按与上面相反的顺序拼回。
    parts = {(i, j): sample(TILE, (i, j)) for i in (1, 0) for j in (1, 0)}
    tiles = jnp.concatenate([jnp.concatenate([parts[i, j] for j in (0, 1)], axis=1) for i in (0, 1)], axis=0)
    # 按 [16,128] 分块。
    halves = jnp.concatenate([sample((16, 128), (0, j)) for j in (0, 1)], axis=1)
    return [whole, tiles, halves, jnp.zeros(SHAPE, jnp.uint32)]

def main() -> None:
    key = pltpu.to_pallas_key(jax.random.key(0))
    key_data = jax.random.key_data(key)
    print(f'Pallas key 的数据：shape {key_data.shape}，dtype {key_data.dtype}')
    mesh, draw = build(repeated)
    compiled = tpuasm_tools.compile(draw, key_data, mesh=mesh)
    out = np.asarray(compiled(key_data))
    counts = tpuasm_tools.count_mnemonics(tpuasm_tools.kernel_listing(compiled, pallas_only=True))
    print(f'## 同一个 key 采样两次，再分别 fold_in(key, 1)、fold_in(key, 2)：setrngseed {counts["setrngseed"]} 条，vrng {counts["vrng.8x128.u32"]} 条')
    print(f'  两次采样相同：{bool(np.array_equal(out[0], out[1]))}；fold_in 1 与原 key 不同：{not np.array_equal(out[0], out[2])}；fold_in 1 与 2 不同：{not np.array_equal(out[2], out[3])}')
    try:
        jax.random.split(key)
    except NotImplementedError as error:
        print(f'## split：NotImplementedError：{error}')
    mesh, draw = build(blocked)
    out = np.asarray(tpuasm_tools.compile(draw, key_data, mesh=mesh)(key_data))
    print(f'## sample_block：整块与 [8,128] 分块相同 {bool(np.array_equal(out[0], out[1]))}；整块与 [16,128] 分块相同 {bool(np.array_equal(out[0], out[2]))}')

if __name__ == '__main__':
    main()
