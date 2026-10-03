"""计数器式生成器在 TensorCore 上的写法与代价：kernel 中调用 jax.random（threefry2x32），结果与 kernel 外逐位相同；统计每个 u32[8,128] 需要多少条向量指令，并与 Philox 4x32（需要 32 位乘法）和硬件 vrng 对照。"""
import tpu_init
tpu_init.initialise_one_chip()

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.random import philox
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

import tpuasm_tools

def build(rows: int, method: str):
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def draw(seed: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct((rows, 128), jnp.uint32),
            mesh=tc_mesh,
            scratch_types=(pltpu.SMEM((2,), jnp.int32), pltpu.VMEM((rows, 128), jnp.uint32), pltpu.SemaphoreType.DMA),
            name='draw',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(seed_hbm: Ref, o_hbm: Ref, seed_smem: Ref, bits_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(seed_hbm, seed_smem, sem).wait()
            if method == 'threefry2x32':
                # 与 kernel 外相同的写法：由种子得到 key，按用途 fold_in，再生成比特。
                key = jax.random.fold_in(jax.random.key(seed_smem[0], impl='threefry2x32'), seed_smem[1])
                bits_vmem[...] = jax.random.bits(key, (rows, 128), jnp.uint32)
            elif method == 'philox4x32':
                # 计数器是元素的行号 × 128 + 列号；一次 Philox 产生 4 个 u32，这里取 rows / 4 行的计数器得到 rows 行。
                quarter = rows // 4
                counter = (jax.lax.broadcasted_iota(jnp.uint32, (quarter, 128), 0) * 128 + jax.lax.broadcasted_iota(jnp.uint32, (quarter, 128), 1))
                zero = jnp.zeros((quarter, 128), jnp.uint32)
                words = philox.philox_4x32(counter, zero, zero, zero, seed_smem[0].astype(jnp.uint32), seed_smem[1].astype(jnp.uint32))
                bits_vmem[...] = jnp.concatenate(words, axis=0)
            else:
                pltpu.prng_seed(seed_smem[0], seed_smem[1])
                bits_vmem[...] = pltpu.prng_random_bits((rows, 128)).astype(jnp.uint32)
            pltpu.async_copy(bits_vmem, o_hbm, sem).wait()

        return kernel(seed)

    return mesh, draw

def main() -> None:
    seed = jnp.array([2026, 3], jnp.int32)
    for rows in (8, 64):
        for method in ('threefry2x32', 'philox4x32', '硬件 vrng'):
            mesh, draw = build(rows, method)
            compiled = tpuasm_tools.compile(draw, seed, mesh=mesh)
            bits = np.asarray(compiled(seed))
            listing = tpuasm_tools.kernel_listing(compiled, pallas_only=True)
            counts = tpuasm_tools.count_mnemonics(listing)
            vector = sum(count for name, count in counts.items() if name.startswith('v') and not name.startswith(('vld', 'vst', 'vwait', 'vsync', 'vtrace', 'vnop', 'vdelay')))
            bundles = sum(line.startswith('{') for line in listing.splitlines())
            print(f'## {method}，u32[{rows},128]（{rows // 8} 个 TC VREG）：kernel 段 {bundles} 个 bundle，向量运算 {vector} 条，平均每个 TC VREG {vector / (rows // 8):.1f} 条')
            print('  ' + '，'.join(f'{name} {count}' for name, count in counts.most_common(8)))
            if method == 'threefry2x32':
                reference = jax.random.bits(jax.random.fold_in(jax.random.key(2026, impl='threefry2x32'), 3), (rows, 128), jnp.uint32)
                print(f'  与 kernel 外的 jax.random.bits 逐位相同：{bool(np.array_equal(bits, np.asarray(reference)))}')

if __name__ == '__main__':
    main()
