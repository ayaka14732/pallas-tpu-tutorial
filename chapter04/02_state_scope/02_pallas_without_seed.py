"""不调用 prng_seed 就使用 prng_random_bits：kernel 拿到的是 runtime 前导装入的状态。比较连续两次调用、两个 TensorCore 的输出，以及设定种子之后的情况。"""
import tpu_init
tpu_init.initialize_one_chip()

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

import tpuasm_tools

def build(seeded: bool):
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=2)

    @jax.shard_map(
        mesh=mesh,
        in_specs=(),
        out_specs=P(),
        check_vma=False,
    )
    def draw() -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct((16, 128), jnp.uint32),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM((8, 128), jnp.uint32), pltpu.SemaphoreType.DMA),
            name='draw',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(o_hbm: Ref, bits_vmem: Ref, sem: Ref) -> None:
            core = jax.lax.axis_index('tc')
            if seeded:
                pltpu.prng_seed(7, core)
            bits_vmem[...] = pltpu.prng_random_bits((8, 128)).astype(jnp.uint32)
            pltpu.async_copy(bits_vmem, o_hbm.at[pl.ds(core * 8, 8)], sem).wait()

        return kernel()

    return mesh, draw

def main() -> None:
    for seeded in (False, True):
        mesh, draw = build(seeded)
        compiled = tpuasm_tools.compile(draw, mesh=mesh)
        first, second = (np.asarray(compiled()).reshape(2, 8, 128) for _ in range(2))
        counts = tpuasm_tools.count_mnemonics(tpuasm_tools.kernel_listing(compiled, pallas_only=True))
        print(f'## {"prng_seed(7, core)" if seeded else "不调用 prng_seed"}：kernel 段 setrngseed {counts.get("setrngseed", 0)} 条，vrng {counts.get("vrng.8x128.u32", 0)} 条')
        print(f'  连续两次调用，TensorCore 0 的输出相同：{bool(np.array_equal(first[0], second[0]))}')
        print(f'  同一次调用，两个 TensorCore 的输出相同：{bool(np.array_equal(first[0], first[1]))}')

if __name__ == '__main__':
    main()
