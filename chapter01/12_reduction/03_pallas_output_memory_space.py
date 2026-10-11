"""同一个沿 lane 求和的 kernel，输出类型分别写成 jax.ShapeDtypeStruct 与 pltpu.HBM，比较 kernel 写回输出的 DMA 目的地。"""
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

def main() -> None:
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)
    x = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
    for name, out_type in (('jax.ShapeDtypeStruct((8, 1), jnp.float32)', jax.ShapeDtypeStruct((8, 1), jnp.float32)), ('pltpu.HBM((8, 1), jnp.float32)', pltpu.HBM((8, 1), jnp.float32))):

        @jax.shard_map(
            mesh=mesh,
            in_specs=P(),
            out_specs=P(),
            check_vma=False,
        )
        def row_sum(x: jax.Array) -> jax.Array:
            @pl.kernel(
                out_type=out_type,
                mesh=tc_mesh,
                scratch_types=(pltpu.VMEM((8, 128), jnp.float32), pltpu.VMEM((8, 1), jnp.float32), pltpu.SemaphoreType.DMA),
                name='row_sum',
                compiler_params=pltpu.CompilerParams(
                    disable_bounds_checks=True,
                    disable_semaphore_checks=True,
                ),
            )
            def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, o_vmem: Ref, sem: Ref) -> None:
                pltpu.async_copy(x_hbm, x_vmem, sem).wait()
                o_vmem[...] = jnp.sum(x_vmem[...], axis=1, keepdims=True)
                pltpu.async_copy(o_vmem, o_hbm, sem).wait()

            return kernel(x)

        compiled = tpuasm_tools.compile(row_sum, x, mesh=mesh)
        np.testing.assert_array_equal(np.asarray(compiled(x)), np.asarray(x).sum(axis=1, keepdims=True))
        print(f'## out_type={name}：数值检查通过；kernel 中的 DMA：')
        print('\n'.join(line.split('#')[0].rstrip() for line in tpuasm_tools.kernel_listing(compiled, pallas_only=True).splitlines() if 'dma.simple' in line))
        print()

if __name__ == '__main__':
    main()
