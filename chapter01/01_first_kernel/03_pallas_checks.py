"""CompilerParams 的两个检查开关：同一 kernel 在默认参数下与关闭检查时，清单多出哪些指令。"""
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
    x = jnp.arange(16 * 128, dtype=jnp.float32).reshape(16, 128)
    settings = (
        ('默认参数', pltpu.CompilerParams()),
        ('只关闭越界检查', pltpu.CompilerParams(disable_bounds_checks=True)),
        ('只关闭信号量检查', pltpu.CompilerParams(disable_semaphore_checks=True)),
        ('两者都关闭', pltpu.CompilerParams(disable_bounds_checks=True, disable_semaphore_checks=True)),
    )
    for name, params in settings:

        @jax.shard_map(
            mesh=mesh,
            in_specs=P(),
            out_specs=P(),
            check_vma=False,
        )
        def scale(x: jax.Array) -> jax.Array:
            @pl.kernel(
                out_type=jax.ShapeDtypeStruct((8, 128), x.dtype),
                mesh=tc_mesh,
                scratch_types=(pltpu.VMEM((8, 128), x.dtype), pltpu.SemaphoreType.DMA),
                name='scale',
                compiler_params=params,
            )
            def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, sem: Ref) -> None:
                pltpu.async_copy(x_hbm.at[pl.ds(8, 8)], x_vmem, sem).wait()
                x_vmem[...] = x_vmem[...] * 2.0
                pltpu.async_copy(x_vmem, o_hbm, sem).wait()

            return kernel(x)

        compiled = tpuasm_tools.compile(scale, x, mesh=mesh)
        np.testing.assert_array_equal(np.asarray(compiled(x)), np.asarray(x)[8:16] * 2.0)
        listing = tpuasm_tools.kernel_listing(compiled)
        counts = tpuasm_tools.count_mnemonics(listing)
        bundles = sum(line.startswith('{') for line in listing.splitlines())
        print(f'## {name}：{bundles} 个 bundle，shalt {counts["shalt"]} 条，数值检查通过')
        print('\n'.join(line[:170] for line in listing.splitlines() if 'shalt' in line or 'Check' in line or 'sflag' in line and 'vwait' in line))
        print()

if __name__ == '__main__':
    main()
