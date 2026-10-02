"""两个 f32[64,128] 输入先后发起 DMA，再分别等待，然后相加；比较两个输入共用一个 DMA semaphore 与各用一个的写法。"""
import tpu_init
tpu_init.initialise_one_chip()

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
    x = jnp.arange(64 * 128, dtype=jnp.float32).reshape(64, 128)
    y = jnp.ones((64, 128), jnp.float32)
    for semaphores in (1, 2):

        @jax.shard_map(
            mesh=mesh,
            in_specs=(P(), P()),
            out_specs=P(),
            check_vma=False,
        )
        def add(x: jax.Array, y: jax.Array) -> jax.Array:
            @pl.kernel(
                out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
                mesh=tc_mesh,
                scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.VMEM(y.shape, y.dtype), pltpu.SemaphoreType.DMA((semaphores,))),
                name='add',
                compiler_params=pltpu.CompilerParams(
                    disable_bounds_checks=True,
                    disable_semaphore_checks=True,
                ),
            )
            def kernel(x_hbm: Ref, y_hbm: Ref, o_hbm: Ref, x_vmem: Ref, y_vmem: Ref, sems: Ref) -> None:
                # 两次 start 连续发出，DMA 引擎可以同时搬运；之后才开始等待。
                x_copy = pltpu.async_copy(x_hbm, x_vmem, sems.at[0])
                y_copy = pltpu.async_copy(y_hbm, y_vmem, sems.at[semaphores - 1])
                x_copy.wait()
                y_copy.wait()
                x_vmem[...] = x_vmem[...] + y_vmem[...]
                pltpu.async_copy(x_vmem, o_hbm, sems.at[0]).wait()

            return kernel(x, y)

        compiled = tpuasm_tools.compile(add, x, y, mesh=mesh)
        np.testing.assert_array_equal(np.asarray(compiled(x, y)), np.asarray(x) + 1.0)
        listing = tpuasm_tools.kernel_listing(compiled)
        print(f'## {semaphores} 个 DMA semaphore：数值检查通过')
        print('\n'.join(line for line in listing.splitlines() if 'dma.' in line or 'sflag:5' in line))
        print()

if __name__ == '__main__':
    main()
