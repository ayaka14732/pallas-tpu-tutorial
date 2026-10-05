"""单 TC 把一个 f32[8,128] 从 HBM 搬到 TC VMEM、乘以 2、再搬回 HBM；打印完整清单的结构概览和 kernel 段。"""
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
    # runtime 只打开一颗芯片；SPMD 层面再用 TensorCoreMesh 只启用其中一个 TensorCore。
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def scale(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.SemaphoreType.DMA),
            name='scale',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            x_vmem[...] = x_vmem[...] * 2.0
            pltpu.async_copy(x_vmem, o_hbm, sem).wait()

        return kernel(x)

    x = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
    compiled = tpuasm_tools.compile(scale, x, mesh=mesh)
    np.testing.assert_array_equal(np.asarray(compiled(x)), np.asarray(x) * 2.0)
    print('数值检查通过')
    print('\n# 完整清单的结构概览')
    print(tpuasm_tools.listing_outline(compiled))
    listing = tpuasm_tools.kernel_listing(compiled)
    print('\n# kernel 段的指令统计')
    tpuasm_tools.print_mnemonic_counts(listing)
    print('\n# kernel 段')
    print(listing)

if __name__ == '__main__':
    main()
