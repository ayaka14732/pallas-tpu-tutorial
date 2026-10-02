"""单 TC 只做 DMA：f32[64,128] 经 TC VMEM 原样搬回 HBM，以及不经 TC VMEM 的 HBM→HBM 直接复制。"""
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
    params = pltpu.CompilerParams(
        disable_bounds_checks=True,
        disable_semaphore_checks=True,
    )

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def through_vmem(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.SemaphoreType.DMA),
            name='through_vmem',
            compiler_params=params,
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            pltpu.async_copy(x_vmem, o_hbm, sem).wait()

        return kernel(x)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def hbm_to_hbm(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.SemaphoreType.DMA,),
            name='hbm_to_hbm',
            compiler_params=params,
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm, o_hbm, sem).wait()

        return kernel(x)

    x = jnp.arange(64 * 128, dtype=jnp.float32).reshape(64, 128)
    for name, function in (('经 TC VMEM', through_vmem), ('HBM→HBM', hbm_to_hbm)):
        compiled = tpuasm_tools.compile(function, x, mesh=mesh)
        np.testing.assert_array_equal(np.asarray(compiled(x)), np.asarray(x))
        listing = tpuasm_tools.kernel_listing(compiled)
        print(f'## {name}：数值检查通过')
        tpuasm_tools.print_mnemonic_counts(listing)
        print(listing)
        print()

if __name__ == '__main__':
    main()
