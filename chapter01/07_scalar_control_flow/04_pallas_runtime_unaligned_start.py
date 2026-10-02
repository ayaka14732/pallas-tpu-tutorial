"""运行时的起点不必与 tile 对齐：从 f32[64,128] 的第 p[0] 行起取 8 行，分别用 DMA（HBM→TC VMEM）和 vld（TC VMEM→TC VREG）实现。"""
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
        in_specs=(P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    def by_dma(x: jax.Array, p: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct((8, 128), x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM((8, 128), x.dtype), pltpu.SMEM(p.shape, p.dtype), pltpu.SemaphoreType.DMA),
            name='by_dma',
            compiler_params=params,
        )
        def kernel(x_hbm: Ref, p_hbm: Ref, o_hbm: Ref, o_vmem: Ref, p_smem: Ref, sem: Ref) -> None:
            pltpu.async_copy(p_hbm, p_smem, sem).wait()
            pltpu.async_copy(x_hbm.at[pl.ds(p_smem[0], 8)], o_vmem, sem).wait()
            pltpu.async_copy(o_vmem, o_hbm, sem).wait()

        return kernel(x, p)

    @jax.shard_map(
        mesh=mesh,
        in_specs=(P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    def by_vld(x: jax.Array, p: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct((8, 128), x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.VMEM((8, 128), x.dtype), pltpu.SMEM(p.shape, p.dtype), pltpu.SemaphoreType.DMA),
            name='by_vld',
            compiler_params=params,
        )
        def kernel(x_hbm: Ref, p_hbm: Ref, o_hbm: Ref, x_vmem: Ref, o_vmem: Ref, p_smem: Ref, sem: Ref) -> None:
            pltpu.async_copy(p_hbm, p_smem, sem).wait()
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            o_vmem[...] = x_vmem[pl.ds(p_smem[0], 8)]
            pltpu.async_copy(o_vmem, o_hbm, sem).wait()

        return kernel(x, p)

    x = jnp.arange(64 * 128, dtype=jnp.float32).reshape(64, 128)
    host = np.asarray(x)
    for name, function in (('DMA', by_dma), ('vld', by_vld)):
        compiled = tpuasm_tools.compile(function, x, jnp.array([0, 0], jnp.int32), mesh=mesh)
        for start in (8, 3, 13, 56):
            result = np.asarray(compiled(x, jnp.array([start, 0], jnp.int32)))
            np.testing.assert_array_equal(result, host[start:start + 8])
        listing = tpuasm_tools.kernel_listing(compiled)
        print(f'## {name}：起点 8、3、13、56 的结果都等于 x[start:start+8]')
        print('\n'.join(line for line in listing.splitlines() if any(key in line for key in ('dma.simple', 'vld:', '[smem:0x3e]', 'dma_start', '[get;'))))
        print()

if __name__ == '__main__':
    main()
