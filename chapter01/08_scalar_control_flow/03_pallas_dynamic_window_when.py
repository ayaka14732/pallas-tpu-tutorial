"""运行时的标量决定 DMA 窗口和是否计算：从 f32[64,128] 中取第 p[0] 个行 tile，p[1] > 0 时才乘以 2。"""
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

    @jax.shard_map(
        mesh=mesh,
        in_specs=(P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    def select(x: jax.Array, p: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct((8, 128), x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM((8, 128), x.dtype), pltpu.SMEM(p.shape, p.dtype), pltpu.SemaphoreType.DMA),
            name='select',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, p_hbm: Ref, o_hbm: Ref, x_vmem: Ref, p_smem: Ref, sem: Ref) -> None:
            pltpu.async_copy(p_hbm, p_smem, sem).wait()
            # DMA 窗口的起点来自运行时的标量。
            start = p_smem[0] * 8
            pltpu.async_copy(x_hbm.at[pl.ds(start, 8)], x_vmem, sem).wait()

            @pl.when(p_smem[1] > 0)
            def _() -> None:
                x_vmem[...] = x_vmem[...] * 2.0

            pltpu.async_copy(x_vmem, o_hbm, sem).wait()

        return kernel(x, p)

    x = jnp.arange(64 * 128, dtype=jnp.float32).reshape(64, 128)
    p = jnp.array([3, 1], jnp.int32)
    compiled = tpuasm_tools.compile(select, x, p, mesh=mesh)
    host = np.asarray(x)
    for tile, flag in ((3, 1), (5, 0), (7, 1)):
        result = np.asarray(compiled(x, jnp.array([tile, flag], jnp.int32)))
        np.testing.assert_array_equal(result, host[tile * 8:tile * 8 + 8] * (2.0 if flag else 1.0))
        print(f'p = [{tile}, {flag}]：数值检查通过')
    listing = tpuasm_tools.kernel_listing(compiled)
    print(listing)

if __name__ == '__main__':
    main()
