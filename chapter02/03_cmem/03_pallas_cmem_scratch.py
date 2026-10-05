"""公开接口的现状：在 scratch_types 中申请 pltpu.CMEM buffer，经 HBM → CMEM → TC VMEM → HBM 复制一个 f32[8,128]。"""
import tpu_init
tpu_init.initialize_one_chip()

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

import tpuasm_tools

def main() -> None:
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def copy(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=pltpu.HBM(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.CMEM(x.shape, x.dtype), pltpu.VMEM(x.shape, x.dtype), pltpu.SemaphoreType.DMA),
            name='copy',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, x_cmem: Ref, x_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm, x_cmem, sem).wait()
            pltpu.async_copy(x_cmem, x_vmem, sem).wait()
            pltpu.async_copy(x_vmem, o_hbm, sem).wait()

        return kernel(x)

    x = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
    try:
        tpuasm_tools.compile(copy, x, mesh=mesh)
        print('编译通过')
    except Exception as error:
        print('编译失败：' + str(error).splitlines()[0])

if __name__ == '__main__':
    main()
