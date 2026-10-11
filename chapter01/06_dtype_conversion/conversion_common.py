"""本节各实验共用：把整个数组读进 TC VMEM、用 astype 转换后写回的 kernel。"""
import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

import tpuasm_tools

def convert(mesh: jax.sharding.Mesh, x: jax.Array, dtype: jnp.dtype) -> jax.stages.Compiled:
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def function(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct(x.shape, dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.VMEM(x.shape, dtype), pltpu.SemaphoreType.DMA),
            name='convert',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, o_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            o_vmem[...] = x_vmem[...].astype(dtype)
            pltpu.async_copy(o_vmem, o_hbm, sem).wait()

        return kernel(x)

    return tpuasm_tools.compile(function, x, mesh=mesh)
