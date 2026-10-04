"""SMEM 有多大：只改 SMEM scratch buffer 的字数，看多大的 pltpu.SMEM 还能编译并正确运行。kernel 把一个标量存进 buffer 的最后一个字，再读出来参与计算。"""
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

def build(words: int):
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def shift(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.SMEM((words,), jnp.float32), pltpu.SemaphoreType.DMA),
            name='shift',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, s_smem: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            s_smem[words - 1] = 7.0
            x_vmem[...] = x_vmem[...] + s_smem[words - 1]
            pltpu.async_copy(x_vmem, o_hbm, sem).wait()

        return kernel(x)

    return mesh, shift

def main() -> None:
    x = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
    for words in (1024, 131072, 262144):
        mesh, shift = build(words)
        try:
            compiled = tpuasm_tools.compile(shift, x, mesh=mesh)
        except Exception as error:
            print(f'pltpu.SMEM(({words},), f32)，{words * 4 // 1024} KiB：编译失败：{str(error).splitlines()[0]}')
            continue
        np.testing.assert_array_equal(np.asarray(compiled(x)), np.asarray(x) + 7.0)
        print(f'pltpu.SMEM(({words},), f32)，{words * 4 // 1024} KiB：数值检查通过')

if __name__ == '__main__':
    main()
