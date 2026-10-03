"""主机内存中的输出按页组织（f32[4,8,128]，每页 4 KiB），kernel 把结果写到第 p 页：比较页号为常数与页号来自运行时标量两种写法。"""
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

def build(mesh: jax.sharding.Mesh, dynamic: bool):
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=(P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    def publish(x: jax.Array, page: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=(pl.MemoryRef(jax.core.ShapedArray((4, 8, 128), x.dtype), pl.HOST), pltpu.HBM(x.shape, x.dtype)),
            mesh=tc_mesh,
            scratch_types=(pltpu.SMEM(page.shape, page.dtype), pltpu.SemaphoreType.DMA),
            name='publish',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, page_hbm: Ref, pages_host: Ref, staging_hbm: Ref, page_smem: Ref, sem: Ref) -> None:
            pltpu.async_copy(page_hbm, page_smem, sem).wait()
            pltpu.async_copy(x_hbm, staging_hbm, sem).wait()
            target = page_smem[0] if dynamic else 2
            pltpu.async_copy(staging_hbm, pages_host.at[target], sem).wait()

        result, _ = kernel(x, page)
        return pltpu.with_memory_space_constraint(result, pl.HOST)

    return publish

def main() -> None:
    mesh = jax.make_mesh((1,), ('device',))
    host = jax.NamedSharding(mesh, P(), memory_kind='pinned_host')
    x = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
    page = jnp.array([2, 0], jnp.int32)
    for dynamic in (False, True):
        name = '页号来自运行时标量 page[0]' if dynamic else '页号为常数 2'
        try:
            compiled = tpuasm_tools.compile(build(mesh, dynamic), x, page, mesh=mesh, out_shardings=host)
        except Exception as error:
            print(f'## {name}：编译失败：{str(error).splitlines()[0]}')
            continue
        result = np.asarray(compiled(x, page))
        np.testing.assert_array_equal(result[2], np.asarray(x))
        print(f'## {name}：第 2 页的数值检查通过')

if __name__ == '__main__':
    main()
