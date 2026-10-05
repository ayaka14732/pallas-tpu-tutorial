"""主机内存中的输出按页组织（每页 f32[8,128]，4 KiB，共 4 页），kernel 把结果写到第 p 页：比较页号为常数、页号来自运行时标量，以及把输出写成 f32[32,128]、用 pl.multiple_of 声明起始行对齐三种写法。"""
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

def build(mesh: jax.sharding.Mesh, mode: str):
    shape = (32, 128) if mode == 'multiple_of' else (4, 8, 128)
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=(P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    def publish(x: jax.Array, page: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=(pl.MemoryRef(jax.core.ShapedArray(shape, x.dtype), pl.HOST), pltpu.HBM(x.shape, x.dtype)),
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
            if mode == 'multiple_of':
                # 页是 8 行的窗口；起始行来自运行时的标量，用 pl.multiple_of 声明它是 8 的倍数。
                window = pages_host.at[pl.ds(pl.multiple_of(page_smem[0] * 8, 8), 8)]
            else:
                window = pages_host.at[page_smem[0] if mode == 'dynamic' else 2]
            pltpu.async_copy(staging_hbm, window, sem).wait()

        result, _ = kernel(x, page)
        return pltpu.with_memory_space_constraint(result, pl.HOST)

    return publish

def main() -> None:
    mesh = jax.make_mesh((1,), ('device',))
    host = jax.NamedSharding(mesh, P(), memory_kind='pinned_host')
    x = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
    names = {
        'constant': '页号为常数 2',
        'dynamic': '页号来自运行时标量 page[0]，pages_host.at[page]',
        'multiple_of': '页号来自运行时标量 page[0]，输出为 f32[32,128]，pages_host.at[pl.ds(pl.multiple_of(page * 8, 8), 8)]',
    }
    for mode, name in names.items():
        try:
            compiled = tpuasm_tools.compile(build(mesh, mode), x, jnp.array([2, 0], jnp.int32), mesh=mesh, out_shardings=host)
        except Exception as error:
            print(f'## {name}：编译失败：{str(error).splitlines()[0]}')
            continue
        # 运行时的页号只有在 dynamic、multiple_of 两种写法中起作用。
        for number in (2,) if mode == 'constant' else range(4):
            result = np.asarray(compiled(x, jnp.array([number, 0], jnp.int32))).reshape(4, 8, 128)
            np.testing.assert_array_equal(result[number], np.asarray(x))
        print(f'## {name}：数值检查通过')

if __name__ == '__main__':
    main()
