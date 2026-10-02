"""只改 DMA 窗口：从 HBM 中的 f32[64,256] 取行窗口、列窗口和 9 行窗口，经 TC VMEM 写回输出的同一位置。"""
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

# (名称, 行起点, 行数, 列起点, 列数)
WINDOWS = (
    ('行窗口 [16:32, :]', 16, 16, 0, 256),
    ('列窗口 [:, 128:256]', 0, 64, 128, 128),
    ('9 行窗口 [8:17, :]', 8, 9, 0, 256),
    ('非对齐行起点 [3:11, :]', 3, 8, 0, 256),
)

def main() -> None:
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)
    x = jnp.arange(64 * 256, dtype=jnp.float32).reshape(64, 256)
    for name, row, rows, column, columns in WINDOWS:

        @jax.shard_map(
            mesh=mesh,
            in_specs=P(),
            out_specs=P(),
            check_vma=False,
        )
        def window(x: jax.Array) -> jax.Array:
            @pl.kernel(
                out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
                mesh=tc_mesh,
                scratch_types=(pltpu.VMEM((rows, columns), x.dtype), pltpu.SemaphoreType.DMA),
                name='window',
                compiler_params=pltpu.CompilerParams(
                    disable_bounds_checks=True,
                    disable_semaphore_checks=True,
                ),
            )
            def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, sem: Ref) -> None:
                pltpu.async_copy(x_hbm.at[pl.ds(row, rows), pl.ds(column, columns)], x_vmem, sem).wait()
                pltpu.async_copy(x_vmem, o_hbm.at[pl.ds(row, rows), pl.ds(column, columns)], sem).wait()

            return kernel(x)

        try:
            compiled = tpuasm_tools.compile(window, x, mesh=mesh)
        except Exception as error:
            print(f'## {name}：编译失败')
            print(str(error).splitlines()[0])
            print()
            continue
        result = np.asarray(compiled(x))
        expected = np.asarray(x)[row:row + rows, column:column + columns]
        np.testing.assert_array_equal(result[row:row + rows, column:column + columns], expected)
        listing = tpuasm_tools.kernel_listing(compiled)
        print(f'## {name}：窗口内数值检查通过')
        print(listing)
        print()

if __name__ == '__main__':
    main()
