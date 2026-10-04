"""TC VMEM 有多大：只改 scratch buffer 的行数，看多大的 pltpu.VMEM 还能编译并正确运行。kernel 只用 buffer 的最后 8 行。"""
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

def build(rows: int):
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
            scratch_types=(pltpu.VMEM((rows, 128), x.dtype), pltpu.SemaphoreType.DMA),
            name='scale',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, sem: Ref) -> None:
            # 只用 buffer 的最后 8 行，确认这么高的地址确实可用。
            last = x_vmem.at[pl.ds(rows - 8, 8)]
            pltpu.async_copy(x_hbm, last, sem).wait()
            last[...] = last[...] * 2.0
            pltpu.async_copy(last, o_hbm, sem).wait()

        return kernel(x)

    return mesh, scale

def main() -> None:
    x = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
    for mebibytes in (8, 16, 17):
        rows = mebibytes * 2048
        mesh, scale = build(rows)
        try:
            compiled = tpuasm_tools.compile(scale, x, mesh=mesh)
        except Exception as error:
            print(f'pltpu.VMEM(({rows}, 128), f32)，{mebibytes} MiB：编译失败：{str(error).splitlines()[0].split(" :: ")[0]}')
            continue
        np.testing.assert_array_equal(np.asarray(compiled(x)), np.asarray(x) * 2.0)
        store = [line.split('#')[0].strip() for line in tpuasm_tools.kernel_listing(compiled, pallas_only=True).splitlines() if 'vst:' in line]
        print(f'pltpu.VMEM(({rows}, 128), f32)，{mebibytes} MiB：数值检查通过；{store[0]}')

if __name__ == '__main__':
    main()
