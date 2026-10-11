"""矩阵转置：只改并发个数、dtype、大小和对齐，统计清单中 XLU 的提交与取回指令及其所在的槽。"""
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
from transpose_common import CASES, submit_forms, summary

def transpose_kernel(mesh: jax.sharding.Mesh, count: int):
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def function(xs: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        shape, dtype = xs[0].shape, xs[0].dtype
        out_shape = (shape[1], shape[0])

        @pl.kernel(
            out_type=tuple(jax.ShapeDtypeStruct(out_shape, dtype) for _ in range(count)),
            mesh=tc_mesh,
            scratch_types=(
                [pltpu.VMEM(shape, dtype) for _ in range(count)],
                [pltpu.VMEM(out_shape, dtype) for _ in range(count)],
                pltpu.SemaphoreType.DMA,
            ),
            name='transpose',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(*refs: Ref) -> None:
            x_hbm, o_hbm, (x_vmem, o_vmem, sem) = refs[:count], refs[count:2 * count], refs[2 * count:]
            for i in range(count):
                pltpu.async_copy(x_hbm[i], x_vmem[i], sem).wait()
            for i in range(count):
                o_vmem[i][...] = x_vmem[i][...].T
            for i in range(count):
                pltpu.async_copy(o_vmem[i], o_hbm[i], sem).wait()

        return kernel(*xs)

    return function

def main() -> None:
    mesh = jax.make_mesh((1,), ('device',))
    rng = np.random.default_rng(0)
    for name, dtype, shape, count in CASES:
        xs = tuple(jnp.asarray(rng.integers(-1000, 1000, shape)).astype(dtype) for _ in range(count))
        compiled = tpuasm_tools.compile(transpose_kernel(mesh, count), xs, mesh=mesh)
        results = compiled(xs)
        for x, result in zip(xs, results):
            np.testing.assert_array_equal(np.asarray(result), np.asarray(x).T)
        listing = tpuasm_tools.kernel_listing(compiled)
        print(f'## {name}：数值检查通过')
        print(summary(listing))
        print('  提交指令的写法：')
        print(submit_forms(listing))
        if count == 1 and name in ('i32[128,128]', 'bf16[128,128]'):
            print(listing)
        print()

if __name__ == '__main__':
    main()
