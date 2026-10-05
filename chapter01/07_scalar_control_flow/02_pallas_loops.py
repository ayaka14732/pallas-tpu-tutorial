"""逐 tile 处理 f32[64,128]：Python for 循环、静态边界的 pl.loop、展开 4 次的 pl.loop、运行时边界的 pl.loop，比较清单中的循环形式。"""
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

PARAMS = pltpu.CompilerParams(
    disable_bounds_checks=True,
    disable_semaphore_checks=True,
)

def main() -> None:
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)
    x = jnp.arange(64 * 128, dtype=jnp.float32).reshape(64, 128)
    # 循环次数放在 n[0]；数组取 2 个元素，避免 XLA 把单元素数组当作标量直接传入 kernel。
    n = jnp.array([8, 0], jnp.int32)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def unrolled(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.SemaphoreType.DMA),
            name='unrolled',
            compiler_params=PARAMS,
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            # Python 循环在 tracing 时展开成 8 份，每份的起点都是常数。
            for i in range(8):
                x_vmem[pl.ds(i * 8, 8)] = x_vmem[pl.ds(i * 8, 8)] * 2.0
            pltpu.async_copy(x_vmem, o_hbm, sem).wait()

        return kernel(x)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def static_loop(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.SemaphoreType.DMA),
            name='static_loop',
            compiler_params=PARAMS,
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()

            @pl.loop(0, 8)
            def _(i: jax.Array) -> None:
                # i 是运行时的循环变量，起点 i * 8 也是运行时的值。
                start = i * 8
                x_vmem[pl.ds(start, 8)] = x_vmem[pl.ds(start, 8)] * 2.0

            pltpu.async_copy(x_vmem, o_hbm, sem).wait()

        return kernel(x)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def unrolled_loop(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.SemaphoreType.DMA),
            name='unrolled_loop',
            compiler_params=PARAMS,
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()

            # unroll=4：每次迭代处理 4 个 tile，循环只执行 2 次。
            @pl.loop(0, 8, unroll=4)
            def _(i: jax.Array) -> None:
                start = i * 8
                x_vmem[pl.ds(start, 8)] = x_vmem[pl.ds(start, 8)] * 2.0

            pltpu.async_copy(x_vmem, o_hbm, sem).wait()

        return kernel(x)

    @jax.shard_map(
        mesh=mesh,
        in_specs=(P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    def dynamic_loop(x: jax.Array, n: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.SMEM(n.shape, n.dtype), pltpu.SemaphoreType.DMA),
            name='dynamic_loop',
            compiler_params=PARAMS,
        )
        def kernel(x_hbm: Ref, n_hbm: Ref, o_hbm: Ref, x_vmem: Ref, n_smem: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            # 循环次数来自输入：先搬进 SMEM，再由标量单元读出。
            pltpu.async_copy(n_hbm, n_smem, sem).wait()

            @pl.loop(0, n_smem[0])
            def _(i: jax.Array) -> None:
                start = i * 8
                x_vmem[pl.ds(start, 8)] = x_vmem[pl.ds(start, 8)] * 2.0

            pltpu.async_copy(x_vmem, o_hbm, sem).wait()

        return kernel(x, n)

    for name, function, arguments in (('Python for', unrolled, (x,)), ('pl.loop(0, 8)', static_loop, (x,)), ('pl.loop(0, 8, unroll=4)', unrolled_loop, (x,)), ('pl.loop(0, n_smem[0])', dynamic_loop, (x, n))):
        compiled = tpuasm_tools.compile(function, *arguments, mesh=mesh)
        np.testing.assert_array_equal(np.asarray(compiled(*arguments)), np.asarray(x) * 2.0)
        listing = tpuasm_tools.kernel_listing(compiled)
        counts = tpuasm_tools.count_mnemonics(listing)
        bundles = sum(line.startswith('{') for line in listing.splitlines())
        print(f'## {name}：数值检查通过；kernel 段 {bundles} 个 bundle，vld {counts["vld.8x128"]} 条，vmul {counts["vmul.8x128.f32"]} 条')
        print(listing)
        print()

if __name__ == '__main__':
    main()
