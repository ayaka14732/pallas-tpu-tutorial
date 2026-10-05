"""一颗芯片的两个 TensorCore 各算一半：f32[64,128] 乘以 2，TensorCore c 处理第 32c 到 32c+31 行；与只用一个 TensorCore 的版本对照。"""
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

def build(num_cores: int):
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=num_cores)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def scale(x: jax.Array) -> jax.Array:
        rows = x.shape[0] // num_cores

        @pl.kernel(
            out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM((rows, x.shape[1]), x.dtype), pltpu.SemaphoreType.DMA),
            name='scale',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, sem: Ref) -> None:
            # 每个 TensorCore 读到自己在 TensorCoreMesh 中的编号，据此选出自己的行。
            core = jax.lax.axis_index('tc')
            window = pl.ds(core * rows, rows)
            pltpu.async_copy(x_hbm.at[window], x_vmem, sem).wait()
            x_vmem[...] = x_vmem[...] * 2.0
            pltpu.async_copy(x_vmem, o_hbm.at[window], sem).wait()

        return kernel(x)

    return mesh, scale

def main() -> None:
    x = jnp.arange(64 * 128, dtype=jnp.float32).reshape(64, 128)
    for num_cores in (1, 2):
        mesh, scale = build(num_cores)
        compiled = tpuasm_tools.compile(scale, x, mesh=mesh)
        np.testing.assert_array_equal(np.asarray(compiled(x)), np.asarray(x) * 2.0)
        listing = tpuasm_tools.kernel_listing(compiled, pallas_only=True)
        counts = tpuasm_tools.count_mnemonics(listing)
        bundles = sum(line.startswith('{') for line in listing.splitlines())
        print(f'## num_cores={num_cores}：数值检查通过；kernel 段 {bundles} 个 bundle；vld {counts["vld.8x128"]}、vmul {counts["vmul.8x128.f32"]}、vst {counts["vst.8x128"]}；vsyncadd.remote {counts["vsyncadd.remote.s32"]}')
        print(tpuasm_tools.listing_outline(compiled))
        print(listing)
        print()

if __name__ == '__main__':
    main()
