"""沿 sublane 的前缀和逐行串行计算：每次把一行广播到 8 个 sublane 再累加。比较 jnp.broadcast_to 与 stride=0 的 Ref 读取。"""
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

def main() -> None:
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)
    x = np.random.default_rng(0).integers(-100, 100, (8, 128)).astype(np.float32)
    for style in ('broadcast_to', 'stride=0'):

        @jax.shard_map(
            mesh=mesh,
            in_specs=P(),
            out_specs=P(),
            check_vma=False,
        )
        def scan(x: jax.Array) -> jax.Array:
            @pl.kernel(
                out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
                mesh=tc_mesh,
                scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.VMEM(x.shape, x.dtype), pltpu.SemaphoreType.DMA),
                name='scan',
                compiler_params=pltpu.CompilerParams(
                    disable_bounds_checks=True,
                    disable_semaphore_checks=True,
                ),
            )
            def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, o_vmem: Ref, sem: Ref) -> None:
                pltpu.async_copy(x_hbm, x_vmem, sem).wait()
                total = jnp.zeros(x_vmem.shape, x_vmem.dtype)
                for k in range(8):
                    if style == 'broadcast_to':
                        row = jnp.broadcast_to(x_vmem[pl.ds(k, 1), :], x_vmem.shape)
                    else:
                        # 从第 k 行起读 8 行、行距为 0：8 个 sublane 都是第 k 行。
                        row = x_vmem[pl.ds(k, 8, stride=0), :]
                    total = total + row
                    o_vmem[pl.ds(k, 1), :] = total[0:1, :]
                pltpu.async_copy(o_vmem, o_hbm, sem).wait()

            return kernel(x)

        compiled = tpuasm_tools.compile(scan, jnp.asarray(x), mesh=mesh)
        np.testing.assert_array_equal(np.asarray(compiled(jnp.asarray(x))), np.cumsum(x, axis=0))
        listing = tpuasm_tools.kernel_listing(compiled, pallas_only=True)
        counts = tpuasm_tools.count_mnemonics(listing)
        print(f'## {style}：数值检查通过；vld {counts["vld.8x128"]} 条，vadd {counts["vadd.8x128.f32"]} 条，vst {counts["vst.8x128"]} 条')
        print('\n'.join(line.split('#')[0].rstrip() for line in listing.splitlines() if any(key in line for key in ('vld:', 'vst:', 'va1:'))))
        print()

if __name__ == '__main__':
    main()
