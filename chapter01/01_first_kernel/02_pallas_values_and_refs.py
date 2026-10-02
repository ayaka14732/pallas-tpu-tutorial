"""值与 Ref：读一次 Ref 得到的值可以多次使用、连续运算而不写回；每次读写 Ref 才对应一次 load/store。"""
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

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def polynomial(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.VMEM(x.shape, x.dtype), pltpu.SemaphoreType.DMA),
            name='polynomial',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, o_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            x = x_vmem[...]
            # 中间值 y 只存在于 TC VREG 中，不需要为它分配 TC VMEM。
            y = x * x
            o_vmem[...] = y * x + y + x
            # 也可以只写 Ref 的一部分：第 0–7 行改写为 x 的第 8–15 行。
            o_vmem[0:8, :] = x_vmem[8:16, :]
            pltpu.async_copy(o_vmem, o_hbm, sem).wait()

        return kernel(x)

    x = jnp.arange(16 * 128, dtype=jnp.float32).reshape(16, 128) / 1024
    compiled = tpuasm_tools.compile(polynomial, x, mesh=mesh)
    host = np.asarray(x)
    expected = host * host * host + host * host + host
    expected[0:8] = host[8:16]
    np.testing.assert_allclose(np.asarray(compiled(x)), expected, rtol=1e-6)
    print('数值检查通过')
    listing = tpuasm_tools.kernel_listing(compiled)
    print('\n# 向量指令')
    print('\n'.join(line for line in listing.splitlines() if any(f'{slot}: ' in line for slot in ('vld', 'vst', 'va0', 'va1'))))

if __name__ == '__main__':
    main()
