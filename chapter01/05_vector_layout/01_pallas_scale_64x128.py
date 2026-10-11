"""只改 dtype：f32、bf16、int8 的 [64,128] 在 TC VMEM 中乘以 2（int8 为加 1），比较 load/store 条数与 TC VREG 内的 pack/unpack。"""
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

def main() -> None:
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)
    for dtype in (jnp.float32, jnp.bfloat16, jnp.int8):

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
                scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.SemaphoreType.DMA),
                name='scale',
                compiler_params=pltpu.CompilerParams(
                    disable_bounds_checks=True,
                    disable_semaphore_checks=True,
                ),
            )
            def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, sem: Ref) -> None:
                pltpu.async_copy(x_hbm, x_vmem, sem).wait()
                if x_vmem.dtype == jnp.int8:
                    x_vmem[...] = x_vmem[...] + jnp.int8(1)
                else:
                    x_vmem[...] = x_vmem[...] * 2
                pltpu.async_copy(x_vmem, o_hbm, sem).wait()

            return kernel(x)

        x = (jnp.arange(64 * 128) % 100).astype(dtype).reshape(64, 128)
        expected = np.asarray(x) + np.int8(1) if dtype == jnp.int8 else np.asarray(x) * 2
        try:
            compiled = tpuasm_tools.compile(scale, x, mesh=mesh)
        except Exception as error:
            print(f'## {x.dtype}[64,128]：编译失败')
            print(str(error).splitlines()[0])
            print()
            continue
        np.testing.assert_array_equal(np.asarray(compiled(x)), expected)
        listing = tpuasm_tools.kernel_listing(compiled)
        print(f'## {x.dtype}[64,128]：数值检查通过')
        tpuasm_tools.print_mnemonic_counts(listing)
        print(listing)
        print()

if __name__ == '__main__':
    main()
