"""只改 dtype：f32、bf16、int8 的 [64,128] 经 TC VMEM 复制，比较 DMA 的 length。"""
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

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def copy(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.SemaphoreType.DMA),
            name='copy',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            pltpu.async_copy(x_vmem, o_hbm, sem).wait()

        return kernel(x)

    for dtype in (jnp.float32, jnp.bfloat16, jnp.int8):
        x = (jnp.arange(64 * 128) % 100).astype(dtype).reshape(64, 128)
        compiled = tpuasm_tools.compile(copy, x, mesh=mesh)
        np.testing.assert_array_equal(np.asarray(compiled(x)), np.asarray(x))
        listing = tpuasm_tools.kernel_listing(compiled)
        payload = x.size * x.dtype.itemsize
        print(f'## {x.dtype}[64,128]：{payload} B，数值检查通过')
        print('\n'.join(line for line in listing.splitlines() if 'dma.' in line))
        print()

if __name__ == '__main__':
    main()
