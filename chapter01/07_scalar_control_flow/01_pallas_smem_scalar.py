"""标量从 HBM 经 DMA 进入 SMEM，由标量单元读出，再作为向量乘法的一个操作数：o = x * s[0] + s[1]。"""
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
        in_specs=(P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    def affine(x: jax.Array, s: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.SMEM(s.shape, s.dtype), pltpu.SemaphoreType.DMA),
            name='affine',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, s_hbm: Ref, o_hbm: Ref, x_vmem: Ref, s_smem: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            # 标量参数同样先由 DMA 搬运，只是目的地换成 SMEM。
            pltpu.async_copy(s_hbm, s_smem, sem).wait()
            # 读 SMEM Ref 的单个元素得到一个标量值。
            scale = s_smem[0]
            shift = s_smem[1]
            x_vmem[...] = x_vmem[...] * scale + shift
            pltpu.async_copy(x_vmem, o_hbm, sem).wait()

        return kernel(x, s)

    x = jnp.arange(16 * 128, dtype=jnp.float32).reshape(16, 128)
    s = jnp.array([3.0, -1.0], jnp.float32)
    compiled = tpuasm_tools.compile(affine, x, s, mesh=mesh)
    np.testing.assert_array_equal(np.asarray(compiled(x, s)), np.asarray(x) * 3.0 - 1.0)
    print('数值检查通过')
    listing = tpuasm_tools.kernel_listing(compiled)
    tpuasm_tools.print_mnemonic_counts(listing)
    print(listing)

if __name__ == '__main__':
    main()
