"""第一章第 7 节的 o = x * s[0] + s[1]：用 tpuasm 去掉 SMEM 的 DMA 等待之后的 sfence，看 sld 读到的是哪一次运行的标量。"""
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

TRIALS = 20

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
            pltpu.async_copy(s_hbm, s_smem, sem).wait()
            x_vmem[...] = x_vmem[...] * s_smem[0] + s_smem[1]
            pltpu.async_copy(x_vmem, o_hbm, sem).wait()

        return kernel(x, s)

    x = jnp.ones((16, 128), jnp.float32)
    s = jnp.array([1.0, 0.0], jnp.float32)
    compiled = tpuasm_tools.compile(affine, x, s, mesh=mesh)
    serialized = tpuasm_tools.serialize(compiled)
    fences = tpuasm_tools.find_bundles(serialized, 'sfence')
    print(f'kernel 中的 sfence：bundle {fences}')
    for pc in range(fences[0] - 3, fences[0] + 2):
        print(f'  {pc}: {" ".join(tpuasm_tools.bundle_text(serialized, pc).split())}')
    patched = tpuasm_tools.edit_bundles(serialized, {fences[0]: ('s0: sfence', 'misc: vnop')})
    for name, function in (('原程序', compiled), ('去掉 sfence', tpuasm_tools.load(patched, compiled))):
        current, stale = 0, 0
        previous = None
        for trial in range(TRIALS):
            scalars = np.array([trial + 2.0, 100.0 * trial], np.float32)
            output = np.asarray(function(x, jnp.asarray(scalars)))[0, 0]
            current += output == scalars[0] + scalars[1]
            stale += previous is not None and output == previous[0] + previous[1]
            previous = scalars
        print(f'{name}：{TRIALS} 次运行中，用本次标量 {current} 次，用上一次运行的标量 {stale} 次')

if __name__ == '__main__':
    main()
