"""TensorCore 0 计算 2x 写进 HBM，TensorCore 1 读它再加 1。比较 TensorCore 1 等待 TensorCore 0 的信号与不等待两种写法。"""
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

def build(wait: bool):
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=2)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def pipeline(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=pltpu.HBM((2, 8, 128), x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM((8, 128), x.dtype), pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.REGULAR),
            name='pipeline',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, vmem: Ref, sem: Ref, ready: Ref) -> None:
            core = jax.lax.axis_index('tc')

            @pl.when(core == 0)
            def _() -> None:
                pltpu.async_copy(x_hbm, vmem, sem).wait()
                vmem[...] = vmem[...] * 2.0
                pltpu.async_copy(vmem, o_hbm.at[0], sem).wait()
                # 写进 HBM 的 DMA 已经等到，再通知 TensorCore 1。
                pl.semaphore_signal(ready, 1, device_id={'tc': 1})

            @pl.when(core == 1)
            def _() -> None:
                if wait:
                    pl.semaphore_wait(ready, 1)
                pltpu.async_copy(o_hbm.at[0], vmem, sem).wait()
                vmem[...] = vmem[...] + 1.0
                pltpu.async_copy(vmem, o_hbm.at[1], sem).wait()

        return kernel(x)

    return mesh, pipeline

def main() -> None:
    x = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
    for wait in (True, False):
        mesh, pipeline = build(wait)
        compiled = tpuasm_tools.compile(pipeline, x, mesh=mesh)
        wrong = 0
        for trial in range(200):
            value = x + trial
            result = np.asarray(compiled(value))
            np.testing.assert_array_equal(result[0], np.asarray(value) * 2.0)
            wrong += not np.array_equal(result[1], np.asarray(value) * 2.0 + 1.0)
        listing = tpuasm_tools.kernel_listing(compiled, pallas_only=True)
        name = 'TensorCore 1 等待信号' if wait else 'TensorCore 1 不等待'
        print(f'## {name}：TensorCore 0 的结果 200 次全部正确；TensorCore 1 的结果错误 {wrong} 次')
        print('\n'.join(line.split('#')[0].rstrip() for line in listing.splitlines() if 'sflag:s' in line and 'remote' in line or 'semaphore' in line and ('vwait' in line or 'vsyncadd' in line)))
        print()

if __name__ == '__main__':
    main()
