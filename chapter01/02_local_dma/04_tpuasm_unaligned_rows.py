"""Mosaic 拒绝的两种行窗口，用 tpuasm 改写已编译的对齐版本后在真机上运行：非对齐行起点 [3:11, :] 与 9 行窗口 [8:17, :]。"""
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

def replace_once(source: str, old: str, new: str) -> str:
    assert source.count(old) >= 1, old
    return source.replace(old, new, 1)

def main() -> None:
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)
    x = jnp.arange(64 * 128, dtype=jnp.float32).reshape(64, 128)

    def window(rows: int) -> jax.stages.Compiled:
        """编译合法的对齐窗口 x[8:8+rows, :]，作为改写的载体。"""

        @jax.shard_map(
            mesh=mesh,
            in_specs=P(),
            out_specs=P(),
            check_vma=False,
        )
        def function(x: jax.Array) -> jax.Array:
            @pl.kernel(
                out_type=jax.ShapeDtypeStruct((rows, 128), x.dtype),
                mesh=tc_mesh,
                scratch_types=(pltpu.VMEM((rows, 128), x.dtype), pltpu.SemaphoreType.DMA),
                name='window',
                compiler_params=pltpu.CompilerParams(
                    disable_bounds_checks=True,
                    disable_semaphore_checks=True,
                ),
            )
            def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, sem: Ref) -> None:
                pltpu.async_copy(x_hbm.at[pl.ds(8, rows)], x_vmem, sem).wait()
                pltpu.async_copy(x_vmem, o_hbm, sem).wait()

            return kernel(x)

        return tpuasm_tools.compile(function, x, mesh=mesh)

    host = np.asarray(x)

    # 一、非对齐行起点：载体读 x[8:16]，HBM 源地址偏移 8 改为 3。
    carrier = window(8)
    listing = tpuasm_tools.kernel_listing(carrier)
    print('## 载体 x[8:16, :] 的地址与 DMA')
    print('\n'.join(line for line in listing.splitlines() if 'sadd.s32 s9' in line or 'dma.' in line))
    np.testing.assert_array_equal(np.asarray(carrier(x)), host[8:16])

    def unaligned(source: str) -> str:
        return replace_once(source, 'sadd.s32 s9, 8, s0', 'sadd.s32 s9, 3, s0')

    patched = tpuasm_tools.load(tpuasm_tools.replace_listing(tpuasm_tools.serialize(carrier), unaligned), carrier)
    np.testing.assert_array_equal(np.asarray(patched(x)), host[3:11])
    print('改写：sadd.s32 s9, 8, s0 → sadd.s32 s9, 3, s0')
    print('结果等于 x[3:11, :]，数值检查通过\n')

    # 二、9 行窗口：载体读 x[8:24]，输入 DMA 的 length 与等待计数从 16 改为 9。
    carrier = window(16)
    listing = tpuasm_tools.kernel_listing(carrier)
    print('## 载体 x[8:24, :] 的输入 DMA 与等待')
    print('\n'.join(line for line in listing.splitlines() if 'dma.' in line or 'vwait' in line or 'vsyncadd.s32 [sflag:52]' in line))
    np.testing.assert_array_equal(np.asarray(carrier(x)), host[8:24])

    def nine_rows(source: str) -> str:
        source = replace_once(source, 'dma.simple [vmem:s10], [hbm:s9], length=16', 'dma.simple [vmem:s10], [hbm:s9], length=9')
        source = replace_once(source, 'vwait.ge [sflag:52], 16', 'vwait.ge [sflag:52], 9')
        return replace_once(source, 'vsyncadd.s32 [sflag:52], -16', 'vsyncadd.s32 [sflag:52], -9')

    patched = tpuasm_tools.load(tpuasm_tools.replace_listing(tpuasm_tools.serialize(carrier), nine_rows), carrier)
    result = np.asarray(patched(x))
    np.testing.assert_array_equal(result[:9], host[8:17])
    print('改写：输入 DMA length=16 → 9，第一处 vwait.ge 16 → 9，第一处 vsyncadd -16 → -9')
    print('结果前 9 行等于 x[8:17, :]，数值检查通过')

if __name__ == '__main__':
    main()
