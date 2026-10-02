"""Mosaic 拒绝的窗口 f32[64,256] 的 [3:11, :]：编译对齐的 [8:16, :] 作为载体，用 tpuasm 把一次连续 DMA 改成两次 strided DMA，在真机上运行。"""
import tpu_init
tpu_init.initialise_one_chip()

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np
from tpuasm import executable_programs, format_assembly, parse_assembly

import tpuasm_tools

ORIGINAL = 'dma.simple [vmem:s10], [hbm:s9], length=16, dst_flag=[sflag:52]'
# 第二段：第 8–10 行，在 HBM 中位于第 1 行 tile（偏移 16 个 granule）的前 3 行；两个列 tile 各一段，每段 3 个 granule。
SECOND = 'dma.strided [vmem:s14], [hbm:s9], length=6, dst_stride=s13, src_stride=s13, elements_per_stride=s15, dst_flag=[sflag:52]'
# 第一段：第 3–7 行，在第 0 行 tile 的后 5 行，源地址 = 载体的 16 − 13 = 3；两个列 tile 各一段，每段 5 个 granule。
FIRST = '''{ s0: sadd.s32 s11, -13, s9 ; s1: simm.s32 s12, 5 }
{ s0: simm.s32 s13, 8 ; s1: sadd.s32 s14, 5, s10 }
{ s1: simm.s32 s15, 3 }
{ s0: @p0 dma.strided [vmem:s10], [hbm:s11], length=10, dst_stride=s13, src_stride=s13, elements_per_stride=s12, dst_flag=[sflag:52] }'''

def main() -> None:
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)
    x = jnp.arange(64 * 256, dtype=jnp.float32).reshape(64, 256)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def window(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct((8, 256), x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM((8, 256), x.dtype), pltpu.SemaphoreType.DMA),
            name='window',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm.at[pl.ds(8, 8)], x_vmem, sem).wait()
            pltpu.async_copy(x_vmem, o_hbm, sem).wait()

        return kernel(x)

    carrier = tpuasm_tools.compile(window, x, mesh=mesh)
    host = np.asarray(x)
    np.testing.assert_array_equal(np.asarray(carrier(x)), host[8:16])
    print('## 载体 x[8:16, :]：数值检查通过')
    print('\n'.join(line.split('#')[0].rstrip() for line in tpuasm_tools.kernel_listing(carrier, pallas_only=True).splitlines() if 'sadd.s32 s9' in line or 'dma.' in line or 'vwait' in line))

    serialized = tpuasm_tools.serialize(carrier)
    (_, _, image), = executable_programs(serialized)
    program = parse_assembly(format_assembly(image, target=tpuasm_tools.TARGET))
    pc = next(pc for pc, bundle in enumerate(program.bundles) if any(instruction.mnemonic == 'dma.simple' and instruction.operands[1] == '[hbm:s9]' for instruction in bundle.instructions))

    def second_part(source: str) -> str:
        assert source.count(ORIGINAL) == 1
        return source.replace(ORIGINAL, SECOND)

    patched = tpuasm_tools.insert_bundles(tpuasm_tools.replace_listing(serialized, second_part), {pc: FIRST})
    result = np.asarray(tpuasm_tools.load(patched, carrier)(x))
    np.testing.assert_array_equal(result, host[3:11])
    print('\n## 改写后：原 DMA 之前插入 4 个 bundle，原 DMA 换成第二段')
    print(FIRST)
    print('{ s0: @p0 ' + SECOND + ' }')
    print('结果等于 x[3:11, :]，数值检查通过')

if __name__ == '__main__':
    main()
