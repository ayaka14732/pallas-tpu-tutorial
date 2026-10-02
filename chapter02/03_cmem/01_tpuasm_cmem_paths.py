"""Megacore Shared CMEM 的两条读取通路：用 tpuasm 改写一个只读 x 的载体 kernel，先把 y 从 HBM 搬进 CMEM，再分别经 DMA 搬进 TC VMEM（staging），或用 cld 直接读进 TC VREG。结果应为 2y 而非 2x。"""
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

# 把第二个输入 y（地址在 s1）搬到 CMEM 地址 0；本 kernel 独占 CMEM，从地址 0 开始使用。
Y_TO_CMEM = '''{ s0: simm.s32 s20, 0 }
{ s0: dma.simple [cmem:s20], [hbm:s1], length=8, dst_flag=[sflag:52] }
{ misc: vwait.ge [sflag:52], 8 }
{ misc: vsyncadd.s32 [sflag:52], -8 }'''

def main() -> None:
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=(P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    def carrier(x: jax.Array, y: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=pltpu.HBM(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.SemaphoreType.DMA),
            name='carrier',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, y_hbm: Ref, o_hbm: Ref, x_vmem: Ref, sem: Ref) -> None:
            # 载体只读 x；y 只是一个输入，等待改写后的程序使用。
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            x_vmem[...] = x_vmem[...] * 2.0
            pltpu.async_copy(x_vmem, o_hbm, sem).wait()

        # 两个输入都固定在 HBM，CMEM 不被 XLA 占用。
        return kernel(pltpu.with_memory_space_constraint(x, pltpu.HBM), pltpu.with_memory_space_constraint(y, pltpu.HBM))

    x = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
    y = -x - 1000
    compiled = tpuasm_tools.compile(carrier, x, y, mesh=mesh)
    np.testing.assert_array_equal(np.asarray(compiled(x, y)), np.asarray(x) * 2.0)
    serialized = tpuasm_tools.serialize(compiled)
    print('## 载体：结果为 2x')
    print('\n'.join(line.split('#')[0].rstrip() for line in tpuasm_tools.kernel_listing(serialized, pallas_only=True).splitlines() if 'dma.simple' in line or 'vld:' in line or 'vmul' in line))

    dma_pc, = tpuasm_tools.find_bundles(serialized, 'dma.simple [vmem:s10], [hbm:s0]')
    vld_pc, = tpuasm_tools.find_bundles(serialized, 'vld: vld.8x128 v0, [vmem:0x0]')
    mul_pc, = tpuasm_tools.find_bundles(serialized, 'vmul.8x128.f32 v1, 2.0, v0')

    # 一：staging。原来的输入 DMA 改为从 CMEM 读。
    staging = tpuasm_tools.edit_bundles(serialized, {dma_pc: ('[vmem:s10], [hbm:s0]', '[vmem:s10], [cmem:s20]')})
    staging = tpuasm_tools.insert_bundles(staging, {dma_pc: Y_TO_CMEM})
    result = np.asarray(tpuasm_tools.load(staging, compiled)(x, y))
    np.testing.assert_array_equal(result, np.asarray(y) * 2.0)
    print('\n## 一：HBM → CMEM → TC VMEM（staging）：结果为 2y，数值检查通过')
    print(Y_TO_CMEM)
    print('{ s0: dma.simple [vmem:s10], [cmem:s20], length=8, dst_flag=[sflag:52] }   # 原来的输入 DMA，源改为 CMEM')

    # 二：cld。保留 x 的输入 DMA，把 vld 换成从 CMEM 读的 cld，并在乘法之前插入从 crf 取回的 vpop。
    direct = tpuasm_tools.edit_bundles(serialized, {vld_pc: ('vld: vld.8x128 v0, [vmem:0x0]', 'cld: cld.8x128 crf, [cmem:0x0]')})
    direct = tpuasm_tools.insert_bundles(direct, {dma_pc: Y_TO_CMEM, mul_pc: '{ vr0: vpop.8x128 v0, crf }'})
    result = np.asarray(tpuasm_tools.load(direct, compiled)(x, y))
    np.testing.assert_array_equal(result, np.asarray(y) * 2.0)
    print('\n## 二：HBM → CMEM → TC VREG（cld）：结果为 2y，数值检查通过')
    print('{ cld: cld.8x128 crf, [cmem:0x0] }   # 原来的 vld，改为从 CMEM 地址 0 读入队列 crf')
    print('{ vr0: vpop.8x128 v0, crf }          # 新插入：从 crf 取回到 v0')
    print('{ va0: vmul.8x128.f32 v1, 2.0, v0 }  # 原来的乘法')

if __name__ == '__main__':
    main()
