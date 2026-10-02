"""两个 TensorCore 共享 Megacore Shared CMEM：TensorCore 0 把整个 y 从 HBM 搬进 CMEM，两个 TensorCore 再各从 CMEM 读自己的一半。比较把这次搬运放在入口汇合之前与之后。"""
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

# kernel 入口处 s1 是 y 的地址，但编译器随后把 s1 另作他用；先把它保存到此处未使用的 s22。
SAVE = '{ s0: smov s22, s1 }'
# 只由 TensorCore 0 执行（s9 是 TensorCore 编号，p10 是此处未使用的谓词寄存器）：把整个 y 搬到 CMEM 地址 0。
PUBLISH = '''{ s0: seq.s32 p10, s9, 0 ; s1: simm.s32 s21, 0 }
{ s0: @p10 dma.simple [cmem:s21], [hbm:s22], length=64, dst_flag=[sflag:52] }
{ misc: @p10 vwait.ge [sflag:52], 64 }
{ misc: @p10 vsyncadd.s32 [sflag:52], -64 }'''

def main() -> None:
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=2)

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
            scratch_types=(pltpu.VMEM((32, 128), x.dtype), pltpu.SemaphoreType.DMA),
            name='carrier',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, y_hbm: Ref, o_hbm: Ref, x_vmem: Ref, sem: Ref) -> None:
            window = pl.ds(jax.lax.axis_index('tc') * 32, 32)
            pltpu.async_copy(x_hbm.at[window], x_vmem, sem).wait()
            x_vmem[...] = x_vmem[...] * 2.0
            pltpu.async_copy(x_vmem, o_hbm.at[window], sem).wait()

        return kernel(pltpu.with_memory_space_constraint(x, pltpu.HBM), pltpu.with_memory_space_constraint(y, pltpu.HBM))

    x = jnp.arange(64 * 128, dtype=jnp.float32).reshape(64, 128)
    compiled = tpuasm_tools.compile(carrier, x, x, mesh=mesh)
    serialized = tpuasm_tools.serialize(compiled)
    barrier_pc = tpuasm_tools.find_bundles(serialized, 'vsyncadd.remote.s32')[0]
    dma_pc, = tpuasm_tools.find_bundles(serialized, 'dma.simple [vmem:s24], [hbm:s23]')
    # 每个 TensorCore 的输入 DMA 改为从 CMEM 读：s1 此时是 TensorCore 编号 × 32，即它那一半在 CMEM 中的起点。
    read_cmem = tpuasm_tools.edit_bundles(serialized, {dma_pc: ('[vmem:s24], [hbm:s23]', '[vmem:s24], [cmem:s1]')})
    versions = {
        '搬运在入口汇合之前': tpuasm_tools.load(tpuasm_tools.insert_bundles(read_cmem, {barrier_pc: SAVE + '\n' + PUBLISH}), compiled),
        '搬运在入口汇合之后': tpuasm_tools.load(tpuasm_tools.insert_bundles(read_cmem, {barrier_pc: SAVE, dma_pc: PUBLISH}), compiled),
    }
    print(f'入口汇合从 bundle {barrier_pc} 开始，每个 TensorCore 的输入 DMA 在 bundle {dma_pc}')
    for name, function in versions.items():
        wrong = [0, 0]
        for trial in range(200):
            y = jnp.full((64, 128), float(trial), jnp.float32)
            result = np.asarray(function(x, y))
            for core in range(2):
                wrong[core] += not np.array_equal(result[core * 32:core * 32 + 32], np.asarray(y)[core * 32:core * 32 + 32] * 2.0)
        print(f'## {name}：200 次中结果错误的次数，TensorCore 0 的一半 {wrong[0]} 次，TensorCore 1 的一半 {wrong[1]} 次')

if __name__ == '__main__':
    main()
