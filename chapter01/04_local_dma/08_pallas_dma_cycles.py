"""一次 DMA 要多少周期：用 KernelClock 测 HBM → TC VMEM 与 TC VMEM → HBM 两个方向、几种大小，从发起到等待结束；再比较两个输入的 DMA 先后进行与同时进行。每个区间运行 8 次。"""
import tpu_init
tpu_init.initialize_one_chip()

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P
import numpy as np

import tpuasm_tools

MESH = jax.make_mesh((1,), ('device',))
TC = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)
PARAMS = pltpu.CompilerParams(
    disable_bounds_checks=True,
    disable_semaphore_checks=True,
)

@jax.shard_map(
    mesh=MESH,
    in_specs=P(),
    out_specs=P(),
    check_vma=False,
)
def copy(x: jax.Array) -> jax.Array:
    @pl.kernel(
        out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
        mesh=TC,
        scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.SemaphoreType.DMA),
        name='copy',
        compiler_params=PARAMS,
    )
    def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, sem: Ref) -> None:
        pltpu.async_copy(x_hbm, x_vmem, sem).wait()
        pltpu.async_copy(x_vmem, o_hbm, sem).wait()

    return kernel(x)

def add(together: bool):
    @jax.shard_map(
        mesh=MESH,
        in_specs=(P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    def function(x: jax.Array, y: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
            mesh=TC,
            scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.VMEM(y.shape, y.dtype), pltpu.SemaphoreType.DMA((2,))),
            name='add',
            compiler_params=PARAMS,
        )
        def kernel(x_hbm: Ref, y_hbm: Ref, o_hbm: Ref, x_vmem: Ref, y_vmem: Ref, sems: Ref) -> None:
            if together:
                # 两个 DMA 都发起之后再等待。
                x_copy = pltpu.async_copy(x_hbm, x_vmem, sems.at[0])
                y_copy = pltpu.async_copy(y_hbm, y_vmem, sems.at[1])
                x_copy.wait()
                y_copy.wait()
            else:
                # 等到第一个 DMA 完成，才发起第二个。
                pltpu.async_copy(x_hbm, x_vmem, sems.at[0]).wait()
                pltpu.async_copy(y_hbm, y_vmem, sems.at[1]).wait()
            x_vmem[...] = x_vmem[...] + y_vmem[...]
            pltpu.async_copy(x_vmem, o_hbm, sems.at[0]).wait()

        return kernel(x, y)

    return function

def report(name: str, cycles: list[int]) -> None:
    print(f'  {name}：中位数 {int(np.median(cycles))}，范围 {min(cycles)}–{max(cycles)}')

def main() -> None:
    clock = tpuasm_tools.KernelClock(num_cores=2)
    rng = np.random.default_rng(0)
    print('## 一次 DMA：从发起到等待结束（dma.simple、vwait.ge、vsyncadd 三个 bundle）')
    for rows in (8, 64, 512, 2048, 8192):
        x = rng.standard_normal((rows, 128), dtype=np.float32)
        compiled = tpuasm_tools.compile(copy, x, mesh=MESH)
        first, second = tpuasm_tools.find_bundles(tpuasm_tools.serialize(compiled), 'dma.simple')

        def call(timed: jax.stages.Compiled) -> None:
            np.testing.assert_array_equal(np.asarray(timed(x)), x)

        print(f'f32[{rows},128]，{rows // 2} KiB')
        report('HBM → TC VMEM', clock.interval(compiled, first, second, call, samples=8))
        report('TC VMEM → HBM', clock.interval(compiled, second, second + 3, call, samples=8))

    print('## 两个输入的 DMA：从发起第一个到两个都等到')
    for rows in (64, 2048):
        x, y = (rng.standard_normal((rows, 128), dtype=np.float32) for _ in range(2))
        print(f'两个 f32[{rows},128]，各 {rows // 2} KiB')
        for name, together in (('先后进行', False), ('同时进行', True)):
            compiled = tpuasm_tools.compile(add(together), x, y, mesh=MESH)
            serialized = tpuasm_tools.serialize(compiled)
            first = tpuasm_tools.find_bundles(serialized, 'dma.simple')[0]
            # 两个输入都等到之后的第一个 bundle：第一条 vld 所在的 bundle。
            last = tpuasm_tools.find_bundles(serialized, 'vld.8x128')[0]

            def call(timed: jax.stages.Compiled) -> None:
                np.testing.assert_array_equal(np.asarray(timed(x, y)), x + y)

            report(f'{name}，bundle [{first}, {last})', clock.interval(compiled, first, last, call, samples=8))

if __name__ == '__main__':
    main()
