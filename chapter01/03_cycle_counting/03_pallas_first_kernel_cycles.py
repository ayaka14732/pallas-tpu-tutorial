"""第 1 节的最小 kernel 要多少周期：用 KernelClock 在已编译的程序中插入 LCC 读数，分别测整个 kernel、输入 DMA、计算和输出 DMA。每个区间只在两端各读一次，各运行 8 次。"""
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
            x_vmem[...] = x_vmem[...] * 2.0
            pltpu.async_copy(x_vmem, o_hbm, sem).wait()

        return kernel(x)

    x = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
    compiled = tpuasm_tools.compile(scale, x, mesh=mesh)
    serialized = tpuasm_tools.serialize(compiled)
    (_, start, end), = [op for op in tpuasm_tools.hlo_ops(compiled) if op[0].startswith('scale')]
    # kernel 主体的 8 个 bundle 从第一条 dma.simple 开始：输入 DMA 与等待 2 个，vld、vmul、vst 3 个，输出 DMA、等待与 vsyncadd 3 个。
    first_dma, second_dma = tpuasm_tools.find_bundles(serialized, 'dma.simple')
    print('## kernel 主体')
    for pc in range(first_dma, second_dma + 3):
        print(f'  {pc}: ' + ' '.join(tpuasm_tools.bundle_text(serialized, pc).split()))
    clock = tpuasm_tools.KernelClock(num_cores=2)

    def call(timed: jax.stages.Compiled) -> None:
        np.testing.assert_array_equal(np.asarray(timed(x)), np.asarray(x) * 2.0)

    print('## 周期数（区间 [first, last)，8 次运行）')
    intervals = {
        '整个 kernel（两个 vtrace 标记之间）': (start, end + 1),
        '输入 DMA：发起到等待结束': (first_dma, first_dma + 2),
        '计算：vld、vmul、vst': (first_dma + 2, second_dma),
        '输出 DMA：发起到等待结束': (second_dma, second_dma + 3),
    }
    for name, (first, last) in intervals.items():
        cycles = clock.interval(compiled, first, last, call, samples=8)
        print(f'  {name}，bundle [{first}, {last})：{cycles}，中位数 {int(np.median(cycles))}')

if __name__ == '__main__':
    main()
