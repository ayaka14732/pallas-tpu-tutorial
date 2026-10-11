"""读数本身怎样改变被测的程序：先测两次 KernelClock 读数之间隔 n 个 bundle 时的差值，再在一个 kernel 的计算段中间加入 k 次读数，把各段的周期数相加，与只在两端读数的结果比较。"""
import tpu_init
tpu_init.initialize_one_chip()

from collections.abc import Callable

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

import tpuasm_tools
from tpuasm_tools import bundle, clock_read

MESH = jax.make_mesh((1,), ('device',))
ROWS = 128
OPERATIONS = {
    '2x + 1': lambda x: x * 2.0 + 1.0,
    'exp(x)': jnp.exp,
    'x 加上各行的和': lambda x: x + jnp.sum(x, axis=1, keepdims=True),
}
FILLERS = {
    '空 bundle': bundle(),
    'vnop': bundle('misc: vnop'),
    'vadd': bundle('va0: vadd.8x128.s32 v31, 0, v31'),
}

def build(operation: Callable[[jax.Array], jax.Array]) -> Callable[[jax.Array], jax.Array]:
    """f32[128,128]：搬进 TC VMEM，计算，搬回 HBM。"""
    @jax.shard_map(
        mesh=MESH,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def function(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct((ROWS, 128), jnp.float32),
            mesh=pltpu.TensorCoreMesh(axis_name='tc', num_cores=1),
            scratch_types=(pltpu.VMEM((ROWS, 128), jnp.float32), pltpu.SemaphoreType.DMA),
            name='density',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            x_vmem[...] = operation(x_vmem[...])
            pltpu.async_copy(x_vmem, o_hbm, sem).wait()

        return kernel(x)

    return function

def readings(clock: tpuasm_tools.KernelClock, timed: jax.stages.Compiled, x: jax.Array, count: int) -> list[int]:
    """运行 4 次，确认各次的相邻读数之差相同，返回这些差值。"""
    runs = set()
    for _ in range(4):
        timed(x).block_until_ready()
        row = clock.read(count)[0].astype(np.int64)
        runs.add(tuple((row[1:] - row[:-1]).tolist()))
    differences, = runs
    return list(differences)

def main() -> None:
    clock = tpuasm_tools.KernelClock(num_cores=2)
    x = jnp.asarray(np.random.default_rng(0).standard_normal((ROWS, 128), dtype=np.float32))
    programs = {}
    for name, operation in OPERATIONS.items():
        compiled = tpuasm_tools.compile(build(operation), x, mesh=MESH)
        serialized = tpuasm_tools.serialize(compiled)
        # 计算段：输入 DMA 的 vwait、vsyncadd 之后，到发起输出 DMA 的 bundle 之前。
        first = tpuasm_tools.find_bundles(serialized, 'vwait')[0] + 2
        last = tpuasm_tools.find_bundles(serialized, 'dma.')[1]
        programs[name] = (compiled, serialized, first, last)

    print('## 两次读数之间隔 n 个 bundle，读数之差')
    compiled, serialized, first, _ = programs['2x + 1']
    for name, filler in FILLERS.items():
        row = []
        for count in (0, 1, 2, 3, 4, 8, 16):
            timed = tpuasm_tools.load(tpuasm_tools.insert_bundles(serialized, {first: clock_read(0) + filler * count + clock_read(1)}), compiled)
            difference, = readings(clock, timed, x, 2)
            row.append(f'n = {count}：{difference}')
        print(f'  {name}：' + '，'.join(row))

    print('## 计算段两端各读一次，中间再均匀地加入 k 次读数；每段的读数差扣除 20')
    for name, (compiled, _, first, last) in programs.items():
        print(f'  {name}：计算段 {last - first} 个 bundle')
        for extra in (0, 1, 2, 4, 8, 16):
            points = [first, *(first + (index + 1) * (last - first) // (extra + 1) for index in range(extra)), last]
            segments = [difference - tpuasm_tools.CLOCK_OVERHEAD for difference in readings(clock, clock.instrument(compiled, points), x, len(points))]
            print(f'    k = {extra:2d}：合计 {sum(segments):3d}，各段 {segments}')

if __name__ == '__main__':
    main()
