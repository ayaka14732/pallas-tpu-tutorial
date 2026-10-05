"""两条 DMA 同时进行时省下的是什么：一个 TensorCore 上，对 f32[R,128] 的 tile 分别做“只读入”“只写出”“先读入再写出”“读入与写出同时进行”“两条读入同时进行”，比较每个 tile 的时间。"""
import tpu_init
tpu_init.initialize_one_chip()

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

import tpuasm_tools

ROWS = 32768
PATTERNS = ('只读入', '只写出', '先读入再写出', '读入与写出同时进行', '两条读入同时进行')

def build(pattern: str, tile_rows: int, repeats: int):
    tiles = ROWS // tile_rows
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def copy(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=pltpu.HBM(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM((2, tile_rows, 128), x.dtype), pltpu.SemaphoreType.DMA((2,))),
            name='copy',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, vmem: Ref, sems: Ref) -> None:
            def load(tile: jax.Array, slot: int):
                return pltpu.make_async_copy(x_hbm.at[pl.ds(tile * tile_rows, tile_rows)], vmem.at[slot], sems.at[slot])

            def store(tile: jax.Array, slot: int):
                return pltpu.make_async_copy(vmem.at[slot], o_hbm.at[pl.ds(tile * tile_rows, tile_rows)], sems.at[slot])

            @pl.loop(0, repeats)
            def _(_: jax.Array) -> None:
                @pl.loop(0, tiles)
                def _(tile: jax.Array) -> None:
                    # 每个 tile 发起一条或两条 DMA；同时进行的两条用不同的 buffer 和信号量。
                    if pattern == '只读入':
                        copies = [[load(tile, 0)]]
                    elif pattern == '只写出':
                        copies = [[store(tile, 1)]]
                    elif pattern == '先读入再写出':
                        copies = [[load(tile, 0)], [store(tile, 1)]]
                    elif pattern == '读入与写出同时进行':
                        copies = [[load(tile, 0), store(tile, 1)]]
                    else:
                        copies = [[load(tile, 0), load((tile + 1) % tiles, 1)]]
                    for group in copies:
                        for dma in group:
                            dma.start()
                        for dma in group:
                            dma.wait()

        return kernel(x)

    return mesh, copy

def cycles_per_tile(clock: tpuasm_tools.KernelClock, pattern: str, tile_rows: int, x: jax.Array) -> float:
    """每个 tile 的周期数：kernel 内部重复 8 遍与 4 遍，各用 LCC 读出 kernel 的周期数，相减再除以 4 遍的 tile 数。"""
    cycles = {}
    for repeats in (4, 8):
        mesh, copy = build(pattern, tile_rows, repeats)
        compiled = tpuasm_tools.compile(copy, x, mesh=mesh)
        cycles[repeats] = clock.kernel_cycles(compiled, lambda timed: jax.block_until_ready(timed(x)))
    return (cycles[8] - cycles[4]) / (4 * ROWS // tile_rows)

def main() -> None:
    x = jnp.arange(ROWS * 128, dtype=jnp.float32).reshape(ROWS, 128)
    clock = tpuasm_tools.KernelClock(num_cores=1)
    for tile_rows in (512, 2048):
        print(f'## tile 为 f32[{tile_rows},128]（{tile_rows // 2} KiB）')
        for pattern in PATTERNS:
            print(f'  {pattern}：每个 tile {cycles_per_tile(clock, pattern, tile_rows, x):.0f} 个周期')

if __name__ == '__main__':
    main()
