"""逐 tile 计算 y = 2x + 1：f32[32768,128] 切成 f32[512,128] 的 tile。比较串行、输入双缓冲、输入输出都双缓冲三种调度的数值、清单与每遍时间；再只改 tile 大小。"""
import tpu_init
tpu_init.initialise_one_chip()

import statistics
import time

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

import tpuasm_tools

ROWS = 32768

def build(style: str, repeats: int = 1, tile_rows: int = 512):
    tiles = ROWS // tile_rows
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def transform(x: jax.Array) -> jax.Array:
        @pl.kernel(
            # 输入输出都固定在 HBM：否则 XLA 可能把它们放进 Megacore Shared CMEM（第 3 节），测到的就不是 HBM 的数据通路。
            out_type=pltpu.HBM(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(
                pltpu.VMEM((2, tile_rows, 128), x.dtype),
                pltpu.VMEM((2, tile_rows, 128), x.dtype),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
            ),
            name='transform',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, o_vmem: Ref, x_sems: Ref, o_sems: Ref) -> None:
            def load(tile: jax.Array, slot: int | jax.Array):
                return pltpu.make_async_copy(x_hbm.at[pl.ds(tile * tile_rows, tile_rows)], x_vmem.at[slot], x_sems.at[slot])

            def store(tile: jax.Array, slot: int | jax.Array):
                return pltpu.make_async_copy(o_vmem.at[slot], o_hbm.at[pl.ds(tile * tile_rows, tile_rows)], o_sems.at[slot])

            # 整个流水线重复 repeats 遍，用于计时：每遍都从 HBM 读入全部数据、写回全部结果。
            @pl.loop(0, repeats)
            def _(_: jax.Array) -> None:
                if style == '串行':
                    @pl.loop(0, tiles)
                    def _(tile: jax.Array) -> None:
                        load(tile, 0).start()
                        load(tile, 0).wait()
                        o_vmem[0] = x_vmem[0] * 2.0 + 1.0
                        store(tile, 0).start()
                        store(tile, 0).wait()

                    return

                # prologue：先发出第 0 个 tile 的输入 DMA。
                load(0, 0).start()

                @pl.loop(0, tiles)
                def _(tile: jax.Array) -> None:
                    slot = tile % 2
                    # 等当前 tile 到达，然后立即发出下一个 tile 的输入 DMA，写进另一个 buffer。
                    load(tile, slot).wait()

                    @pl.when(tile + 1 < tiles)
                    def _() -> None:
                        load(tile + 1, 1 - slot).start()

                    if style == '输入双缓冲':
                        o_vmem[0] = x_vmem[slot] * 2.0 + 1.0
                        store(tile, 0).start()
                        store(tile, 0).wait()
                    else:
                        # 输出 buffer slot 上一次用于 tile - 2；写之前先等那次输出 DMA 完成。
                        @pl.when(tile >= 2)
                        def _() -> None:
                            store(tile - 2, slot).wait()

                        o_vmem[slot] = x_vmem[slot] * 2.0 + 1.0
                        store(tile, slot).start()

                if style == '输入输出双缓冲':
                    # epilogue：排空最后两个 tile 的输出 DMA。
                    store(tiles - 2, tiles % 2).wait()
                    store(tiles - 1, (tiles - 1) % 2).wait()

        return kernel(x)

    return mesh, transform

def pass_time(style: str, x: jax.Array, tile_rows: int = 512) -> float:
    """每遍流水线的时间（微秒）：kernel 内部重复 8 遍与 4 遍的主机计时之差除以 4，抵消调用 kernel 的固定开销。"""
    times = {}
    for repeats in (4, 8):
        mesh, transform = build(style, repeats, tile_rows)
        compiled = tpuasm_tools.compile(transform, x, mesh=mesh)
        jax.block_until_ready(compiled(x))
        samples = []
        for _ in range(20):
            start = time.perf_counter()
            jax.block_until_ready(compiled(x))
            samples.append(time.perf_counter() - start)
        times[repeats] = statistics.median(samples)
    return (times[8] - times[4]) / 4 * 1e6

def main() -> None:
    x = jnp.arange(ROWS * 128, dtype=jnp.float32).reshape(ROWS, 128) / 1024
    expected = np.asarray(x) * 2.0 + 1.0
    for style in ('串行', '输入双缓冲', '输入输出双缓冲'):
        mesh, transform = build(style)
        compiled = tpuasm_tools.compile(transform, x, mesh=mesh)
        np.testing.assert_array_equal(np.asarray(compiled(x)), expected)
        listing = tpuasm_tools.kernel_listing(compiled, pallas_only=True)
        counts = tpuasm_tools.count_mnemonics(listing)
        dma = '、'.join(line.split('#')[0].strip() for line in listing.splitlines() if 'dma.' in line)
        print(f'## {style}：数值检查通过；每遍约 {pass_time(style, x):.1f} µs')
        print(f'  DMA 指令 {counts["dma.simple"] + counts["dma.strided"]} 条，vwait.ge {counts["vwait.ge"]} 条，vld {counts["vld.8x128"]}、vst {counts["vst.8x128"]}')
        print(listing)
        print()

    # 只改 tile 大小：输入输出双缓冲，每个 tile 的字节数从 256 KiB 增加到 1 MiB。
    for tile_rows in (512, 1024, 2048):
        print(f'## 输入输出双缓冲，tile 为 f32[{tile_rows},128]（{tile_rows * 128 * 4 // 1024} KiB）：每遍约 {pass_time("输入输出双缓冲", x, tile_rows):.1f} µs')

if __name__ == '__main__':
    main()
