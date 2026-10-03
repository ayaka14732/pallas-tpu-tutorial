"""在 kernel 中连续生成 T 个 u32[8,128]（每次 8 个），比较硬件 vrng、threefry2x32 与两种分布的生成速度：取 T = 576 与 T = 64 两个 kernel 在 XProf 中的时间之差，除以多生成的 512 个 TC VREG。"""
import tpu_init
tpu_init.initialise_one_chip()

from collections import defaultdict
from pathlib import Path
import statistics

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

import tpuasm_tools
import xprof_tools

CHUNK = 64  # 每次生成的行数：8 个 TC VREG

def generate(method: str, chunk: jax.Array) -> jax.Array:
    if method == '硬件 vrng':
        return pltpu.prng_random_bits((CHUNK, 128)).astype(jnp.uint32)
    if method == 'threefry2x32':
        key = jax.random.fold_in(jax.random.key(1, impl='threefry2x32'), chunk)
        return jax.random.bits(key, (CHUNK, 128), jnp.uint32)
    if method == '硬件 vrng → 均匀分布':
        return jax.lax.bitcast_convert_type(pltpu.stateful_uniform((CHUNK, 128), jnp.float32), jnp.uint32)
    return jax.lax.bitcast_convert_type(pltpu.stateful_normal((CHUNK, 128), jnp.float32), jnp.uint32)

def build(method: str, chunks: int):
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=(),
        out_specs=P(),
        check_vma=False,
    )
    def draw() -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct((CHUNK, 128), jnp.uint32),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM((CHUNK, 128), jnp.uint32), pltpu.SemaphoreType.DMA),
            name='draw',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(o_hbm: Ref, acc_vmem: Ref, sem: Ref) -> None:
            pltpu.prng_seed(1)
            acc_vmem[...] = jnp.zeros((CHUNK, 128), jnp.uint32)

            # 每次的结果异或进同一个 buffer，编译器不能删去任何一次生成。
            @pl.loop(0, chunks)
            def _(chunk: jax.Array) -> None:
                acc_vmem[...] = acc_vmem[...] ^ generate(method, chunk)

            pltpu.async_copy(acc_vmem, o_hbm, sem).wait()

        return kernel()

    return mesh, draw

def kernel_us(compiled) -> float:
    events = xprof_tools.device_events(xprof_tools.capture(lambda: [compiled().block_until_ready() for _ in range(16)], Path('/tmp/pallas_tpu_tutorial/xprof')))
    durations = defaultdict(list)
    for event in events:
        if event['device'] == '/device:TPU:0' and event['track'] == 'XLA Ops':
            durations[event['name']].append(xprof_tools.duration_us(event))
    (values,) = durations.values()
    return statistics.median(values)

def main() -> None:
    for method in ('硬件 vrng', '硬件 vrng → 均匀分布', '硬件 vrng → 正态分布', 'threefry2x32'):
        times = {}
        for chunks in (8, 72):
            mesh, draw = build(method, chunks)
            compiled = tpuasm_tools.compile(draw, mesh=mesh)
            compiled().block_until_ready()
            times[chunks] = kernel_us(compiled)
        tiles = (72 - 8) * CHUNK // 8
        cycles = (times[72] - times[8]) * 1.05e3 / tiles
        print(f'{method}：T = 64 时 {times[8]:.2f} µs，T = 576 时 {times[72]:.2f} µs；每个 TC VREG 约 {cycles:.1f} 个周期，约 {4096 / cycles * 1.05:.0f} GB/s')

if __name__ == '__main__':
    main()
