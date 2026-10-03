"""在 kernel 中连续生成随机数，比较硬件 vrng、各种分布、threefry2x32 与 Philox 的生成速度：同一方法编译生成 64 个与 576 个 TC VREG 的两个 kernel，取 XProf 中的时间之差除以 512；再与清单中每个 TC VREG 的计算指令数比较。"""
import tpu_init
tpu_init.initialise_one_chip()

from collections import Counter, defaultdict
import math
from pathlib import Path
import re
import statistics

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.random import philox
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

import tpuasm_tools
import xprof_tools

TILES = (64, 576)  # 两个 kernel 各生成多少个 TC VREG

def uniform(rows: int) -> jax.Array:
    return jax.lax.bitcast_convert_type((pltpu.prng_random_bits((rows, 128)).astype(jnp.uint32) >> 9) | jnp.uint32(0x3F800000), jnp.float32) - 1.0

def as_bits(value: jax.Array) -> jax.Array:
    return jax.lax.bitcast_convert_type(value, jnp.uint32)

def box_muller(rows: int) -> jax.Array:
    u1, u2 = 1.0 - uniform(rows // 2), uniform(rows // 2)
    r = jnp.sqrt(-2.0 * jnp.log(u1))
    return as_bits(jnp.concatenate([r * jnp.cos(2.0 * math.pi * u2), r * jnp.sin(2.0 * math.pi * u2)], axis=0))

def threefry(rows: int, step: jax.Array) -> jax.Array:
    return jax.random.bits(jax.random.fold_in(jax.random.key(1, impl='threefry2x32'), step), (rows, 128), jnp.uint32)

def philox4x32(rows: int, step: jax.Array) -> jax.Array:
    # 计数器是这一次生成的 rows / 4 行在整个序列中的位置；一次 Philox 产生 4 个 u32。
    quarter = rows // 4
    row = jax.lax.broadcasted_iota(jnp.uint32, (quarter, 128), 0) + step.astype(jnp.uint32) * quarter
    counter = row * 128 + jax.lax.broadcasted_iota(jnp.uint32, (quarter, 128), 1)
    zero = jnp.zeros((quarter, 128), jnp.uint32)
    return jnp.concatenate(philox.philox_4x32(counter, zero, zero, zero, jnp.uint32(1), jnp.uint32(2)), axis=0)

# 名称 → (每次生成的行数, 生成函数(行数, 第几次))；每个函数返回 u32[行数,128]。
METHODS = {
    '硬件 vrng': (64, lambda rows, step: pltpu.prng_random_bits((rows, 128)).astype(jnp.uint32)),
    'vrng → 均匀分布（stateful_uniform）': (64, lambda rows, step: as_bits(pltpu.stateful_uniform((rows, 128), jnp.float32))),
    'vrng → 伯努利（整数阈值）': (64, lambda rows, step: ((pltpu.prng_random_bits((rows, 128)).astype(jnp.uint32) >> 8).astype(jnp.int32) < (1 << 22)).astype(jnp.uint32)),
    'vrng → 指数分布': (64, lambda rows, step: as_bits(-jnp.log(1.0 - uniform(rows)))),
    'vrng → Gumbel 分布': (64, lambda rows, step: as_bits(-jnp.log(-jnp.log(1.0 - uniform(rows))))),
    'vrng → 正态分布（stateful_normal）': (64, lambda rows, step: as_bits(pltpu.stateful_normal((rows, 128), jnp.float32))),
    'vrng → 正态分布（Box–Muller）': (64, lambda rows, step: box_muller(rows)),
    'threefry2x32，每次 64 行': (64, threefry),
    'threefry2x32，每次 16 行': (16, threefry),
    'philox4x32，每次 64 行': (64, philox4x32),
}

def build(rows: int, generate, steps: int):
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
            out_type=jax.ShapeDtypeStruct((rows, 128), jnp.uint32),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM((rows, 128), jnp.uint32), pltpu.SemaphoreType.DMA),
            name='draw',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(o_hbm: Ref, acc_vmem: Ref, sem: Ref) -> None:
            pltpu.prng_seed(1)
            acc_vmem[...] = jnp.zeros((rows, 128), jnp.uint32)

            # 每次的结果异或进同一个 buffer，编译器不能删去任何一次生成。
            @pl.loop(0, steps)
            def _(step: jax.Array) -> None:
                acc_vmem[...] = acc_vmem[...] ^ generate(rows, step)

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

def compute_ops(compiled) -> int:
    """kernel 段中的计算指令。pl.loop 不展开，循环体在清单中只出现一次。"""
    counts = tpuasm_tools.count_mnemonics(tpuasm_tools.kernel_listing(compiled, pallas_only=True))
    return sum(count for name, count in counts.items() if name.startswith('v') and not name.startswith(('vld', 'vst', 'vwait', 'vsync', 'vtrace', 'vnop')))

def main() -> None:
    for name, (rows, generate) in METHODS.items():
        times = {}
        for tiles in TILES:
            mesh, draw = build(rows, generate, tiles * 8 // rows)
            compiled = tpuasm_tools.compile(draw, mesh=mesh)
            compiled().block_until_ready()
            times[tiles] = kernel_us(compiled)
        # 与循环体只生成全零的 kernel 相比，多出的计算指令就是每次生成的代价。
        mesh, empty = build(rows, lambda rows, step: jnp.zeros((rows, 128), jnp.uint32), 1)
        ops = (compute_ops(compiled) - compute_ops(tpuasm_tools.compile(empty, mesh=mesh))) / (rows // 8)
        cycles = (times[TILES[1]] - times[TILES[0]]) * 1.05e3 / (TILES[1] - TILES[0])
        if name == 'threefry2x32，每次 64 行':
            # 按槽统计：移位指令只能在 va1 发射。
            listing = tpuasm_tools.kernel_listing(compiled, pallas_only=True)
            slots = Counter(re.findall(r'(va[01]): (v[a-z]+)\.', listing))
            print('  threefry2x32 kernel 段按槽统计：' + '，'.join(f'{slot} {mnemonic} {count}' for (slot, mnemonic), count in slots.most_common(10)))
        print(f'{name}：{TILES[0]} 个 TC VREG {times[TILES[0]]:.2f} µs，{TILES[1]} 个 {times[TILES[1]]:.2f} µs；每个 TC VREG 实测 {cycles:.1f} 个周期，约 {4096 / cycles * 1.05:.0f} GB/s；清单中每个 TC VREG {ops:.1f} 条计算指令，两个向量 ALU 槽需要 {ops / 2:.1f} 个周期')

if __name__ == '__main__':
    main()
