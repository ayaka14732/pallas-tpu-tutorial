"""在 kernel 中连续生成随机数，比较硬件 vrng、各种分布、threefry2x32 与 Philox 的生成速度：同一方法编译生成 64 个与 576 个 TC VREG 的两个 kernel，取 XProf 中的时间之差除以 512；再与清单中每个 TC VREG 的计算指令数比较。"""
import tpu_init
tpu_init.initialise_one_chip()

from collections import Counter, defaultdict
from pathlib import Path
import re
import statistics

import jax.numpy as jnp

from generation_common import METHODS, build
import tpuasm_tools
import xprof_tools

TILES = (64, 576)  # 两个 kernel 各生成多少个 TC VREG

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
