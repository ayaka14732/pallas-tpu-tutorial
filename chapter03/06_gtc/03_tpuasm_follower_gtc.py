"""时间源与跟随者读到的 GTC：在 TPU v4 (2x2x2) 的 8 颗芯片上各自执行同样的片段。先连续 4 个周期读 GTC，统计相邻读数之差与低 4 位；再用 vdelay 停 10^9 个周期，比较 GTC 的增量与本地周期数。运行方式：podrun -- /srv/workspace/venv/bin/python 本文件。"""
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import numpy as np

REPEATS = 4000
DELAY = 10**9
REGULAR = {(1, 15, 16), (15, 16, 1), (16, 1, 15)}

def main() -> None:
    jax.distributed.initialize()
    import tpuasm_tools
    from tpuasm_tools import bundle, read_gtc, read_lcc

    lines = []
    for index, device in enumerate(jax.local_devices()):
        probe = tpuasm_tools.LccProbe(device=index)
        # 连续 4 个 bundle 读 GTC：相邻两次读数相隔 1 个本地周期。
        dense = probe.run_raw(read_gtc(20) + read_gtc(21) + read_gtc(22) + read_gtc(23), 4, repeats=REPEATS)[:, 0].astype(np.int64)
        patterns = Counter(tuple(np.diff(row).tolist()) for row in dense)
        regular = sum(count for pattern, count in patterns.items() if pattern in REGULAR)
        others = sorted(((pattern, count) for pattern, count in patterns.items() if pattern not in REGULAR), key=lambda item: -item[1])
        low = Counter((dense & 15).ravel().tolist())
        # L0、G0、vdelay、sfence、G1、L1：GTC 两个读数之间的本地周期数是 ΔL − 2。
        setup = bundle(f's0: simm.s32 s24, {DELAY}')
        body = read_lcc(20) + read_gtc(21) + bundle('misc: vdelay s24') + bundle('s0: sfence') + read_gtc(22) + read_lcc(23)
        counters = probe.run_raw(body, 4, repeats=4, setup=setup)[:, 0].astype(np.int64)
        cycles = counters[:, 3] - counters[:, 0] - 2
        excess = 3 * (counters[:, 2] - counters[:, 1]) - 32 * cycles
        lines.append(
            f'device {device.id}，坐标 {tuple(device.coords)}，进程 {device.process_index}\n'
            f'  连续 4 次读 GTC，{REPEATS} 次运行：1、15、16 的三种轮换 {regular} 次，其他 {REPEATS - regular} 次 {others[:6]}\n'
            f'  低 4 位的取值与次数：{sorted(low.items())}\n'
            f'  停 {DELAY} 个周期，4 次运行：本地周期数 {sorted(set(cycles.tolist()))}，3ΔG − 32ΔL = {excess.tolist()}，ΔG 相对 32/3 × ΔL 偏离 {np.median(excess / (32 * cycles)) * 1e6:+.3f} ppm'
        )
    print('\n'.join(lines), flush=True)

if __name__ == '__main__':
    main()
