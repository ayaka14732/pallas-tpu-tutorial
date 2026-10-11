"""LCC 的频率，以主机的 CLOCK_MONOTONIC 为参照：LCC 在两次运行之间也持续计数，把约 20 秒内的 200 次运行串成一条时间轴，整段 LCC 之差除以主机区间的上下界。"""
import tpu_init
tpu_init.initialize_one_chip()

import importlib.util
from pathlib import Path
import time

import jax.numpy as jnp
import numpy as np

import tpuasm_tools

spec = importlib.util.spec_from_file_location('five_times', Path(__file__).resolve().parents[1] / '01_five_times' / '01_pallas_host_and_device_time.py')
five_times = importlib.util.module_from_spec(spec)
spec.loader.exec_module(five_times)
CALLS = 200
GAP = 0.1  # 相邻两次运行之间主机休眠的秒数

def main() -> None:
    clock = tpuasm_tools.KernelClock(num_cores=2)
    x = jnp.zeros((8, 128), jnp.float32)
    mesh, wait = five_times.build(10000)
    compiled = tpuasm_tools.compile(wait, x, mesh=mesh)
    (_, start, end), = [op for op in tpuasm_tools.hlo_ops(compiled) if op[0] != 'module']
    timed = clock.instrument(compiled, [start, end + 1])
    timed(x).block_until_ready()
    for run in range(2):
        samples = []
        for _ in range(CALLS):
            host_start = time.monotonic_ns()
            timed(x).block_until_ready()
            host_stop = time.monotonic_ns()
            begin, finish = (int(value) for value in clock.read(2)[0])
            samples.append((host_start, host_stop, begin, finish))
            time.sleep(GAP)
        first, last = samples[0], samples[-1]
        # 第一次运行的起点读数发生在 first 的两次主机读数之间，最后一次运行的终点读数发生在 last 的两次主机读数之间。
        cycles = last[3] - first[2]
        shortest, longest = last[0] - first[1], last[1] - first[0]
        middle = np.array([((host_start + host_stop) / 2, (begin + finish) / 2) for host_start, host_stop, begin, finish in samples])
        slope = float(np.polyfit(middle[:, 0] - middle[0, 0], middle[:, 1] - middle[0, 1], 1)[0])
        calls = np.array([host_stop - host_start for host_start, host_stop, _, _ in samples]) / 1e3
        print(f'第 {run + 1} 轮：{CALLS} 次运行，主机区间 {shortest / 1e9:.6f} 至 {longest / 1e9:.6f} 秒，LCC 增加 {cycles} 个周期')
        print(f'  频率的包络：{cycles / longest * 1e3:.4f} 至 {cycles / shortest * 1e3:.4f} MHz（宽度 {(longest - shortest) / longest * 1e6:.1f} ppm）；各次运行中点的拟合：{slope * 1e3:.4f} MHz')
        print(f'  单次调用的主机耗时：中位数 {np.median(calls):.1f} µs；若直接用一次调用的周期数除以它的主机耗时：{float(np.median([(finish - begin) / (host_stop - host_start) for host_start, host_stop, begin, finish in samples])) * 1e3:.1f} MHz')

if __name__ == '__main__':
    main()
