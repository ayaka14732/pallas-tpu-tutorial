"""同一颗芯片两个 TensorCore 的 GTC 偏移，精确到一个计数：两个 TensorCore 共用一个时钟，互发信号时用 LCC 记下发出与收到的时刻，由两个方向的最小时延求出两个 LCC 的固定差 D，再把两边的 GTC 读数换到同一时刻比较。"""
import tpu_init
tpu_init.initialize_one_chip()

from collections import Counter
import importlib.util
from pathlib import Path

import numpy as np

import tpuasm_tools
from tpuasm_tools import bundle, read_gtc, read_lcc

spec = importlib.util.spec_from_file_location('two_core', Path(__file__).with_name('04_tpuasm_two_core_gtc.py'))
two_core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(two_core)
# G0、L0（发出时刻）→ 给对方发信号 → 等对方的信号 → sfence → L1（收到时刻）、G1。
BODY = (
    read_gtc(20)
    + read_lcc(21)
    + bundle('misc: vsyncadd.remote.s32 [sflag:s29], 1')
    + bundle('misc: vwait.ge [sflag:45], 1')
    + bundle('misc: vsyncadd.s32 [sflag:45], -1')
    + bundle('s0: sfence')
    + read_lcc(22)
    + read_gtc(23)
)
REPEATS = 400

def smallest(values: np.ndarray) -> str:
    return '、'.join(f'{value}（{count} 次）' for value, count in sorted(Counter(values.tolist()).items())[:4])

def main() -> None:
    probe = tpuasm_tools.LccProbe(num_cores=2)
    counters = probe.run_raw(BODY, 4, repeats=REPEATS, setup=two_core.REMOTE_FLAG).astype(np.int64)
    g0, sent, received, g1 = counters[:, :, 0], counters[:, :, 1], counters[:, :, 2], counters[:, :, 3]
    # 每个方向：收到时刻（接收方的 LCC）减去发出时刻（发送方的 LCC）。
    forward = received[:, 1] - sent[:, 0]
    backward = received[:, 0] - sent[:, 1]
    print(f'{REPEATS} 次运行')
    print(f'  TensorCore 0 → 1：收到时刻 − 发出时刻，最小的几个值：{smallest(forward)}')
    print(f'  TensorCore 1 → 0：收到时刻 − 发出时刻，最小的几个值：{smallest(backward)}')
    # 两个 LCC 若不同步，D 会随时间漂移，最小值就不会在整个实验期间反复出现。
    span = int(sent[-1, 0] - sent[0, 0])
    hits = np.flatnonzero(forward == forward.min())
    print(f'  第一次与最后一次运行相隔 {span} 个周期；0 → 1 取到最小值的运行从第 {hits[0]} 次到第 {hits[-1]} 次，各四分之一中的次数 {[int(np.sum((hits >= k * REPEATS // 4) & (hits < (k + 1) * REPEATS // 4))) for k in range(4)]}')
    hits = np.flatnonzero(backward == backward.min())
    print(f'  1 → 0 取到最小值的运行从第 {hits[0]} 次到第 {hits[-1]} 次，各四分之一中的次数 {[int(np.sum((hits >= k * REPEATS // 4) & (hits < (k + 1) * REPEATS // 4))) for k in range(4)]}')
    total, difference = int(forward.min() + backward.min()), int(forward.min() - backward.min())
    print(f'  两个最小值之和 {total}，之差 {difference}：单程时延 d = {total // 2} 个周期，LCC 之差 D = {difference // 2}')
    offset = difference // 2
    # 每个 TensorCore 上 3G − 32L 只取三个值（GTC 的三种相位）；G0 比 L0 早一个周期，G1 比 L1 晚一个周期。
    phases = []
    for core in range(2):
        values = np.concatenate([3 * g0[:, core] - 32 * (sent[:, core] - 1), 3 * g1[:, core] - 32 * (received[:, core] + 1)])
        phases.append(sorted(set(values.tolist())))
        print(f'  TensorCore {core} 的 3G − 32L：{phases[core]}')
    print(f'  同一时刻两个 TensorCore 的 GTC 之差的 3 倍，按三种相位分别计算：{[second - first + 32 * offset for first, second in zip(*phases)]}')
    lower, upper = g0[:, 1] - g1[:, 0], g1[:, 1] - g0[:, 0]
    print(f'  对照：只用因果顺序得到的区间 [{int(lower.max())}, {int(upper.min())}] 个 GTC 计数')

if __name__ == '__main__':
    main()
