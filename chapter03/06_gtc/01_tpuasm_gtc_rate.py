"""LCC 与 GTC 的计数速率：用 vdelay 让向量一侧停住 H 个周期，在其两侧用 LCC 包住 GTC 读数；再与主机计时比较，得出两者的频率。最后连续读 4 次 GTC，看它每个周期加多少，并把读数拆成高 60 位与低 4 位；再读出单芯片会话中这颗芯片的同步角色。"""
import tpu_init
tpu_init.initialise_one_chip()

from collections import Counter
import time

import numpy as np

import gtc_common
import tpuasm_tools
from tpuasm_tools import bundle, read_gtc, read_lcc

REPEATS = 2000
DELAYS = (10**3, 10**4, 10**5, 10**6, 10**7, 10**8, 10**9)

def main() -> None:
    probe = tpuasm_tools.LccProbe()
    print('## L0、G0、vdelay H、sfence、G1、L1')
    rows = []
    for delay in DELAYS:
        setup = bundle(f's0: simm.s32 s24, {delay}')
        body = read_lcc(20) + read_gtc(21) + bundle('misc: vdelay s24') + bundle('s0: sfence') + read_gtc(22) + read_lcc(23)
        start = time.perf_counter()
        counters = probe.run_raw(body, 4, repeats=4, setup=setup)[:, 0].astype(np.int64)
        seconds = (time.perf_counter() - start) / 4
        lcc = sorted(set((counters[:, 3] - counters[:, 0]).tolist()))
        gtc = counters[:, 2] - counters[:, 1]
        # LCC 的两端各比 GTC 多一个相邻读数，GTC 两个读数之间的本地周期数是 ΔL − 2。
        ratio = gtc / (counters[:, 3] - counters[:, 0] - 2)
        rows.append((lcc[0], int(np.median(gtc)), seconds))
        print(f'  H = {delay:>10}：ΔL = {lcc}，ΔG = {sorted(set(gtc.tolist()))}，ΔG / (ΔL − 2) = {ratio.min():.6f}–{ratio.max():.6f}，主机时间 {seconds * 1e3:.3f} ms（含装载程序，取差值时抵消）')
    (lcc_1, gtc_1, host_1), (lcc_2, gtc_2, host_2) = rows[-2], rows[-1]
    print(f'H = 10^8 与 10^9 之差：LCC 频率 {(lcc_2 - lcc_1) / (host_2 - host_1) / 1e9:.4f} GHz，GTC 频率 {(gtc_2 - gtc_1) / (host_2 - host_1) / 1e9:.3f} GHz（以主机时钟为准）')
    print('## 连续 4 次读 GTC，相邻读数之差')
    body = read_gtc(20) + read_gtc(21) + read_gtc(22) + read_gtc(23)
    counters = probe.run_raw(body, 4, repeats=REPEATS)[:, 0].astype(np.int64)
    patterns = Counter(tuple(np.diff(row).tolist()) for row in counters)
    print(f'  {REPEATS} 次运行出现的 (G1 − G0, G2 − G1, G3 − G2) 与次数：{sorted(patterns.items())}')
    print(f'  G3 − G0：{sorted(set((counters[:, 3] - counters[:, 0]).tolist()))}')
    print(f'  低 4 位的取值与次数：{sorted(Counter((counters & 15).ravel().tolist()).items())}')
    print('## 把读数拆成高 60 位 T = G >> 4 与低 4 位 O = G & 15（每种轮换取一次运行，T 以第一次读数为 0）')
    for pattern in sorted(patterns):
        row = next(row for row in counters if tuple(np.diff(row).tolist()) == pattern)
        print(f'  增量 {pattern}：T = {((row >> 4) - (row[0] >> 4)).tolist()}，O = {(row & 15).tolist()}')
    print('## 单芯片会话中这颗芯片 6 个 ICI 端口的同步角色（从主机只读地读寄存器）')
    first = gtc_common.read_ports(0)
    time.sleep(0.01)
    second = gtc_common.read_ports(0)
    print(f'  {gtc_common.describe(first)}')
    print(f'  相隔 10 ms 两次读数，6 个端口的计数都在增加：{all(later > earlier for (_, earlier), (_, later) in zip(first, second))}')

if __name__ == '__main__':
    main()
