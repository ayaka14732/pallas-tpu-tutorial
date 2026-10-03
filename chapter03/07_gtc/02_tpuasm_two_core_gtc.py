"""同一颗芯片的两个 TensorCore 各读 GTC：互发一次信号，用“先读、再发信号”和“收到信号、再读”的因果顺序，界定两个 TensorCore 的 GTC 偏移；同时读 LCC，对照两个 TensorCore 的 LCC 是否可比。"""
import tpu_init
tpu_init.initialise_one_chip()

import numpy as np

import tpuasm_tools
from tpuasm_tools import bundle, read_gtc, read_lcc

# 与编译器生成的入口汇合相同：由 SMEM 中的芯片信息和本核编号拼出对方 TensorCore 的 sflag 45 地址，放进 s29。
REMOTE_FLAG = (
    bundle('s1: sld s29, [smem:0x0]')
    + bundle('s1: sld s30, [smem:0x1]')
    + bundle('s0: sand.u32 s29, 0xfff, s29 ; s1: ssub.s32 s30, 1, s30')
    + bundle('s0: sshll.u32 s29, s29, 0x12 ; s1: sshll.u32 s30, s30, 0xe')
    + bundle('s0: sor.u32 s29, s29, s30')
    + bundle('s0: sor.u32 s29, 0x802d, s29')
)
# G0 → 给对方发信号 → 等对方的信号 → sfence → G1，紧接着读 LCC。
BODY = (
    read_gtc(20)
    + bundle('misc: vsyncadd.remote.s32 [sflag:s29], 1')
    + bundle('misc: vwait.ge [sflag:45], 1')
    + bundle('misc: vsyncadd.s32 [sflag:45], -1')
    + bundle('s0: sfence')
    + read_gtc(21)
    + read_lcc(22)
)
REPEATS = 32

def main() -> None:
    probe = tpuasm_tools.LccProbe(num_cores=2)
    counters = probe.run_raw(BODY, 3, repeats=REPEATS, setup=REMOTE_FLAG).astype(np.int64)
    g0, g1, lcc = counters[:, :, 0], counters[:, :, 1], counters[:, :, 2]
    # TensorCore 0 的 G0 早于 TensorCore 1 的 G1，反之亦然；记 o 为 TensorCore 1 的 GTC 减去 TensorCore 0 的 GTC 的固定偏移。
    lower = g0[:, 1] - g1[:, 0]
    upper = g1[:, 1] - g0[:, 0]
    print(f'{REPEATS} 次运行')
    print(f'  每次的 G1 − G0（TensorCore 0）：{int(np.min(g1[:, 0] - g0[:, 0]))}–{int(np.max(g1[:, 0] - g0[:, 0]))} 个 GTC 计数')
    print(f'  每次的 G1 − G0（TensorCore 1）：{int(np.min(g1[:, 1] - g0[:, 1]))}–{int(np.max(g1[:, 1] - g0[:, 1]))} 个 GTC 计数')
    print(f'  两个 TensorCore 的 G0 之差：{int(np.min(g0[:, 1] - g0[:, 0]))}–{int(np.max(g0[:, 1] - g0[:, 0]))} 个 GTC 计数')
    print(f'  单次运行给出的偏移区间宽度：{int(np.min(upper - lower))}–{int(np.max(upper - lower))} 个 GTC 计数')
    print(f'  {REPEATS} 个区间的交集：[{int(np.max(lower))}, {int(np.min(upper))}] 个 GTC 计数，即 [{np.max(lower) / 11.2:.1f}, {np.min(upper) / 11.2:.1f}] ns')
    print(f'  紧随 G1 的 LCC 读数之差（TensorCore 1 − TensorCore 0）：{int(np.min(lcc[:, 1] - lcc[:, 0]))}–{int(np.max(lcc[:, 1] - lcc[:, 0]))}')
    print(f'  TensorCore 0 的 LCC 读数：{int(lcc[0, 0])}，TensorCore 1：{int(lcc[0, 1])}')

if __name__ == '__main__':
    main()
