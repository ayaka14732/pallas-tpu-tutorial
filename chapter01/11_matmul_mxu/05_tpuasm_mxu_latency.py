"""MXU 的时序：vmatmul 到可以取回的延迟，提交与取回的发射间隔，以及提交与取回交错排列的效果。"""
import tpu_init
tpu_init.initialize_one_chip()

import tpuasm_tools
from tpuasm_tools import LCC_END, bundle, read_lcc, show_latency, show_lcc

# 计时之前先装好权重。
SETUP = bundle('vx0: vmatpush.packed.8x128.f16 gsfn0, v10') * 8 + bundle('vx0: vdwg.128x128.f16 gmr0, gsfn0')
MATMUL = bundle('vx0: vmatmul.8x128.f32 mrf0, v10')
POP = bundle('vr0: vpop.8x128 v11, mrf0')

def main() -> None:
    probe = tpuasm_tools.LccProbe()
    show_latency(probe, 'MXU vmatmul → vpop mrf0', 'vx0: vmatmul.8x128.f32 mrf0, v10', 'vr0: vpop.8x128 v11, mrf0', SETUP, distances=(1, 12, 40, 80, 82, 83, 84, 88))
    print('## 连续 k 次 vmatmul，再连续取回 k 次')
    for count in (1, 2, 4, 8, 12, 16):
        show_lcc(probe, f'k = {count:2d}', read_lcc(20) + MATMUL * count + POP * count + LCC_END, SETUP)
    print('## 8 次 vmatmul，200 个 vnop 之后取回 j 次（其余的在读数之后取回）')
    for count in (1, 2, 4, 8):
        show_lcc(probe, f'j = {count}', read_lcc(20) + MATMUL * 8 + bundle('misc: vnop') * 200 + POP * count + LCC_END + POP * (8 - count), SETUP)
    print('## 16 次 vmatmul：从第 9 次起与 vpop 交错')
    interleaved = MATMUL * 8 + (MATMUL + POP) * 8 + POP * 8
    show_lcc(probe, '8 次提交，8 组（提交、取回），8 次取回', read_lcc(20) + interleaved + LCC_END, SETUP)

if __name__ == '__main__':
    main()
