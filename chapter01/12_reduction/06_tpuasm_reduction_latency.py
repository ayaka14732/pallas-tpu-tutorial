"""归约用到的三条通路的时序：XLU 的跨通道归约 vadd.xlane，向量 ALU 中的子通道循环移位 vrot.slane.down，以及从向量单元到标量单元的 vpush → spop。"""
import tpu_init
tpu_init.initialize_one_chip()

import tpuasm_tools
from tpuasm_tools import LCC_END, bundle, read_lcc, show_latency, show_lcc

POP = bundle('vr0: vpop.8x128 v11, trf0')

def main() -> None:
    probe = tpuasm_tools.LccProbe()
    for mnemonic in ('vadd.xlane', 'vmax.xlane', 'vmin.xlane'):
        show_latency(probe, f'XLU {mnemonic} → vpop trf0', f'vx0: {mnemonic}.0.8x128.f32 trf0, v10', 'vr0: vpop.8x128 v11, trf0', distances=(1, 8, 40, 76, 78, 79, 80, 84))
    print('## 连续 k 次 vadd.xlane，再连续取回 k 次（一个队列）')
    for count in (1, 2, 4, 8, 16):
        show_lcc(probe, f'k = {count:2d}', read_lcc(20) + bundle('vx0: vadd.xlane.0.8x128.f32 trf0, v10') * count + POP * count + LCC_END)
    print('## 2k 次 vadd.xlane 分给两个队列：每个 bundle 向 trf0、trf1 各提交一次，再成对取回')
    for count in (1, 2, 4, 8):
        submit = bundle('vx0: vadd.xlane.0.8x128.f32 trf0, v10 ; vx1: vadd.xlane.1.8x128.f32 trf1, v10')
        fetch = bundle('vr0: vpop.8x128 v11, trf0 ; vr1: vpop.8x128 v12, trf1')
        show_lcc(probe, f'k = {count}', read_lcc(20) + submit * count + fetch * count + LCC_END)
    print('## N 条相互依赖的 vrot.slane.down（子通道循环移位，在向量 ALU 中执行）')
    for count in (1, 4, 7, 16):
        show_lcc(probe, f'N = {count:2d}', read_lcc(20) + bundle('va0: vrot.slane.down.8x128.u32 v11, v10') + bundle('va0: vrot.slane.down.8x128.u32 v11, v11') * (count - 1) + LCC_END)
    print('## 向量 → 标量：vpush v2sf 之后隔 d − 1 个 vnop 再 spop')
    for distance in (1, 12, 40, 42, 44, 48):
        show_lcc(probe, f'd = {distance:2d}', read_lcc(20) + bundle('vst: vpush v2sf, v10') + bundle('misc: vnop') * (distance - 1) + bundle('s0: spop s24, v2sf') + LCC_END)

if __name__ == '__main__':
    main()
