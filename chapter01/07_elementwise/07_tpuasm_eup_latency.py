"""EUP 的时序：五种函数从提交到可以取回的延迟，以及连续提交、连续取回的发射间隔。"""
import tpu_init
tpu_init.initialize_one_chip()

import tpuasm_tools
from tpuasm_tools import LCC_END, bundle, read_lcc, show_latency, show_lcc

def main() -> None:
    probe = tpuasm_tools.LccProbe()
    for function in ('vpow2', 'vrcp', 'vrsqrt', 'vtanh', 'vlog2'):
        show_latency(probe, f'EUP {function} → vpop erf', f'va0: {function}.8x128.f32 erf, v10', 'vr0: vpop.8x128 v11, erf')
    push, pop = bundle('va0: vpow2.8x128.f32 erf, v10'), bundle('vr0: vpop.8x128 v11, erf')
    print('## 连续提交 k 次，读数之后才取回')
    for count in (1, 2, 4, 8, 16):
        show_lcc(probe, f'k = {count:2d}', read_lcc(20) + push * count + LCC_END + pop * count)
    print('## 连续提交 k 次，再连续取回 k 次')
    for count in (1, 2, 4, 8, 16):
        show_lcc(probe, f'k = {count:2d}', read_lcc(20) + push * count + pop * count + LCC_END)

if __name__ == '__main__':
    main()
