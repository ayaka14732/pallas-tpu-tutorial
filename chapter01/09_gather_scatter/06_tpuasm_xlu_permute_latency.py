"""XLU 沿通道的循环移位 vrot 与重排 vperm 的时序：从提交到可以取回的延迟，以及连续提交的发射间隔。"""
import tpu_init
tpu_init.initialize_one_chip()

import tpuasm_tools
from tpuasm_tools import LCC_END, bundle, read_lcc, show_latency, show_lcc

POP = bundle('vr0: vpop.8x128 v11, trf0')
# 循环移位的位移量 5 放在 s23 中；重排的模式事先用 vsetperm 装好。
OPERATIONS = {
    'vrot（通道循环移位）': ('vx0: vrot.0.8x128 trf0, v10, s23', bundle('s0: simm.s32 s23, 5')),
    'vperm（通道重排）': ('vx0: vperm.0.8x128 trf0, v10', bundle('vx0: vsetperm.2.all.u8 pcr0, v10') + bundle('misc: vnop') * 16),
}

def main() -> None:
    probe = tpuasm_tools.LccProbe()
    for name, (instruction, setup) in OPERATIONS.items():
        show_latency(probe, f'XLU {name} → vpop trf0', instruction, 'vr0: vpop.8x128 v11, trf0', setup, distances=(1, 8, 40, 64, 68, 69, 70, 72))
    for name, (instruction, setup) in OPERATIONS.items():
        print(f'## 连续 k 次 {name}，再连续取回 k 次')
        for count in (1, 2, 4, 8):
            show_lcc(probe, f'k = {count}', read_lcc(20) + bundle(instruction) * count + POP * count + LCC_END, setup)

if __name__ == '__main__':
    main()
