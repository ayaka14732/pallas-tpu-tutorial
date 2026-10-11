"""bundle 数不是周期数：N 条向量指令排成 N 个 bundle，相互依赖与否、结果要几个周期才可用，决定了它们实际要多少周期。"""
import tpu_init
tpu_init.initialize_one_chip()

import tpuasm_tools
from tpuasm_tools import bundle, read_lcc

# 每种序列给出第一条指令和后续每一条指令：相互依赖的序列中，后一条用前一条写出的 v11。
SEQUENCES = {
    '互不依赖的 vadd.f32': ('va1: vadd.8x128.f32 v11, v10, v10', 'va1: vadd.8x128.f32 v11, v10, v10'),
    '相互依赖的 vadd.f32': ('va1: vadd.8x128.f32 v11, v10, v10', 'va1: vadd.8x128.f32 v11, v11, v10'),
    '互不依赖的 vmul.f32': ('va0: vmul.8x128.f32 v11, v10, v10', 'va0: vmul.8x128.f32 v11, v10, v10'),
    '相互依赖的 vmul.f32': ('va0: vmul.8x128.f32 v11, v10, v10', 'va0: vmul.8x128.f32 v11, v11, v10'),
}

def main() -> None:
    probe = tpuasm_tools.LccProbe()
    for name, (first, rest) in SEQUENCES.items():
        print(f'## {name}')
        for count in (1, 4, 16):
            body = read_lcc(20) + bundle(first) + bundle(rest) * (count - 1) + read_lcc(21) + bundle('s0: sfence') + read_lcc(22)
            values = sorted({tuple(row) for row in probe.run(body).tolist()})
            print(f'  N = {count:2d}：(R1 − R0, R2 − R0) = {values}')

if __name__ == '__main__':
    main()
