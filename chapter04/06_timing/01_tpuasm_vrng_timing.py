"""用 LCC 测 vrng 的时序：结果何时可用、连续发射的间隔、va0 与 va1 是否各有一个生成器，以及它与 setrngseed、getrngseed、普通向量运算的关系。"""
import tpu_init
tpu_init.initialize_one_chip()

import tpuasm_tools
from tpuasm_tools import bundle, read_lcc

END = read_lcc(21) + bundle('s0: sfence') + read_lcc(22)
SETUP = bundle('va0: setrngseed v10') + bundle('misc: vnop') * 16
VRNG = bundle('va0: vrng.8x128.u32 v11')

def main() -> None:
    probe = tpuasm_tools.LccProbe()

    def show(label: str, body: str) -> None:
        print(f'  {label}：(R1 − R0, R2 − R0) = {sorted({tuple(row) for row in probe.run(read_lcc(20) + body + END, setup=SETUP).tolist()})}')

    print('## vrng 之后隔 d − 1 个 vnop，一条使用其结果的 vadd')
    for distance in (1, 2, 4, 8):
        show(f'd = {distance}', VRNG + bundle('misc: vnop') * (distance - 1) + bundle('va1: vadd.8x128.s32 v12, v11, v11'))
    print('## 连续 k 条 vrng（都在 va0）')
    for count in (1, 2, 4, 8, 16):
        show(f'k = {count:2d}', VRNG * count)
    print('## 连续 k 条 vrng，va0 与 va1 交替')
    for count in (2, 8):
        show(f'k = {count}', (VRNG + bundle('va1: vrng.8x128.u32 v12')) * (count // 2))
    print('## k 组“vrng + 一条无关的 vadd”')
    for count in (2, 4):
        show(f'k = {count}', (VRNG + bundle('va1: vadd.8x128.s32 v12, v10, v10')) * count)
    print('## vrng 与读写状态的指令')
    show('vrng → getrngseed', VRNG + bundle('va0: getrngseed v12'))
    show('vrng → setrngseed', VRNG + bundle('va0: setrngseed v10'))
    show('setrngseed → vrng', bundle('va0: setrngseed v10') + VRNG)

if __name__ == '__main__':
    main()
