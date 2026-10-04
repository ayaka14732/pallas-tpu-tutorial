"""标量单元的时序：相互依赖的 smul、连续的 sld、sld 的结果多久后可用、sst，以及 sst 之后读同一地址。每种序列给出 R1 − R0 与 R2 − R0。"""
import tpu_init
tpu_init.initialise_one_chip()

import tpuasm_tools
from tpuasm_tools import bundle, read_lcc

END = read_lcc(21) + bundle('s0: sfence') + read_lcc(22)
LOAD = bundle('s1: sld s24, [smem:0x0]')
USE = bundle('s0: sadd.s32 s24, 1, s24')

def main() -> None:
    probe = tpuasm_tools.LccProbe()

    def show(label: str, body: str) -> None:
        print(f'  {label}：(R1 − R0, R2 − R0) = {sorted({tuple(row) for row in probe.run(read_lcc(20) + body + END).tolist()})}')

    print('## N 条相互依赖的标量乘法 s24 = 3 × s24')
    for count in (1, 4, 16):
        show(f'N = {count:2d}', bundle('s0: smul.u32 s24, 3, s24') * count)
    print('## N 条连续的 sld（彼此无关）')
    for count in (1, 2, 4, 16):
        show(f'N = {count:2d}', LOAD * count)
    print('## N 组“sld + 3 条无关的标量加法”')
    for count in (2, 4):
        show(f'N = {count}', (LOAD + bundle('s0: sadd.s32 s23, 1, s23') * 3) * count)
    print('## sld 之后隔 d − 1 个空 bundle，一条使用其结果的 sadd')
    for distance in (1, 2, 3, 4, 5):
        show(f'd = {distance}', LOAD + bundle() * (distance - 1) + USE)
    print('## N 条连续的 sst')
    for count in (1, 4, 16):
        show(f'N = {count:2d}', bundle('s1: sst [smem:0x7f0], s24') * count)
    print('## sst 之后读同一地址，再使用读到的值')
    show('sst → sld → sadd', bundle('s1: sst [smem:0x7f0], s24') + bundle('s1: sld s23, [smem:0x7f0]') + bundle('s0: sadd.s32 s23, 1, s23'))
    print('## sld 之后立即使用：读到的是新值还是旧值')
    # 先把 7 存进 SMEM、把 s24 清零；再读回 s24，隔 d − 1 个空 bundle 后由 vmov 或 sadd 使用，结果写进 tile 1。
    gap = bundle('misc: vnop') * 16
    prepare = bundle('s0: simm.s32 s24, 7') + bundle('s1: sst [smem:0x7f0], s24') + bundle('s0: simm.s32 s24, 0') + bundle('s0: sfence') + bundle('s1: sld s24, [smem:0x7f0]')
    for name, use in (('vmov v11, s24', bundle('va0: vmov.8x128 v11, s24')), ('sadd s23 = s24 + 100，再 vmov', bundle('s0: sadd.s32 s23, 100, s24') + bundle('va0: vmov.8x128 v11, s23'))):
        values = [int(probe.run_tiles(prepare + bundle() * (distance - 1) + use + gap + bundle('vst: vst.8x128 [vmem:0x8], v11') + gap)[0, 0, 0, 0, 0]) for distance in (1, 2, 3)]
        print(f'  {name}，d = 1、2、3：{values}')

if __name__ == '__main__':
    main()
