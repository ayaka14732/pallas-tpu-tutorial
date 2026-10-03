"""向量侧比标量侧慢的指令序列：N 条相互依赖的向量乘法、N 组从 CMEM 读入（cld 加 vpop）、一次 DMA 加等待。比较不加 sfence 的读数 R1 与加 sfence 之后的读数 R2。"""
import tpu_init
tpu_init.initialise_one_chip()

import tpuasm_tools
from tpuasm_tools import bundle, read_lcc

# v10 是输入；先把它复制一份到 CMEM 地址 0，供 cld 读取（本 kernel 独占 CMEM）。
CMEM_SETUP = bundle('s0: simm.s32 s23, 0') + bundle('s0: dma.simple [cmem:s23], [vmem:s23], length=64, dst_flag=[sflag:52]') + bundle('misc: vwait.ge [sflag:52], 64') + bundle('misc: vsyncadd.s32 [sflag:52], -64')

def dma(granules: int) -> str:
    """TC VMEM → CMEM 的一次 DMA 及其等待；载体的 DMA 信号量 sflag:52 此时为 0。"""
    return bundle(f's0: dma.simple [cmem:s23], [vmem:s23], length={granules}, dst_flag=[sflag:52]') + bundle(f'misc: vwait.ge [sflag:52], {granules}') + bundle(f'misc: vsyncadd.s32 [sflag:52], -{granules}')

def main() -> None:
    probe = tpuasm_tools.LccProbe()
    print('## N 条相互依赖的向量乘法 v11 = v11 × v10')
    for count in (1, 4, 16, 32, 64):
        body = read_lcc(20) + bundle('va0: vmul.8x128.f32 v11, v11, v10') * count + read_lcc(21) + bundle('s0: sfence') + read_lcc(22)
        print(f'  N = {count:2d}：(R1 − R0, R2 − R0) = {sorted({tuple(row) for row in probe.run(body).tolist()})}')
    print('## N 组 cld + vpop（从 CMEM 读入 TC VREG）')
    for count in (1, 4, 16, 32, 64):
        work = (bundle('cld: cld.8x128 crf, [cmem:0x0]') + bundle('vr0: vpop.8x128 v11, crf')) * count
        body = read_lcc(20) + work + read_lcc(21) + bundle('s0: sfence') + read_lcc(22)
        print(f'  N = {count:2d}：(R1 − R0, R2 − R0) = {sorted({tuple(row) for row in probe.run(body, setup=CMEM_SETUP).tolist()})}')
    print('## 一次 TC VMEM → CMEM DMA 及其等待')
    for granules in (8, 64, 256):
        body = read_lcc(20) + dma(granules) + read_lcc(21) + bundle('s0: sfence') + read_lcc(22)
        print(f'  {granules * 512 // 1024:3d} KiB：(R1 − R0, R2 − R0) = {sorted({tuple(row) for row in probe.run(body, setup=CMEM_SETUP).tolist()})}')

if __name__ == '__main__':
    main()
