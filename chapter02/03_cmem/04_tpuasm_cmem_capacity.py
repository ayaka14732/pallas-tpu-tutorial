"""Megacore Shared CMEM 有多大：把一个 tile 写到 CMEM 地址 0，把它的按位取反写到地址 A，再把两处读回。A 超出容量时会绕回到已有的地址，地址 0 的内容随之被改写。"""
import tpu_init
tpu_init.initialize_one_chip()

import tpuasm_tools
from tpuasm_tools import bundle

GAP = bundle('misc: vnop') * 16
ADDRESSES = (0x1000, 0x20000, 0x3FFF8, 0x40000, 0x7FFF8, 0x80000)

def dma(destination: str, source: str) -> str:
    """搬 8 个 granule（一个 f32 tile）并等待完成。"""
    return bundle(f's0: dma.simple {destination}, {source}, length=8, dst_flag=[sflag:52]') + bundle('misc: vwait.ge [sflag:52], 8') + bundle('misc: vsyncadd.s32 [sflag:52], -8')

def body(address: int) -> str:
    return (
        # 输入的第 0 个 tile（v10）记作 P，写到 TC VMEM 的 0x100；它的按位取反记作 Q，写到 0x108。
        bundle('vst: vst.8x128 [vmem:0x100], v10')
        + bundle('va0: vxor.8x128.u32 v11, 0xffffffff, v10')
        + GAP
        + bundle('vst: vst.8x128 [vmem:0x108], v11')
        + GAP
        + bundle('s0: simm.s32 s20, 0')
        + bundle(f's0: simm.s32 s21, {address}')
        + bundle('s0: simm.s32 s22, 256')
        + bundle('s0: simm.s32 s23, 264')
        + bundle('s0: simm.s32 s24, 8')
        + bundle('s0: simm.s32 s26, 16')
        # P → CMEM 地址 0，Q → CMEM 地址 A；再把两处读回输出的第 1、2 个 tile。
        + dma('[cmem:s20]', '[vmem:s22]')
        + dma('[cmem:s21]', '[vmem:s23]')
        + dma('[vmem:s24]', '[cmem:s20]')
        + dma('[vmem:s26]', '[cmem:s21]')
        + bundle('s0: sfence')
    )

def main() -> None:
    probe = tpuasm_tools.LccProbe()
    pattern = probe.host[:8]
    for address in ADDRESSES:
        tiles = probe.run_tiles(body(address))[0, 0]
        print(f'A = 0x{address:x}（{address * 512 / 2**20:.3f} MiB 处）：地址 A 读回 Q {bool((tiles[1] == ~pattern).all())}；地址 0 仍是 P {bool((tiles[0] == pattern).all())}，变成了 Q {bool((tiles[0] == ~pattern).all())}')

if __name__ == '__main__':
    main()
