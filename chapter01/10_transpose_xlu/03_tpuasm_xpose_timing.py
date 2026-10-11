"""XLU 转置的时序：最后一次提交到第一次取回的延迟，提交与取回的发射间隔，一次完整的 128×128 转置，两个 XLU 同时转置，以及打包的 bf16 和转置后只有 8 行的变体。"""
import tpu_init
tpu_init.initialize_one_chip()

import tpuasm_tools
from tpuasm_tools import LCC_END, bundle, read_lcc, show_latency, show_lcc

POP = bundle('vr0: vpop.8x128 v11, trf0')

def xpose(index: int, count: int = 16, unit: int = 0, packed: str = '', width: int = 128) -> str:
    """count 次提交组成的一次转置中的第 index 次，写法与编译器生成的相同；unit 是 0 或 1，分别用 vx0/trf0 与 vx1/trf1；packed 为 '.packed' 时输入是打包的 16 bit 数据，width 是转置后的行数。"""
    suffix = '.start' if index == 0 else '.end' if index == count - 1 else ''
    variant = unit if packed else index % 2 * 2 + unit
    return f'vx{unit}: vxpose.{variant}{packed}{suffix}.8x128 trf{unit}, v10, {width}'

def main() -> None:
    probe = tpuasm_tools.LccProbe()
    # 转置的结果有 16 个，其余 15 个在读数之后取回，否则 kernel 结束时队列不空，TensorCore 会停机。
    show_latency(probe, 'XLU 最后一次 vxpose → 第一次 vpop trf0（前 15 次提交在计时之前）', xpose(15), 'vr0: vpop.8x128 v11, trf0', ''.join(bundle(xpose(index)) for index in range(15)), POP * 15)
    submit = ''.join(bundle(xpose(index)) for index in range(16))
    print('## 16 次 vxpose，提交之间插入 g 个 vnop（读数之后才取回）')
    for gap in (0, 1, 2, 4, 8):
        spaced = ''.join(bundle(xpose(index)) + bundle('misc: vnop') * gap for index in range(16))
        show_lcc(probe, f'g = {gap}', read_lcc(20) + spaced + LCC_END + POP * 16)
    print('## 转置完成后，连续取回 k 次')
    for count in (1, 2, 4, 8, 16):
        show_lcc(probe, f'k = {count:2d}', submit + bundle('s0: sfence') + read_lcc(20) + POP * count + LCC_END + POP * (16 - count))
    print('## 一次完整的 128×128 转置：16 次提交，16 次取回')
    show_lcc(probe, '一个 XLU', read_lcc(20) + submit + POP * 16 + LCC_END)
    print('## 两个 128×128 转置，分给两个 XLU')
    both = ''.join(bundle(xpose(index) + ' ; ' + xpose(index, unit=1)) for index in range(16))
    pop_both = bundle('vr0: vpop.8x128 v11, trf0 ; vr1: vpop.8x128 v12, trf1')
    show_lcc(probe, '两边的 vpop 放进同一个 bundle', read_lcc(20) + both + pop_both * 16 + LCC_END)
    show_lcc(probe, '先取完 trf0，再取 trf1', read_lcc(20) + both + POP * 16 + bundle('vr1: vpop.8x128 v12, trf1') * 16 + LCC_END)
    print('## 打包的 bf16 转置（8 次提交）与转置后只有 8 行的转置（16 次提交、宽度 8）')
    packed = ''.join(bundle(xpose(index, 8, packed='.packed')) for index in range(8))
    show_lcc(probe, '8 次打包提交，读数之后才取回', read_lcc(20) + packed + LCC_END + POP * 8)
    show_lcc(probe, '8 次打包提交 + 8 次取回', read_lcc(20) + packed + POP * 8 + LCC_END)
    narrow = ''.join(bundle(xpose(index, width=8)) for index in range(16))
    show_lcc(probe, '16 次宽度 8 的提交 + 1 次取回', read_lcc(20) + narrow + POP + LCC_END)

if __name__ == '__main__':
    main()
