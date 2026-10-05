"""“提交—取回”通路的时序参数。延迟：在提交与取回之间插入 d − 1 个 vnop，看 R2 − R0 何时开始随 d 增加；发射间隔：连续发射 k 次，看 R2 − R0 每次增加多少。"""
import tpu_init
tpu_init.initialize_one_chip()

import tpuasm_tools
from tpuasm_tools import bundle, read_lcc

DISTANCES = (1, 2, 4, 6, 7, 8, 12, 40)
END = read_lcc(21) + bundle('s0: sfence') + read_lcc(22)
# 发射之前的准备放在 setup 中，不计入区间：MXU 先装好权重，XLU 先提交转置的前 15 个 TC VREG。
MXU_SETUP = bundle('vx0: vmatpush.packed.8x128.f16 gsfn0, v10') * 8 + bundle('vx0: vdwg.128x128.f16 gmr0, gsfn0')
POP_TRF = bundle('vr0: vpop.8x128 v11, trf0')

def xpose(index: int) -> str:
    """一次 128×128 f32 转置的第 index 次提交，写法与编译器生成的相同。"""
    suffix = '.start' if index == 0 else '.end' if index == 15 else ''
    return f'vx0: vxpose.{index % 2 * 2}{suffix}.8x128 trf0, v10, 128'

XLU_SETUP = ''.join(bundle(xpose(index)) for index in range(15))
PATHS = {
    'EUP vpow2 → vpop erf': ('', 'va0: vpow2.8x128.f32 erf, v10', 'vr0: vpop.8x128 v11, erf', ''),
    'EUP vrcp → vpop erf': ('', 'va0: vrcp.8x128.f32 erf, v10', 'vr0: vpop.8x128 v11, erf', ''),
    'EUP vrsqrt → vpop erf': ('', 'va0: vrsqrt.8x128.f32 erf, v10', 'vr0: vpop.8x128 v11, erf', ''),
    'EUP vtanh → vpop erf': ('', 'va0: vtanh.8x128.f32 erf, v10', 'vr0: vpop.8x128 v11, erf', ''),
    'EUP vlog2 → vpop erf': ('', 'va0: vlog2.8x128.f32 erf, v10', 'vr0: vpop.8x128 v11, erf', ''),
    'MXU vmatmul → vpop mrf0': (MXU_SETUP, 'vx0: vmatmul.8x128.f32 mrf0, v10', 'vr0: vpop.8x128 v11, mrf0', ''),
    # 转置的结果有 16 个，其余 15 个在读数之后取回，否则 kernel 结束时队列不空，TensorCore 会停机。
    'XLU 最后一次 vxpose → 第一次 vpop trf0': (XLU_SETUP, xpose(15), 'vr0: vpop.8x128 v11, trf0', POP_TRF * 15),
    'XLU vadd.xlane → vpop trf0': ('', 'vx0: vadd.xlane.0.8x128.f32 trf0, v10', 'vr0: vpop.8x128 v11, trf0', ''),
    # lane 循环移位：位移量 5 放在 s23 中。
    'XLU vrot（lane 循环移位）→ vpop trf0': (bundle('s0: simm.s32 s23, 5'), 'vx0: vrot.0.8x128 trf0, v10, s23', 'vr0: vpop.8x128 v11, trf0', ''),
    # lane 重排：重排模式事先用 vsetperm 装好。
    'XLU vperm（lane 重排）→ vpop trf0': (bundle('vx0: vsetperm.2.all.u8 pcr0, v10') + bundle('misc: vnop') * 16, 'vx0: vperm.0.8x128 trf0, v10', 'vr0: vpop.8x128 v11, trf0', ''),
}

def show(probe: tpuasm_tools.LccProbe, label: str, body: str, setup: str = '') -> None:
    print(f'  {label}：(R1 − R0, R2 − R0) = {sorted({tuple(row) for row in probe.run(body, setup=setup).tolist()})}')

def main() -> None:
    probe = tpuasm_tools.LccProbe()
    for name, (setup, issue, fetch, after) in PATHS.items():
        print(f'## {name}')
        for distance in DISTANCES:
            show(probe, f'd = {distance:2d}', read_lcc(20) + bundle(issue) + bundle('misc: vnop') * (distance - 1) + bundle(fetch) + END + after, setup)
    print('## 向量 → 标量：vpush v2sf → spop')
    for distance in (1, 12, 40, 42, 44, 48):
        show(probe, f'd = {distance:2d}', read_lcc(20) + bundle('vst: vpush v2sf, v10') + bundle('misc: vnop') * (distance - 1) + bundle('s0: spop s24, v2sf') + END)
    push, pop = bundle('va0: vpow2.8x128.f32 erf, v10'), bundle('vr0: vpop.8x128 v11, erf')
    print('## EUP：连续提交 k 次，读数之后才取回')
    for count in (1, 2, 4, 8, 16):
        show(probe, f'k = {count:2d}', read_lcc(20) + push * count + END + pop * count)
    print('## EUP：连续提交 k 次，再连续取回 k 次')
    for count in (1, 2, 4, 8, 16):
        show(probe, f'k = {count:2d}', read_lcc(20) + push * count + pop * count + END)
    print('## MXU：连续 k 次 vmatmul，再连续取回 k 次')
    for count in (1, 2, 4, 8, 16):
        show(probe, f'k = {count:2d}', read_lcc(20) + bundle('vx0: vmatmul.8x128.f32 mrf0, v10') * count + bundle('vr0: vpop.8x128 v11, mrf0') * count + END, MXU_SETUP)
    print('## MXU：8 次 vmatmul，200 个 vnop 之后取回 j 次（其余的在读数之后取回）')
    for count in (1, 2, 4, 8):
        pops = bundle('vr0: vpop.8x128 v11, mrf0')
        show(probe, f'j = {count}', read_lcc(20) + bundle('vx0: vmatmul.8x128.f32 mrf0, v10') * 8 + bundle('misc: vnop') * 200 + pops * count + END + pops * (8 - count), MXU_SETUP)
    print('## XLU：16 次 vxpose，提交之间插入 g 个 vnop')
    for gap in (0, 1, 2, 4, 8):
        submit = ''.join(bundle(xpose(index)) + bundle('misc: vnop') * gap for index in range(16))
        show(probe, f'g = {gap}', read_lcc(20) + submit + END + POP_TRF * 16)
    print('## XLU：转置完成后，连续取回 k 次')
    for count in (1, 2, 4, 8, 16):
        submit = ''.join(bundle(xpose(index)) for index in range(16)) + bundle('s0: sfence')
        show(probe, f'k = {count:2d}', submit + read_lcc(20) + POP_TRF * count + END + POP_TRF * (16 - count))
    xlu_variants(probe)
    for name, instruction, setup in (
        ('vadd.xlane', 'vx0: vadd.xlane.0.8x128.f32 trf0, v10', ''),
        ('vrot', 'vx0: vrot.0.8x128 trf0, v10, s23', bundle('s0: simm.s32 s23, 5')),
        ('vperm', 'vx0: vperm.0.8x128 trf0, v10', bundle('vx0: vsetperm.2.all.u8 pcr0, v10') + bundle('misc: vnop') * 16),
    ):
        print(f'## XLU：连续 k 次 {name}，再连续取回 k 次')
        for count in (1, 2, 4, 8):
            show(probe, f'k = {count}', read_lcc(20) + bundle(instruction) * count + POP_TRF * count + END, setup)
    print('## N 条相互依赖的 vrot.slane.down（sublane 循环移位，在向量 ALU 中执行）')
    for count in (1, 4, 7, 16):
        show(probe, f'N = {count:2d}', read_lcc(20) + bundle('va0: vrot.slane.down.8x128.u32 v11, v10') + bundle('va0: vrot.slane.down.8x128.u32 v11, v11') * (count - 1) + END)

def xpose_variant(index: int, count: int, packed: str, width: int) -> str:
    """count 次提交组成的一次转置中的第 index 次；packed 为 '.packed' 时输入是打包的 16 bit 数据，width 是转置后的行数。"""
    suffix = '.start' if index == 0 else '.end' if index == count - 1 else ''
    return bundle(f'vx0: vxpose.0{packed}{suffix}.8x128 trf0, v10, {width}')

def xlu_variants(probe: tpuasm_tools.LccProbe) -> None:
    print('## XLU：打包的 bf16 转置（8 次提交）与转置后只有 8 行的转置（16 次提交、宽度 8）')
    packed = ''.join(xpose_variant(index, 8, '.packed', 128) for index in range(8))
    show(probe, '8 次打包提交，读数之后才取回', read_lcc(20) + packed + END + POP_TRF * 8)
    show(probe, '8 次打包提交 + 8 次取回', read_lcc(20) + packed + POP_TRF * 8 + END)
    narrow = ''.join(xpose_variant(index, 16, '', 8) for index in range(16))
    show(probe, '16 次宽度 8 的提交 + 1 次取回', read_lcc(20) + narrow + POP_TRF + END)

if __name__ == '__main__':
    main()
