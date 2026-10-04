"""用发射模型预测第 3、4 节和本节各段手写清单的 LCC 读数，再在真机上运行同样的清单，逐一比较。"""
import tpu_init
tpu_init.initialise_one_chip()

import tpuasm_tools
from tpuasm_tools import bundle, read_lcc
import issue_model

CMEM_SETUP = bundle('s0: simm.s32 s23, 0') + bundle('s0: dma.simple [cmem:s23], [vmem:s23], length=64, dst_flag=[sflag:52]') + bundle('misc: vwait.ge [sflag:52], 64') + bundle('misc: vsyncadd.s32 [sflag:52], -64')
MXU_SETUP = bundle('vx0: vmatpush.packed.8x128.f16 gsfn0, v10') * 8 + bundle('vx0: vdwg.128x128.f16 gmr0, gsfn0')
END = read_lcc(21) + bundle('s0: sfence') + read_lcc(22)

def xpose(index: int, unit: int) -> str:
    """一次 128×128 f32 转置的第 index 次提交；unit 为 0 或 1，对应 trf0、trf1。"""
    suffix = '.start' if index == 0 else '.end' if index == 15 else ''
    return f'vx{unit}: vxpose.{index % 2 * 2 + unit}{suffix}.8x128 trf{unit}, v10, 128'

def transpose(unit: int) -> str:
    return ''.join(bundle(xpose(index, unit)) for index in range(16))

def loop(count: int) -> tuple[str, str]:
    """计数循环：清单中的循环体，以及按 count 次迭代展开的执行轨迹。"""
    body = bundle('s0: sadd.s32 s24, -1, s24') + bundle('s0: sne.s32 p1, s24, 0') + bundle('s0: @p1 sbr.rel probe_loop') + bundle()
    start = bundle(f's0: simm.s32 s24, {count}')
    return start + 'probe_loop:\n' + body, start + body * count

def cases() -> list[tuple[str, str, str, str]]:
    """(名称, setup, 清单, 模型使用的执行轨迹)；after 是读数之后才执行、不计入区间的部分。"""
    result = []
    def add(name: str, body: str, setup: str = '', trace: str | None = None, after: str = '') -> None:
        program = read_lcc(20) + body + END
        result.append((name, setup, program + after, program if trace is None else read_lcc(20) + trace + END))
    for count in (0, 16, 64):
        add(f'第 3 节：{count} 条独立 vadd', bundle('va0: vadd.8x128.s32 v11, 1, v10') * count)
        add(f'第 3 节：{count} 条 vld', bundle('vld: vld.8x128 v11, [vmem:0x0]') * count)
    for count in (4, 16, 32, 64):
        add(f'第 4 节：{count} 条相互依赖的 vmul', bundle('va0: vmul.8x128.f32 v11, v11, v10') * count)
    for count in (1, 4, 16, 32, 64):
        add(f'第 4 节：{count} 组 cld + vpop', (bundle('cld: cld.8x128 crf, [cmem:0x0]') + bundle('vr0: vpop.8x128 v11, crf')) * count, CMEM_SETUP)
    add('vpow2 → vpop', bundle('va0: vpow2.8x128.f32 erf, v10') + bundle('vr0: vpop.8x128 v11, erf'))
    for count in (8, 16):
        add(f'{count} 条 vpow2 → {count} 条 vpop', bundle('va0: vpow2.8x128.f32 erf, v10') * count + bundle('vr0: vpop.8x128 v11, erf') * count)
    add('8 组 vpow2 + vpop', (bundle('va0: vpow2.8x128.f32 erf, v10') + bundle('vr0: vpop.8x128 v11, erf')) * 8)
    matmul, pop = bundle('vx0: vmatmul.8x128.f32 mrf0, v10'), bundle('vr0: vpop.8x128 v11, mrf0')
    add('vmatmul → vpop', matmul + pop, MXU_SETUP)
    for count in (8, 11, 12, 16):
        add(f'{count} 条 vmatmul → {count} 条 vpop', matmul * count + pop * count, MXU_SETUP)
    add('16 条 vmatmul，从第 9 条起与 vpop 交错', matmul * 8 + (matmul + pop) * 8 + pop * 8, MXU_SETUP)
    add('vpush → spop', bundle('vst: vpush v2sf, v10') + bundle('s0: spop s24, v2sf'))
    add('一次转置：16 次 vxpose', transpose(0), after=bundle('vr0: vpop.8x128 v11, trf0') * 16)
    add('一次转置：16 次 vxpose + 16 次 vpop', transpose(0) + bundle('vr0: vpop.8x128 v11, trf0') * 16)
    add('先后两次转置，都用 trf0', (transpose(0) + bundle('vr0: vpop.8x128 v11, trf0') * 16) * 2)
    both = ''.join(bundle(xpose(index, 0) + ' ; ' + xpose(index, 1)) for index in range(16))
    add('两次转置分给 trf0、trf1，成对取回', both + bundle('vr0: vpop.8x128 v11, trf0 ; vr1: vpop.8x128 v12, trf1') * 16)
    add('两次转置分给 trf0、trf1，先取完 trf0', both + bundle('vr0: vpop.8x128 v11, trf0') * 16 + bundle('vr0: vpop.8x128 v12, trf1') * 16)
    load, use = bundle('s1: sld s24, [smem:0x0]'), bundle('s0: sadd.s32 s24, 1, s24')
    for count in (2, 16):
        add(f'第 3 节：{count} 条连续的 sld', load * count)
    add('第 3 节：sld 之后紧接使用其结果', load + use)
    add('第 3 节：4 组“sld + 3 条无关的 sadd”', (load + bundle('s0: sadd.s32 s23, 1, s23') * 3) * 4)
    add('sld 与依赖 sld 结果的向量运算', load + bundle('va0: vmov.8x128 v11, s24') + bundle('va0: vadd.8x128.s32 v12, v11, v10'))
    xlane, pop_trf = bundle('vx0: vadd.xlane.0.8x128.f32 trf0, v10'), bundle('vr0: vpop.8x128 v11, trf0')
    add('vadd.xlane → vpop', xlane + pop_trf)
    add('8 条 vadd.xlane → 8 条 vpop', xlane * 8 + pop_trf * 8)
    add('lane 循环移位 vrot → vpop', bundle('vx0: vrot.0.8x128 trf0, v10, s23') + pop_trf, bundle('s0: simm.s32 s23, 5'))
    add('4 次 lane 循环移位，每次移位前一次的结果', ''.join(bundle(f'vx0: vrot.0.8x128 trf0, {source}, s23') + bundle('vr0: vpop.8x128 v11, trf0') for source in ('v10', 'v11', 'v11', 'v11')), bundle('s0: simm.s32 s23, 5'))
    add('7 条相互依赖的 vrot.slane.down', bundle('va0: vrot.slane.down.8x128.u32 v11, v10') + bundle('va0: vrot.slane.down.8x128.u32 v11, v11') * 6)
    for count in (1, 16, 64):
        listing, trace = loop(count)
        add(f'计数循环 {count} 次', listing, trace=trace)
    return result

def main() -> None:
    probe = tpuasm_tools.LccProbe()
    matches = 0
    all_cases = cases()
    for name, setup, program, trace in all_cases:
        measured = sorted({tuple(row) for row in probe.run(program, setup=setup).tolist()})
        reads = issue_model.replay(issue_model.parse(trace))
        predicted = (reads[21] - reads[20], reads[22] - reads[20])
        matches += measured == [predicted]
        print(f'{name}：模型 {predicted}，实测 {measured}')
    print(f'{matches} / {len(all_cases)} 段清单与模型一致')

if __name__ == '__main__':
    main()
