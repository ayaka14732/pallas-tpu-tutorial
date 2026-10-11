"""向量 ALU 指令的结果要几个周期才可用：把 16 条同样的指令串成一条链，每一条用上一条的结果；结果下一个周期可用时 R2 − R0 = 16 + 13，要 2 个周期时是 2 × 16 + 11。"""
import tpu_init
tpu_init.initialize_one_chip()

import tpuasm_tools
from tpuasm_tools import LCC_END, bundle, read_lcc, show_lcc

COUNT = 16
# 每项是一条链的循环体：一条或两条指令，各占一个 bundle，最后一条写出的 v11 是下一轮第一条的源操作数。
CHAINS = {
    'vadd.8x128.f32': ('va1: vadd.8x128.f32 v11, v11, v10',),
    'vsub.8x128.f32': ('va1: vsub.8x128.f32 v11, v11, v10',),
    'vmul.8x128.f32': ('va0: vmul.8x128.f32 v11, v11, v10',),
    'vmax.8x128.f32': ('va0: vmax.8x128.f32 v11, v11, v10',),
    'vmin.8x128.f32': ('va0: vmin.8x128.f32 v11, v11, v10',),
    'vadd.8x128.s32': ('va0: vadd.8x128.s32 v11, v11, v10',),
    'vsub.8x128.s32': ('va0: vsub.8x128.s32 v11, v11, v10',),
    'vand.8x128.u32': ('va0: vand.8x128.u32 v11, v11, v10',),
    'vor.8x128.u32': ('va0: vor.8x128.u32 v11, v11, v10',),
    'vxor.8x128.u32': ('va0: vxor.8x128.u32 v11, v11, v10',),
    'vshll.8x128.s32': ('va1: vshll.8x128.s32 v11, v11, 0x1',),
    'vshrl.8x128.s32': ('va1: vshrl.8x128.s32 v11, v11, 0x1',),
    'vshra.8x128.s32': ('va1: vshra.8x128.s32 v11, v11, 0x1',),
    'vpcnt': ('va0: vpcnt v11, v11',),
    'vclz': ('va0: vclz v11, v11',),
    'vmov.8x128': ('va0: vmov.8x128 v11, v11',),
    'vsel.8x128（掩码不变）': ('va0: vsel.8x128 v11, vm0, v11, v10',),
    'vcvt.8x128.f32.s32': ('va0: vcvt.8x128.f32.s32 v11, v11, 0xffffffff',),
    'vcvt.8x128.s32.f32': ('va0: vcvt.8x128.s32.f32 v11, v11',),
    'vunpackl.8x128.f16.f32': ('va0: vunpackl.8x128.f16.f32 v11, v11',),
    'vpackc.8x128.f32.f16': ('va0: vpackc.8x128.f32.f16 v11, v11, v11',),
    'vrot.slane.down.8x128.u32': ('va0: vrot.slane.down.8x128.u32 v11, v11',),
    'vgt.8x128.f32 → vsel（两条一轮）': ('va0: vgt.8x128.f32 vm0, v11, v10', 'va0: vsel.8x128 v11, vm0, v11, v10'),
    'vgt.8x128.s32 → vsel（两条一轮）': ('va0: vgt.8x128.s32 vm0, v11, v10', 'va0: vsel.8x128 v11, vm0, v11, v10'),
    'vld → vadd.s32（两条一轮，读别的 tile）': ('vld: vld.8x128 v12, [vmem:0x88]', 'va0: vadd.8x128.s32 v11, v12, v11'),
    'vadd.f32 → vmax.f32（两条一轮）': ('va1: vadd.8x128.f32 v12, v11, v10', 'va0: vmax.8x128.f32 v11, v12, v10'),
}

def main() -> None:
    probe = tpuasm_tools.LccProbe()
    print(f'## {COUNT} 轮相互依赖的链')
    for name, loop in CHAINS.items():
        body = read_lcc(20) + ''.join(bundle(text) for text in loop) * COUNT + LCC_END
        (_, total), = {tuple(row) for row in probe.run(body).tolist()}
        print(f'  {name}：{len(loop) * COUNT} 个 bundle，R2 − R0 = {total}（互不等待时是 {len(loop) * COUNT + 13}）')
    print('## 指令相同、先后不同：8 条互不依赖的 vadd.f32 和写出各自结果的 8 条 vst')
    adds = [bundle(f'va1: vadd.8x128.f32 v{12 + index}, v10, v10') for index in range(8)]
    stores = [bundle(f'vst: vst.8x128 [vmem:0x{0x80 + 8 * index:x}], v{12 + index}') for index in range(8)]
    show_lcc(probe, '每条 vadd 后面紧跟它的 vst', read_lcc(20) + ''.join(add + store for add, store in zip(adds, stores)) + LCC_END)
    show_lcc(probe, '先 8 条 vadd，再 8 条 vst', read_lcc(20) + ''.join(adds) + ''.join(stores) + LCC_END)

if __name__ == '__main__':
    main()
