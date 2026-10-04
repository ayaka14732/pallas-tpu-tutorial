"""vmax.index.xlane 的语义与时序：并列时取哪个下标、NaN 怎样处理、从提交到取回多少周期；再用 LCC 实测 argmax、lax.top_k 与手写 top-8 的计算部分各要多少周期，并与发射模型比较。"""
import tpu_init
tpu_init.initialise_one_chip()

from pathlib import Path
import sys

import jax
import numpy as np

from top_k_common import argmax, run, top_k_by_hand
import tpuasm_tools
from tpuasm_tools import bundle, read_lcc

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'chapter03' / '05_issue_model'))
import issue_model

END = read_lcc(21) + bundle('s0: sfence') + read_lcc(22)
# v2 = lane 号，v4 = f32(lane & 15)，v5 = f32(lane)，v6 = 0。
LANES = bundle('va0: vlaneseq.8x128.u32 v1') + bundle('va0: vand.8x128.u32 v2, 0x7f, v1') + bundle('va0: vand.8x128.u32 v3, 0xf, v2') + bundle('va0: vcvt.8x128.s32.f32 v4, v3') + bundle('va0: vcvt.8x128.s32.f32 v5, v2') + bundle('va0: vimm.8x128.s32 v6, 0')

def at_lanes(lanes: tuple[int, ...], value: str, rest: str) -> str:
    """v13 在 lanes 处为 value，其余与 rest 相同。"""
    text = bundle(f'va0: vmov.8x128 v13, {rest}')
    for lane in lanes:
        text += bundle(f'va0: veq.8x128.s32 vm0, v2, {lane}') + bundle(f'va0: vsel.8x128 v13, vm0, {value}, v13')
    return text

# 名称 → 构造 v13 的片段；每个 sublane 都相同。
INPUTS = {
    '全为 0': bundle('va0: vmov.8x128 v13, v6'),
    '全为 -inf': bundle('va0: vimm.8x128.s32 v13, 0xff800000'),
    'lane & 15（最大值 15 在 lane 15、31、…、127）': bundle('va0: vmov.8x128 v13, v4'),
    '−lane（最大值在 lane 0）': bundle('va1: vsub.8x128.f32 v13, 0.0, v5'),
    'lane 5 与 lane 70 为 1，其余为 0': at_lanes((5, 70), '0x3f800000', 'v6'),
    'lane 3 为 NaN，其余为 lane': at_lanes((3,), '0x7fc00000', 'v5'),
    'lane 9 为 −0，其余为 +0': at_lanes((9,), '0x80000000', 'v6'),
}
# 名称 → 发射一次的 XLU 指令与它的结果队列
LATENCY = {
    'vmax.xlane': ('vx0: vmax.xlane.0.8x128.f32 trf0, v10', 'trf0'),
    'vmax.index.xlane': ('vx0: vmax.index.xlane.0.8x128.f32 trf0, v10', 'trf0'),
    'vmax.index.xlane（vx1、trf1）': ('vx1: vmax.index.xlane.1.8x128.f32 trf1, v10', 'trf1'),
    'vmin.xlane': ('vx0: vmin.xlane.0.8x128.f32 trf0, v10', 'trf0'),
}
KERNELS = {
    'argmax（max 与 argmax）': argmax,
    'lax.top_k(x, 8, is_stable=False)': lambda x: jax.lax.top_k(x, 8, is_stable=False),
    '手写 top-8（掩码排除已选位置）': top_k_by_hand(8),
}

def main() -> None:
    probe = tpuasm_tools.LccProbe()

    print('## vmax.index.xlane、vmax.xlane 与 vmin.index.xlane 对同一个 TC VREG 的结果')
    for name, build in INPUTS.items():
        body = LANES + build + bundle('vx0: vmax.index.xlane.0.8x128.f32 trf0, v13') + bundle('vx0: vmax.xlane.2.8x128.f32 trf0, v13') + bundle('vx0: vmin.index.xlane.0.8x128.f32 trf0, v13') + ''.join(bundle(f'vr0: vpop.8x128 v{14 + tile}, trf0') for tile in range(3)) + ''.join(bundle(f'vst: vst.8x128 [vmem:0x{8 * (tile + 1):x}], v{14 + tile}') for tile in range(3))
        index, maximum, smallest = probe.run_tiles(body)[0, 0, :3]
        assert all(np.all(tile == tile[0, 0]) for tile in (index, maximum, smallest))
        print(f'  {name}：vmax.index 下标 {int(index[0, 0])}，vmax 位型 0x{int(maximum[0, 0]):08x}，vmin.index 下标 {int(smallest[0, 0])}（各自所有 lane、所有 sublane 相同）')

    print('## 提交之后紧接取回：R2 − R0 = 13 + 延迟')
    latencies = {}
    for name, (push, queue) in LATENCY.items():
        (r1, r2), = {tuple(row) for row in probe.run(read_lcc(20) + bundle(push) + bundle(f'vr0: vpop.8x128 v11, {queue}') + END).tolist()}
        latencies[name] = r2 - 13
        print(f'  {name}：R2 − R0 = {r2}，延迟 {r2 - 13}')

    print('## 计算部分的周期数：LCC 实测与发射模型')
    x = np.stack([np.random.default_rng(0).permutation(128) for _ in range(8)]).astype(np.float32)
    for name, f in KERNELS.items():
        _, listing = run(f, x)
        section = tpuasm_tools.compute_section(listing)
        xlu = sum(text.count('xlane') for text in section)
        _, r2 = probe.time_section(section)
        reads = issue_model.replay(issue_model.parse(tpuasm_tools.section_program(section)))
        print(f'  {name}：{len(section)} 个 bundle，{xlu} 次 XLU 归约；实测 R2 − R0 = {r2}，模型 {reads[22] - reads[20]}')

if __name__ == '__main__':
    main()
