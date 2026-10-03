"""硬件随机数状态的作用域：它是否跨 kernel 调用保留、getrngseed 是否推进状态、能否保存后恢复，以及一颗芯片的两个 TensorCore 是否共享状态。"""
import tpu_init
tpu_init.initialise_one_chip()

import jax
import numpy as np

import rng_oracle
import tpuasm_tools
from tpuasm_tools import bundle

GAP = bundle('misc: vnop') * 16

def to_tile1(instruction: str) -> str:
    """执行一条随机数指令，结果写进 tile 1。"""
    return bundle(f'va0: {instruction}') + GAP + bundle('vst: vst.8x128 [vmem:0x8], v11') + GAP

def remove_prelude(probe: tpuasm_tools.LccProbe) -> list[str]:
    """把编译器在程序开头加入的 setrngseed 与 vrng 换成 vnop，返回被替换的指令。"""
    edits = {}
    for mnemonic in ('setrngseed', 'vrng'):
        pc, = tpuasm_tools.find_bundles(probe.serialized, mnemonic, whole_program=True)
        instruction, = [item.split('#')[0].strip(' {};\n') for item in tpuasm_tools.bundle_text(probe.serialized, pc).split(';') if mnemonic in item]
        edits[pc] = (instruction, 'misc: vnop')
    probe.serialized = tpuasm_tools.edit_bundles(probe.serialized, edits)
    return [f'bundle {pc}：{old}' for pc, (old, _) in sorted(edits.items())]

def main() -> None:
    probe = tpuasm_tools.LccProbe()
    print('## 编译器加入的随机数前导')
    print('  ' + '；'.join(remove_prelude(probe)))
    programs = {
        '装入并生成': bundle('va0: setrngseed v10') + GAP + to_tile1('vrng.8x128.u32 v11'),
        '读状态': to_tile1('getrngseed v11'),
        '生成': to_tile1('vrng.8x128.u32 v11'),
    }
    s0, s1 = rng_oracle.state_from_tile(probe.host[:8])
    print('## 去掉前导后，依次调用不同的 executable')
    checkpoint = None
    for step, name in enumerate(('装入并生成', '读状态', '生成', '读状态', '生成', '读状态')):
        got = probe.run_tiles(programs[name])[0, 0, 0]
        if name == '读状态':
            want = rng_oracle.state_view(s0, s1)
            # 保存第 4 次调用读出的状态（第 2 次生成之后）。
            checkpoint = got if step == 3 else checkpoint
        else:
            want, s0, s1 = rng_oracle.vrng(s0, s1)
            if step == 4:
                second_draw = got
        print(f'  第 {step + 1} 次：{name}，与模型一致 {bool(np.array_equal(got, want))}')
    # 恢复：把第 4 次读出的状态作为输入装回，再生成一次，应当重现第 5 次生成的结果。
    host = probe.host.copy()
    host[:8] = checkpoint
    probe.x = jax.device_put(host, jax.local_devices()[0])
    restored = probe.run_tiles(programs['装入并生成'])[0, 0, 0]
    print(f'  第 7 次：装入第 4 次读出的状态并生成，与第 5 次的结果相同 {bool(np.array_equal(restored, second_draw))}')
    print('## 保留编译器前导：前导用 runtime 写在 [smem:0x3ffe0] 的值计算状态')
    plain = tpuasm_tools.LccProbe()
    # 把 [smem:0x3ffe0] 读进 s20，作为第 0 个读数的低半部分返回（高半部分 s25 置 0）。
    seeds = plain.run_raw(bundle('s1: sld s20, [smem:0x3ffe0]') + bundle('s0: simm.s32 s25, 0'), 1, repeats=4)[:, 0, 0]
    print('  连续 4 次调用中 [smem:0x3ffe0] 的值：' + '、'.join(f'0x{int(value):08x}' for value in seeds))
    states = [plain.run_tiles(programs['读状态'])[0, 0, 0] for _ in range(2)]
    print(f'  连续两次调用读出的状态相同：{bool(np.array_equal(states[0], states[1]))}')
    pair_plain = tpuasm_tools.LccProbe(num_cores=2)
    states = pair_plain.run_tiles(programs['读状态'])[0, :, 0]
    print(f'  同一次调用中两个 TensorCore 读出的状态相同：{bool(np.array_equal(states[0], states[1]))}')
    print('## 一颗芯片的两个 TensorCore')
    pair = tpuasm_tools.LccProbe(num_cores=2)
    remove_prelude(pair)
    same = pair.run_tiles(bundle('va0: setrngseed v10') + GAP + to_tile1('vrng.8x128.u32 v11'))[0, :, 0]
    print(f'  装入相同状态：两个 TensorCore 的输出相同 {bool(np.array_equal(same[0], same[1]))}')
    # 用 SMEM 中的本核编号（[smem:0x1]）与输入异或，两个 TensorCore 装入不同的状态。
    differ = bundle('s1: sld s29, [smem:0x1]') + bundle('s0: sfence') + bundle('va0: vmov.8x128 v15, s29') + GAP + bundle('va0: vxor.8x128.u32 v16, v10, v15') + GAP
    differ += bundle('va0: setrngseed v16') + GAP + to_tile1('vrng.8x128.u32 v11')
    tiles = pair.run_tiles(differ)[0, :, 0]
    for core in range(2):
        want, _, _ = rng_oracle.vrng(*rng_oracle.state_from_tile(pair.host[:8] ^ np.uint32(core)))
        print(f'  TensorCore {core} 装入“输入 ^ {core}”：与模型一致 {bool(np.array_equal(tiles[core], want))}')

if __name__ == '__main__':
    main()
