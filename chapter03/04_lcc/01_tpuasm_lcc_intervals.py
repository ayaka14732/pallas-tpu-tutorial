"""用 LCC 读数测量几种指令序列：N 个空 bundle、N 条相互依赖的标量加法、N 条独立的向量加法、N 条 vld。每种都给出 R1 − R0（紧接其后的读数）与 R2 − R0（中间隔一条 sfence）。"""
import tpu_init
tpu_init.initialise_one_chip()

import tpuasm_tools
from tpuasm_tools import bundle, read_lcc

SEQUENCES = {
    '空 bundle': '',
    '标量加法链 s24 += 1': 's0: sadd.s32 s24, 1, s24',
    '独立的向量加法': 'va0: vadd.8x128.s32 v11, 1, v10',
    'vld': 'vld: vld.8x128 v11, [vmem:0x0]',
}

def main() -> None:
    probe = tpuasm_tools.LccProbe()
    for name, instruction in SEQUENCES.items():
        print(f'## {name}')
        for count in (0, 1, 4, 16, 64):
            body = read_lcc(20) + bundle(instruction) * count + read_lcc(21) + bundle('s0: sfence') + read_lcc(22)
            deltas = probe.run(body)
            values = sorted({tuple(row) for row in deltas.tolist()})
            print(f'  N = {count:2d}：(R1 − R0, R2 − R0) = {values}')

if __name__ == '__main__':
    main()
