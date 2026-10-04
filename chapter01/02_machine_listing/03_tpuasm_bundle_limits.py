"""一个 bundle 能装下什么：用 tpuasm 汇编几个手写的 bundle，看立即数、向量槽读取的标量寄存器和谓词寄存器各有什么限制。只做汇编，不需要 TPU。"""
from tpuasm import assemble_listing

import tpuasm_tools

ADD = 'vadd.8x128.s32'
# (说明, bundle 的文本)
CASES = (
    ('3 个不同的 32 位立即数', f'va0: {ADD} v1, 0x11111111, v0 ; va1: {ADD} v2, 0x22222222, v0 ; s0: simm.s32 s1, 0x33333333'),
    ('4 个不同的 32 位立即数', f'va0: {ADD} v1, 0x11111111, v0 ; va1: {ADD} v2, 0x22222222, v0 ; s0: simm.s32 s1, 0x33333333 ; s1: simm.s32 s2, 0x44444444'),
    ('4 个相同的 32 位立即数', f'va0: {ADD} v1, 0x11111111, v0 ; va1: {ADD} v2, 0x11111111, v0 ; s0: simm.s32 s1, 0x11111111 ; s1: simm.s32 s2, 0x11111111'),
    ('2 个 32 位立即数加 2 个 16 位以内的立即数', f'va0: {ADD} v1, 0x11111111, v0 ; va1: {ADD} v2, 0x22222222, v0 ; s0: simm.s32 s1, 0x1111 ; s1: simm.s32 s2, 0x2222'),
    ('向量槽读 3 个标量寄存器', f'va0: {ADD} v1, s3, v0 ; va1: {ADD} v2, s4, v0 ; vst: vst.8x128 [vmem:s5], v1'),
    ('向量槽读 4 个标量寄存器', f'va0: {ADD} v1, s3, v0 ; va1: {ADD} v2, s4, v0 ; vst: vst.8x128 [vmem:s5], v1 ; vld: vld.8x128 v3, [vmem:s6]'),
    ('谓词 @p14', 's0: @p14 simm.s32 s1, 1'),
    ('谓词 @!p14', 's0: @!p14 simm.s32 s1, 1'),
    ('谓词 @p15', 's0: @p15 simm.s32 s1, 1'),
)

def main() -> None:
    for name, text in CASES:
        try:
            image = assemble_listing(f'.target {tpuasm_tools.TARGET}\n{{ {text} }}\n.empty 9\n')
            print(f'{name}：可以汇编；程序映像 {len(image)} 字节（10 个 bundle）')
        except ValueError as error:
            print(f'{name}：不能汇编：{str(error).splitlines()[0].split(": ", 2)[-1].split(";")[0]}')
        print(f'  {{ {text} }}')

if __name__ == '__main__':
    main()
