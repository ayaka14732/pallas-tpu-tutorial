"""f32 → int32 的 vcvt 有第三个操作数，Mosaic 总是写 0xffffffff。用 tpuasm 改变这个操作数，确定它的含义：小数部分大于它时向远离零的方向进 1。"""
import tpu_init
tpu_init.initialise_one_chip()

import numpy as np

import tpuasm_tools
from tpuasm_tools import bundle

VALUES = (0.25, 0.5, 0.75, -0.25, -0.5, -0.75, 2.5, -2.5)
THRESHOLDS = (0xFFFFFFFF, 0xC0000000, 0xBFFFFFFF, 0x80000000, 0x7FFFFFFF, 0x40000000, 0x3FFFFFFF, 0x0)

def as_bits(value: float) -> int:
    return int(np.float32(value).view(np.uint32))

def fraction(value: float) -> int:
    """|value| 的小数部分，记为 32 位定点数。"""
    return round(abs(value) % 1 * 2**32)

def convert_tiles(probe: tpuasm_tools.LccProbe, values: tuple[float, ...], operand: str) -> np.ndarray:
    """第 i 个 tile：把全为 values[i] 的 TC VREG 用 vcvt 转为 int32，第三个操作数为 operand。返回 (len(values), 8, 128) 的 int32。"""
    body = ''
    for tile, value in enumerate(values, 1):
        body += bundle(f'va0: vimm.8x128.s32 v13, 0x{as_bits(value):x}') + bundle(f'va0: vcvt.8x128.f32.s32 v14, v13, {operand}') + bundle(f'vst: vst.8x128 [vmem:0x{tile * 8:x}], v14')
    return probe.run_tiles(body)[0, 0, :len(values)].view(np.int32)

def main() -> None:
    probe = tpuasm_tools.LccProbe()

    print('## 第三个操作数为立即数 T：每个值转换后的结果')
    print('T            ' + ''.join(f'{value:>7}' for value in VALUES))
    agree = True
    for threshold in THRESHOLDS:
        tiles = convert_tiles(probe, VALUES, f'0x{threshold:x}')
        assert np.all(tiles == tiles[:, :1, :1])
        results = [int(tile[0, 0]) for tile in tiles]
        print(f'0x{threshold:08x}   ' + ''.join(f'{result:>7}' for result in results))
        # 规则：|x| 的小数部分记为 32 位定点数 F（0.25 → 0x40000000），F > T 时向远离零的方向进 1，否则向零截断。
        agree &= results == [int(np.trunc(value)) + int(np.sign(value)) * int(fraction(value) > threshold) for value in VALUES]
    print(f'规则“小数部分 F > T 时远离零进 1”与以上 64 个结果全部一致：{agree}')

    print('## 第三个操作数为 TC VREG：每个元素用各自的 T')
    thresholds = probe.host[:8].astype(np.uint64)  # 片段开始前已读进 v10 的输入
    values = (2.25, -2.25, 0.75, 0.5, 1.125, -7.875)
    tiles = convert_tiles(probe, values, 'v10')
    for value, tile in zip(values, tiles):
        expected = int(np.trunc(value)) + int(np.sign(value)) * (fraction(value) > thresholds).astype(np.int64)
        rounded = np.mean(tile != int(np.trunc(value)))
        print(f'  {value:>7}：1024 个元素中 {np.sum(tile != int(np.trunc(value)))} 个进 1（比例 {rounded:.3f}，小数部分 {abs(value) % 1}）；逐元素与规则一致：{bool(np.array_equal(tile, expected))}')

if __name__ == '__main__':
    main()
