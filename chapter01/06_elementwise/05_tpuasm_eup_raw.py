"""EUP 五条指令本身的精度：用 tpuasm 直接执行 vrcp、vrsqrt、vlog2、vpow2、vtanh，不加编译器的任何修正，与 float64 参考比较。"""
import tpu_init
tpu_init.initialise_one_chip()

import numpy as np

import tpuasm_tools
from tpuasm_tools import bundle

# v13 = y：随机 u32 取低 23 位作尾数，拼成 [1, 2) 中的 f32；v15 = x = (y − 1.5) × 8，落在 [−4, 4)。
INPUTS = (
    bundle('va0: vand.8x128.u32 v13, 0x7fffff, v10')
    + bundle('va0: vor.8x128.u32 v13, 0x3f800000, v13')
    + bundle('va1: vadd.8x128.f32 v14, -1.5, v13')
    + bundle('va0: vmul.8x128.f32 v15, 8.0, v14')
    + bundle('misc: vnop') * 4
)
# 名称 → (EUP 指令, 输入寄存器, float64 参考)
FUNCTIONS = {
    '1 / y': ('vrcp.8x128.f32', 'v13', lambda x, y: 1.0 / y),
    'rsqrt(y)': ('vrsqrt.8x128.f32', 'v13', lambda x, y: 1.0 / np.sqrt(y)),
    'log2(y)': ('vlog2.8x128.f32', 'v13', lambda x, y: np.log2(y)),
    'exp2(x)': ('vpow2.8x128.f32', 'v15', lambda x, y: np.exp2(x)),
    'tanh(x)': ('vtanh.8x128.f32', 'v15', lambda x, y: np.tanh(x)),
}

def main() -> None:
    probe = tpuasm_tools.LccProbe()
    body = INPUTS
    for tile, (instruction, register, _) in enumerate(FUNCTIONS.values(), 1):
        body += bundle(f'va0: {instruction} erf, {register}') + bundle('vr0: vpop.8x128 v16, erf') + bundle(f'vst: vst.8x128 [vmem:0x{tile * 8:x}], v16')
    body += bundle('vst: vst.8x128 [vmem:0x30], v13') + bundle('vst: vst.8x128 [vmem:0x38], v15')
    tiles = probe.run_tiles(body)[0, 0].view(np.float32)
    y, x = tiles[5].astype(np.float64), tiles[6].astype(np.float64)
    print(f'输入：y 在 [{y.min():.4f}, {y.max():.4f}]，x 在 [{x.min():.4f}, {x.max():.4f}]，各 1024 个')
    for tile, (name, (instruction, _, reference)) in enumerate(FUNCTIONS.items()):
        result = tiles[tile].astype(np.float64)
        expected = reference(x, y)
        relative = np.max(np.abs(result - expected) / np.maximum(np.abs(expected), np.finfo(np.float32).tiny))
        ulp = np.max(np.abs(result - expected) / np.spacing(np.abs(expected).astype(np.float32)).astype(np.float64))
        print(f'{name}（{instruction}）：最大绝对误差 {np.max(np.abs(result - expected)):.1e}，最大相对误差 {relative:.1e}（约 {-np.log2(relative):.1f} 位），最大 {ulp:.0f} ULP')

if __name__ == '__main__':
    main()
