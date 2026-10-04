"""沿 sublane 的前缀和逐行串行计算：每次把一行广播到 8 个 sublane 再累加。比较 jnp.broadcast_to 与 stride=0 的 Ref 读取。"""
import tpu_init
tpu_init.initialise_one_chip()

import numpy as np

import scan_common
import tpuasm_tools

def main() -> None:
    x = np.random.default_rng(0).integers(-100, 100, (8, 128)).astype(np.float32)
    for style in ('broadcast_to', 'stride=0'):
        result, listing = scan_common.run_row_serial(style, x)
        np.testing.assert_array_equal(result, np.cumsum(x, axis=0))
        counts = tpuasm_tools.count_mnemonics(listing)
        print(f'## {style}：数值检查通过；vld {counts["vld.8x128"]} 条，vadd {counts["vadd.8x128.f32"]} 条，vst {counts["vst.8x128"]} 条')
        print('\n'.join(line.split('#')[0].rstrip() for line in listing.splitlines() if any(key in line for key in ('vld:', 'vst:', 'va1:'))))
        print()

if __name__ == '__main__':
    main()
