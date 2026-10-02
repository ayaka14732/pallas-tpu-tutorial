"""原生 XLA 的同一运算 y = 2x + 1，f32[32768,128]：统计 XLA 生成的 DMA 与每个 TensorCore 的工作量。"""
import tpu_init
tpu_init.initialise_one_chip()

import importlib

import jax.numpy as jnp
import numpy as np

import tpuasm_tools

pipeline = importlib.import_module('01_pallas_double_buffer')

def main() -> None:
    x = jnp.arange(pipeline.ROWS * 128, dtype=jnp.float32).reshape(pipeline.ROWS, 128) / 1024
    function = lambda x: x * 2.0 + 1.0
    compiled = tpuasm_tools.compile(function, x)
    np.testing.assert_array_equal(np.asarray(compiled(x)), np.asarray(x) * 2.0 + 1.0)
    listing = tpuasm_tools.kernel_listing(compiled)
    counts = tpuasm_tools.count_mnemonics(listing)
    entries = [line[len('# entry bundle: '):] for line in listing.splitlines() if line.startswith('# entry bundle')]
    print(f'数值检查通过；HLO 段：{"；".join(entries)}')
    print(f'vld {counts["vld.8x128"]}、vmul {counts["vmul.8x128.f32"]}、vadd {counts["vadd.8x128.f32"]}、vst {counts["vst.8x128"]}，sbr.rel {counts["sbr.rel"]}')
    print('DMA：' + '、'.join(f'{mnemonic}×{count}' for mnemonic, count in sorted(counts.items()) if mnemonic.startswith('dma')) + f'；vwait.ge×{counts["vwait.ge"]}')
    print('\n'.join(line.split('#')[0].strip() for line in listing.splitlines() if 'dma.' in line))

if __name__ == '__main__':
    main()
