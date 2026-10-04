"""原生 XLA 的拼接对照：与 Pallas 版本相同的四组 shape 与轴。"""
import tpu_init
tpu_init.initialise_one_core()

import jax.numpy as jnp
import numpy as np

import tpuasm_tools

SKIPPED = ('vsync', 'vwait', 'vtrace', 'cfence', 's', 'p', 'dma')

def main() -> None:
    for shape, axis in (((8, 128), 0), ((8, 128), 1), ((9, 128), 0), ((8, 129), 1)):
        a = jnp.arange(shape[0] * shape[1], dtype=jnp.float32).reshape(shape)
        b = -a - 1
        compiled = tpuasm_tools.compile(lambda a, b: jnp.concatenate([a, b], axis=axis), a, b)
        np.testing.assert_array_equal(np.asarray(compiled(a, b)), np.concatenate([np.asarray(a), np.asarray(b)], axis=axis))
        listing = tpuasm_tools.kernel_listing(compiled)
        counts = tpuasm_tools.count_mnemonics(listing)
        entries = [line[len('# entry bundle: '):] for line in listing.splitlines() if line.startswith('# entry bundle')]
        bundles = sum(line.startswith('{') for line in listing.splitlines())
        vector = '、'.join(f'{mnemonic}×{count}' for mnemonic, count in sorted(counts.items()) if mnemonic.startswith('v') and not mnemonic.startswith(SKIPPED[:3]))
        dmas = '、'.join(f'{mnemonic}×{count}' for mnemonic, count in sorted(counts.items()) if mnemonic.startswith('dma'))
        print(f'## f32{list(shape)} × 2，axis={axis}：数值检查通过；HLO 段共 {bundles} 个 bundle：{"；".join(entries)}')
        print(f'  向量指令：{vector or "无"}；DMA：{dmas or "无"}')
        print()

if __name__ == '__main__':
    main()
