"""原生 XLA 在单芯片上把 f32[8,128] 乘以 2；打印完整清单的结构概览和 fusion 段，与 Pallas 版本对照。"""
import tpu_init
tpu_init.initialise_one_chip()

import jax
import jax.numpy as jnp
import numpy as np

import tpuasm_tools

def main() -> None:
    # 原生 XLA 不做任何 SPMD 设置，只由 runtime 决定使用一颗芯片。
    x = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
    compiled = tpuasm_tools.compile(lambda x: x * 2.0, x)
    np.testing.assert_array_equal(np.asarray(compiled(x)), np.asarray(x) * 2.0)
    print('数值检查通过')
    print('\n# 完整清单的结构概览')
    print(tpuasm_tools.listing_outline(compiled))
    listing = tpuasm_tools.kernel_listing(compiled)
    print('\n# fusion 段的指令统计')
    tpuasm_tools.print_mnemonic_counts(listing)
    print('\n# fusion 段')
    print(listing)

if __name__ == '__main__':
    main()
