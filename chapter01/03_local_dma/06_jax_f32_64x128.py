"""原生 XLA 在单个 TensorCore（split-chip）上把 f32[64,128] 乘以 2，观察 XLA 自己生成的 DMA。"""
import tpu_init
tpu_init.initialize_one_core()

import jax
import jax.numpy as jnp
import numpy as np

import tpuasm_tools

def main() -> None:
    x = jnp.arange(64 * 128, dtype=jnp.float32).reshape(64, 128)
    compiled = tpuasm_tools.compile(lambda x: x * 2.0, x)
    np.testing.assert_array_equal(np.asarray(compiled(x)), np.asarray(x) * 2.0)
    print('数值检查通过')
    listing = tpuasm_tools.kernel_listing(compiled)
    print('\n# fusion 段的指令统计')
    tpuasm_tools.print_mnemonic_counts(listing)
    print('\n# fusion 段')
    print(listing)

if __name__ == '__main__':
    main()
