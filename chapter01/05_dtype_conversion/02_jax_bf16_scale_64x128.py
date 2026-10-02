"""原生 XLA 在单芯片上把 bf16[64,128] 乘以 2，观察 XLA 怎样把 f32 结果打包写回。"""
import tpu_init
tpu_init.initialise_one_chip()

import jax
import jax.numpy as jnp
import numpy as np

import tpuasm_tools

def main() -> None:
    x = (jnp.arange(64 * 128) % 100).astype(jnp.bfloat16).reshape(64, 128)
    compiled = tpuasm_tools.compile(lambda x: x * 2, x)
    np.testing.assert_array_equal(np.asarray(compiled(x)), np.asarray(x) * 2)
    print('数值检查通过')
    listing = tpuasm_tools.kernel_listing(compiled)
    print('\n# fusion 段的指令统计')
    tpuasm_tools.print_mnemonic_counts(listing)
    print('\n# 向量指令')
    print('\n'.join(line for line in listing.splitlines() if any(f'{slot}: ' in line for slot in ('vld', 'vst', 'va0', 'va1'))))

if __name__ == '__main__':
    main()
