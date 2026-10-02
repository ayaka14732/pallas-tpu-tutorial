"""原生 XLA 的矩阵乘法对照：bf16[16,128] @ bf16[128,128]、RHS 以 (N,K) 存放、f32 输入、bf16[128,128] @ bf16[128,128]。"""
import tpu_init
tpu_init.initialise_one_chip()

import jax
import jax.numpy as jnp
import numpy as np

import mxu_common
import tpuasm_tools

def main() -> None:
    rng = np.random.default_rng(0)
    cases = (
        ('bf16[16,128] @ bf16[128,128]', (16, 128), (128, 128), jnp.bfloat16, lambda a, b: jnp.dot(a, b, preferred_element_type=jnp.float32)),
        ('bf16[16,128] @ bf16[128,128]ᵀ（RHS 以 (N,K) 存放）', (16, 128), (128, 128), jnp.bfloat16, lambda a, b: jnp.dot(a, b.T, preferred_element_type=jnp.float32)),
        ('f32[16,128] @ f32[128,128]', (16, 128), (128, 128), jnp.float32, lambda a, b: jnp.dot(a, b)),
        ('bf16[128,128] @ bf16[128,128]', (128, 128), (128, 128), jnp.bfloat16, lambda a, b: jnp.dot(a, b, preferred_element_type=jnp.float32)),
    )
    for name, lhs_shape, rhs_shape, dtype, f in cases:
        lhs = jnp.asarray(rng.integers(-8, 8, lhs_shape)).astype(dtype)
        rhs = jnp.asarray(rng.integers(-8, 8, rhs_shape)).astype(dtype)
        compiled = tpuasm_tools.compile(f, lhs, rhs)
        result = np.asarray(compiled(lhs, rhs))
        expected = np.asarray(f(np.asarray(lhs, np.float64), np.asarray(rhs, np.float64)))
        listing = tpuasm_tools.kernel_listing(compiled)
        entries = [line[len('# entry bundle: '):] for line in listing.splitlines() if line.startswith('# entry bundle')]
        print(f'## {name}：与精确结果逐元素一致 {bool(np.array_equal(result, expected))}；HLO 段：{"；".join(entries)}')
        print(mxu_common.summary(listing))
        print()

if __name__ == '__main__':
    main()
