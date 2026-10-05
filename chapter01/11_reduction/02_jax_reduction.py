"""原生 XLA 的归约对照：f32[8,128] 沿 lane、沿 sublane、全部求和，以及 f32[64,128] 沿 sublane 求和。"""
import tpu_init
tpu_init.initialize_one_core()

import jax
import jax.numpy as jnp
import numpy as np

import reduction_common
import tpuasm_tools

def main() -> None:
    rng = np.random.default_rng(0)
    small = rng.integers(-100, 100, (8, 128)).astype(np.float32)
    tall = rng.integers(-100, 100, (64, 128)).astype(np.float32)
    cases = (
        ('sum(x, axis=1)，f32[8,128]', lambda x: jnp.sum(x, axis=1, keepdims=True), small),
        ('sum(x, axis=0)，f32[8,128]', lambda x: jnp.sum(x, axis=0, keepdims=True), small),
        ('max(x, axis=0)，f32[8,128]', lambda x: jnp.max(x, axis=0, keepdims=True), small),
        ('sum(x)，f32[8,128]', lambda x: jnp.sum(x, keepdims=True), small),
        ('sum(x, axis=0)，f32[64,128]', lambda x: jnp.sum(x, axis=0, keepdims=True), tall),
    )
    for name, f, x in cases:
        compiled = tpuasm_tools.compile(f, x)
        result = np.asarray(compiled(x))
        listing = tpuasm_tools.kernel_listing(compiled)
        entries = [line[len('# entry bundle: '):] for line in listing.splitlines() if line.startswith('# entry bundle')]
        print(f'## {name}：与 CPU 结果一致 {bool(np.array_equal(result, np.asarray(jax.jit(f, backend="cpu")(x))))}；HLO 段：{"；".join(entries)}')
        print(f'  {reduction_common.describe(listing)}')
        print()

if __name__ == '__main__':
    main()
