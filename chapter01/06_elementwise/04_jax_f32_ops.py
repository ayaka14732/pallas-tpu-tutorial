"""原生 XLA 的同一批 f32 逐元素运算：对照 Pallas 版本，看 XLA 选用的计算指令与误差。"""
import tpu_init
tpu_init.initialise_one_core()

import jax
import jax.numpy as jnp
import numpy as np

import tpuasm_tools

OPS = (
    ('x / y', lambda x, y: x / y, lambda x, y: x / y),
    ('exp(x)', lambda x, y: jnp.exp(x), lambda x, y: np.exp(x)),
    ('log(y)', lambda x, y: jnp.log(y), lambda x, y: np.log(y)),
    ('tanh(x)', lambda x, y: jnp.tanh(x), lambda x, y: np.tanh(x)),
    ('sin(x)', lambda x, y: jnp.sin(x), lambda x, y: np.sin(x)),
    ('x * y + 1', lambda x, y: x * y + 1, lambda x, y: x * y + 1),
)
SKIPPED = ('vld', 'vst', 'vsync', 'vwait', 'vtrace', 'cfence', 'dma.', 's', 'p')

def ulp_error(result: np.ndarray, expected: np.ndarray) -> int:
    target = expected.astype(np.float32)
    return int(np.max(np.abs(result.view(np.int32).astype(np.int64) - target.view(np.int32).astype(np.int64))))

def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(-4, 4, (8, 128)).astype(np.float32)
    y = rng.uniform(0.25, 4, (8, 128)).astype(np.float32)
    for name, f, reference in OPS:
        compiled = tpuasm_tools.compile(f, x, y)
        result = np.asarray(compiled(x, y))
        expected = reference(x.astype(np.float64), y.astype(np.float64))
        counts = tpuasm_tools.count_mnemonics(tpuasm_tools.kernel_listing(compiled))
        described = '、'.join(f'{mnemonic}×{count}' for mnemonic, count in sorted(counts.items()) if not mnemonic.startswith(SKIPPED))
        print(f'{name}：最大误差 {ulp_error(result, expected)} ULP；{described}')

if __name__ == '__main__':
    main()
