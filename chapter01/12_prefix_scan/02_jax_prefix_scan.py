"""原生 XLA 的前缀和对照：jnp.cumsum 沿 lane 与沿 sublane。"""
import tpu_init
tpu_init.initialise_one_core()

import jax.numpy as jnp
import numpy as np

import scan_common
import tpuasm_tools

def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.integers(-100, 100, (8, 128)).astype(np.float32)
    for axis in (1, 0):
        compiled = tpuasm_tools.compile(lambda x: jnp.cumsum(x, axis=axis), x)
        result = np.asarray(compiled(x))
        listing = tpuasm_tools.kernel_listing(compiled)
        entries = [line[len('# entry bundle: '):] for line in listing.splitlines() if line.startswith('# entry bundle')]
        bundles = sum(line.startswith('{') for line in listing.splitlines())
        print(f'## jnp.cumsum(x, axis={axis})：与 NumPy 结果一致 {bool(np.array_equal(result, np.cumsum(x, axis=axis)))}；HLO 段共 {bundles} 个 bundle：{"；".join(entries)}')
        print(f'  {scan_common.describe(listing)}')
        print()

if __name__ == '__main__':
    main()
