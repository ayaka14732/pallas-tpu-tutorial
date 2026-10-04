"""原生 XLA 的同一组转置，用与 Pallas 版本相同的方式统计 XLU 的提交与取回。"""
import tpu_init
tpu_init.initialise_one_core()

import jax
import jax.numpy as jnp
import numpy as np

import tpuasm_tools
from transpose_common import CASES, summary

def main() -> None:
    rng = np.random.default_rng(0)
    for name, dtype, shape, count in CASES:
        xs = tuple(jnp.asarray(rng.integers(-1000, 1000, shape)).astype(dtype) for _ in range(count))
        compiled = tpuasm_tools.compile(lambda xs: tuple(x.T for x in xs), xs)
        for x, result in zip(xs, compiled(xs)):
            np.testing.assert_array_equal(np.asarray(result), np.asarray(x).T)
        listing = tpuasm_tools.kernel_listing(compiled)
        entries = [line[len('# entry bundle: '):] for line in listing.splitlines() if line.startswith('# entry bundle')]
        print(f'## {name}：数值检查通过；HLO 段：{"；".join(entries)}')
        print(summary(listing))
        print()

if __name__ == '__main__':
    main()
