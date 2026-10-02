"""原生 XLA 的 top-k 对照：f32[8,128] 的 top-1、top-8，以及每行只有 3 个有限值时的 top-8。"""
import tpu_init
tpu_init.initialise_one_chip()

import jax
import numpy as np

import tpuasm_tools

def main() -> None:
    rng = np.random.default_rng(0)
    x = np.stack([rng.permutation(128) for _ in range(8)]).astype(np.float32)
    sparse = np.full((8, 128), -np.inf, np.float32)
    sparse[:, :3] = x[:, :3]
    for name, data, k in (('top-1，f32[8,128]', x, 1), ('top-8，f32[8,128]', x, 8), ('top-8，每行只有 3 个有限值', sparse, 8)):
        compiled = tpuasm_tools.compile(lambda a: jax.lax.top_k(a, k), data)
        values, indices = (np.asarray(array) for array in compiled(data))
        expected_values, expected_indices = (np.asarray(array) for array in jax.jit(jax.lax.top_k, static_argnums=1, backend='cpu')(data, k))
        listing = tpuasm_tools.kernel_listing(compiled)
        entries = [line[len('# entry bundle: '):] for line in listing.splitlines() if line.startswith('# entry bundle')]
        bundles = sum(line.startswith('{') for line in listing.splitlines())
        print(f'## {name}：值一致 {bool(np.array_equal(values, expected_values))}，下标一致 {bool(np.array_equal(indices, expected_indices))}；HLO 段共 {bundles} 个 bundle')
        print('  ' + '；'.join(entries))
        print()

if __name__ == '__main__':
    main()
