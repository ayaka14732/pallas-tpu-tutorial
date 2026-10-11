"""原生 XLA 的三个对照：按行 take、lane gather（take_along_axis）、lane scatter（每行索引互不相同的置换）。"""
import tpu_init
tpu_init.initialize_one_core()

import jax
import jax.numpy as jnp
import numpy as np

import tpuasm_tools

def main() -> None:
    rng = np.random.default_rng(0)
    table = np.arange(64 * 128, dtype=np.float32).reshape(64, 128)
    rows = np.array([5, 3, 60, 0, 17, 17, 42, 9], np.int32)
    x = np.arange(8 * 128, dtype=np.float32).reshape(8, 128)
    lane_indices = rng.integers(0, 128, (8, 128)).astype(np.int32)
    permutation = np.stack([rng.permutation(128) for _ in range(8)]).astype(np.int32)
    expected_scatter = np.zeros_like(x)
    np.put_along_axis(expected_scatter, permutation, x, axis=1)
    cases = (
        ('按行 take：table[rows]', lambda a, b: jnp.take(a, b, axis=0), table, rows, table[rows]),
        ('lane gather：take_along_axis', lambda a, b: jnp.take_along_axis(a, b, axis=1), x, lane_indices, np.take_along_axis(x, lane_indices, axis=1)),
        ('lane scatter：out[s, p[s,l]] = x[s,l]', lambda a, b: jnp.zeros_like(a).at[jnp.arange(8)[:, None], b].set(a, unique_indices=True), x, permutation, expected_scatter),
    )
    for name, f, a, b, expected in cases:
        compiled = tpuasm_tools.compile(f, a, b)
        np.testing.assert_array_equal(np.asarray(compiled(a, b)), expected)
        listing = tpuasm_tools.kernel_listing(compiled)
        counts = tpuasm_tools.count_mnemonics(listing)
        bundles = sum(line.startswith('{') for line in listing.splitlines())
        print(f'## {name}：数值检查通过；HLO 段共 {bundles} 个 bundle')
        print('指令统计：', '、'.join(f'{mnemonic}×{count}' for mnemonic, count in sorted(counts.items())))
        print('HLO 段：', '；'.join(line[len('# entry bundle: '):] for line in listing.splitlines() if line.startswith('# entry bundle')))
        print()

if __name__ == '__main__':
    main()
