"""每行取最大的 k 个值及其下标：argmax、top-1、top-8，跨两个 lane tile 的 top-8，bf16 输入，以及有效值少于 k 个的边界情况。"""
import tpu_init
tpu_init.initialize_one_chip()

import jax
import ml_dtypes
import numpy as np

from top_k_common import argmax, run, top_k_by_hand
import tpuasm_tools

SKIPPED = ('vld', 'vst', 'vsync', 'vwait', 'vtrace', 'cfence', 'dma.', 's', 'p')

def main() -> None:
    rng = np.random.default_rng(0)
    # 每行是互不相同的值的随机排列，避免并列。
    x128 = np.stack([rng.permutation(128) for _ in range(8)]).astype(np.float32)
    x256 = np.stack([rng.permutation(256) for _ in range(8)]).astype(np.float32)
    ties = np.tile(np.arange(128) % 16, (8, 1)).astype(np.float32)
    sparse = np.full((8, 128), -np.inf, np.float32)
    sparse[:, :3] = x128[:, :3]
    cases = (
        ('max 与 argmax，f32[8,128]', argmax, x128, 1),
        ('max 与 argmax，每行的最大值在 lane 15、31、…、127 并列', argmax, ties, 1),
        ('lax.top_k(x, 1)，f32[8,128]', lambda x: jax.lax.top_k(x, 1), x128, 1),
        ('lax.top_k(x, 1, is_stable=False)，f32[8,128]', lambda x: jax.lax.top_k(x, 1, is_stable=False), x128, 1),
        ('lax.top_k(x, 8, is_stable=False)，f32[8,128]', lambda x: jax.lax.top_k(x, 8, is_stable=False), x128, 8),
        ('lax.top_k(x, 8, is_stable=False)，f32[8,256]', lambda x: jax.lax.top_k(x, 8, is_stable=False), x256, 8),
        ('lax.top_k(x, 8, is_stable=False)，bf16[8,128]', lambda x: jax.lax.top_k(x, 8, is_stable=False), x128.astype(ml_dtypes.bfloat16), 8),
        ('lax.top_k(x, 8, is_stable=False)，每行只有 3 个有限值，其余为 -inf', lambda x: jax.lax.top_k(x, 8, is_stable=False), sparse, 8),
        ('手写 top-8，用掩码排除已选位置，f32[8,128]', top_k_by_hand(8), x128, 8),
        ('手写 top-8，用掩码排除已选位置，每行只有 3 个有限值，其余为 -inf', top_k_by_hand(8), sparse, 8),
    )
    for name, f, x, k in cases:
        try:
            (values, indices), listing = run(f, x)
        except Exception as error:
            print(f'## {name}：编译失败：{str(error).splitlines()[0][:250]}')
            print()
            continue
        expected_values, expected_indices = (np.asarray(array) for array in jax.lax.top_k(jax.device_put(x, jax.local_devices(backend='cpu')[0]), k))
        print(f'## {name}')
        print(f'  值与 CPU 结果一致：{bool(np.array_equal(values, expected_values))}；下标与 CPU 结果一致：{bool(np.array_equal(indices, expected_indices))}')
        if not np.array_equal(indices, expected_indices):
            print(f'  第 0 行下标：{indices[0].tolist()}；CPU：{expected_indices[0].tolist()}')
        counts = tpuasm_tools.count_mnemonics(listing)
        print('  计算指令：' + '、'.join(f'{mnemonic}×{count}' for mnemonic, count in sorted(counts.items()) if not mnemonic.startswith(SKIPPED)))
        print()

if __name__ == '__main__':
    main()
