"""归约：沿 lane、沿 sublane、跨多个 tile、bf16 输入，各单独编译一个 kernel，列出计算指令并检查数值。"""
import tpu_init
tpu_init.initialize_one_chip()

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import ml_dtypes
import numpy as np

import reduction_common

def max_tree(shifts: tuple[int, ...]):
    """手写 sublane 方向的二叉树：依次与循环移动 4、2、1 行的自己取最大值，每个 sublane 都得到 8 行的最大值。"""

    def f(x: jax.Array) -> jax.Array:
        for shift in shifts:
            x = jnp.maximum(x, pltpu.roll(x, shift, axis=0))
        return x[0:1, :]

    return f

def main() -> None:
    rng = np.random.default_rng(0)
    small = rng.integers(-100, 100, (8, 128)).astype(np.float32)
    tall = rng.integers(-100, 100, (64, 128)).astype(np.float32)
    cases = (
        ('sum(x, axis=1)，f32[8,128]', lambda x: jnp.sum(x, axis=1, keepdims=True), small),
        ('max(x, axis=1)，f32[8,128]', lambda x: jnp.max(x, axis=1, keepdims=True), small),
        ('sum(x, axis=0)，f32[8,128]', lambda x: jnp.sum(x, axis=0, keepdims=True), small),
        ('max(x, axis=0)，f32[8,128]', lambda x: jnp.max(x, axis=0, keepdims=True), small),
        ('手写树形 max，roll 位移 4、2、1', max_tree((4, 2, 1)), small),
        ('手写树形 max，roll 位移 4、6、7', max_tree((4, 6, 7)), small),
        ('sum(x, axis=0)，f32[64,128]', lambda x: jnp.sum(x, axis=0, keepdims=True), tall),
        ('sum(x)，f32[8,128]', lambda x: jnp.sum(x, keepdims=True), small),
        ('sum(x, axis=1)，bf16[16,128]', lambda x: jnp.sum(x.astype(jnp.float32), axis=1, keepdims=True), rng.integers(-100, 100, (16, 128)).astype(ml_dtypes.bfloat16)),
    )
    for name, f, x in cases:
        try:
            result, listing = reduction_common.run(f, x)
        except Exception as error:
            print(f'## {name}：编译失败：{str(error).splitlines()[0][:300]}')
            print()
            continue
        expected = np.asarray(jax.jit(f, backend='cpu')(x))
        print(f'## {name}：与 CPU 结果一致 {bool(np.array_equal(result, expected))}')
        print(f'  {reduction_common.describe(listing)}')
        print('\n'.join(line.split('#')[0].rstrip() for line in listing.splitlines() if any(f'{slot}: ' in line for slot in ('va0', 'va1', 'vx0', 'vx1', 'vr0', 'vr1'))))
        print()

if __name__ == '__main__':
    main()
