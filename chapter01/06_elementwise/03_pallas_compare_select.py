"""比较、选择与向量掩码：where、掩码的与或非，以及同时活跃的掩码超过 8 个时编译器怎样处理。"""
import tpu_init
tpu_init.initialise_one_chip()

import functools

import jax
import jax.numpy as jnp
import numpy as np

import elementwise_common

def many_masks(x: jax.Array, y: jax.Array, count: int) -> jax.Array:
    """先算出 count 个掩码，再依次使用；每个掩码都要从产生一直保留到最后一次使用。"""
    masks = [x > y + i for i in range(count)]
    total = jnp.zeros_like(x)
    for i, mask in enumerate(masks):
        total = jnp.where(mask, total + y, total - x)
    for mask in masks:
        total = jnp.where(mask, total * 2, total)
    return total

OPS = (
    ('where(x > y, x, y)', lambda x, y: jnp.where(x > y, x, y)),
    ('where(x > 0, x, 0.1 * x)', lambda x, y: jnp.where(x > 0, x, 0.1 * x)),
    ('where((x > 0) & (y > 0), x, y)', lambda x, y: jnp.where((x > 0) & (y > 0), x, y)),
    ('where((x > 0) | ~(y > 0), x, y)', lambda x, y: jnp.where((x > 0) | ~(y > 0), x, y)),
    ('同时活跃 6 个掩码', functools.partial(many_masks, count=6)),
    ('同时活跃 12 个掩码', functools.partial(many_masks, count=12)),
)

def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(-4, 4, (8, 128)).astype(np.float32)
    y = rng.uniform(-4, 4, (8, 128)).astype(np.float32)
    for name, f in OPS:
        result, counts, listing = elementwise_common.run(f, x, y)
        expected = np.asarray(jax.jit(f, backend='cpu')(x, y))
        masks = sorted({token.strip(',') for line in listing.splitlines() for token in line.split() if token.strip(',').startswith('vm') and token.strip(',')[2:].isdigit()})
        print(f'{name}：与 CPU 结果一致：{bool(np.allclose(result, expected))}；用到的掩码寄存器：{masks}')
        print(f'  {elementwise_common.describe(counts)}')

if __name__ == '__main__':
    main()
