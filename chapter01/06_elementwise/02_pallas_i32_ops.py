"""uint32 与 int32 [8,128] 的逐元素运算：每种运算单独编译一个 kernel，列出它用到的计算指令，并与 NumPy 逐元素比较。"""
import tpu_init
tpu_init.initialise_one_chip()

import jax
import jax.numpy as jnp
import numpy as np

import elementwise_common

# (名称, 函数, 输入 dtype)；移位与位运算用 uint32，比较与除法另用有符号 int32。
OPS = (
    ('x + y', lambda x, y: x + y),
    ('x - y', lambda x, y: x - y),
    ('x * y', lambda x, y: x * y),
    ('x & y', lambda x, y: x & y),
    ('x | y', lambda x, y: x | y),
    ('x ^ y', lambda x, y: x ^ y),
    ('x << 3', lambda x, y: x << 3),
    ('x >> 3（逻辑）', lambda x, y: x >> 3),
    ('x << y', lambda x, y: x << (y & 31)),
    ('maximum(x, y)', jnp.maximum),
    ('population_count(x)', lambda x, y: jax.lax.population_count(x)),
    ('clz(x)', lambda x, y: jax.lax.clz(x)),
    ('x // y', lambda x, y: x // (y | 1)),
)
SIGNED_OPS = (
    ('maximum(x, y)', jnp.maximum),
    ('x >> 3（算术）', lambda x, y: x >> 3),
    ('x * y', lambda x, y: x * y),
    ('x // y', lambda x, y: x // (y | 1)),
    ('x % y', lambda x, y: x % (y | 1)),
)

def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2**32, (8, 128), dtype=np.uint64).astype(np.uint32)
    y = rng.integers(0, 2**32, (8, 128), dtype=np.uint64).astype(np.uint32)
    signed_x = x.view(np.int32)
    signed_y = y.view(np.int32)
    for name, f, (a, b) in [(name, f, (x, y)) for name, f in OPS] + [(f'int32 {name}', f, (signed_x, signed_y)) for name, f in SIGNED_OPS]:
        try:
            result, counts, _ = elementwise_common.run(f, a, b)
        except Exception as error:
            print(f'{name}：编译失败：{str(error).splitlines()[0]}')
            continue
        expected = np.asarray(jax.jit(f, backend='cpu')(a, b))
        print(f'{name}：与 CPU 结果一致：{bool(np.array_equal(result, expected))}；{elementwise_common.describe(counts)}')

if __name__ == '__main__':
    main()
