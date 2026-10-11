"""f32[8,128] 的逐元素运算：每种运算单独编译一个 kernel，列出它用到的计算指令，并用 ULP 衡量与 float64 参考的误差。"""
import tpu_init
tpu_init.initialize_one_chip()

import jax
import jax.numpy as jnp
import numpy as np

import elementwise_common

OPS = (
    ('x + y', lambda x, y: x + y, lambda x, y: x + y),
    ('x - y', lambda x, y: x - y, lambda x, y: x - y),
    ('x * y', lambda x, y: x * y, lambda x, y: x * y),
    ('x * y + 1', lambda x, y: x * y + 1, lambda x, y: x * y + 1),
    ('maximum(x, y)', jnp.maximum, np.maximum),
    ('-x', lambda x, y: -x, lambda x, y: -x),
    ('abs(x)', lambda x, y: jnp.abs(x), lambda x, y: np.abs(x)),
    ('x / y', lambda x, y: x / y, lambda x, y: x / y),
    ('1 / y', lambda x, y: 1 / y, lambda x, y: 1 / y),
    ('exp(x)', lambda x, y: jnp.exp(x), lambda x, y: np.exp(x)),
    ('exp2(x)', lambda x, y: jnp.exp2(x), lambda x, y: np.exp2(x)),
    ('log(y)', lambda x, y: jnp.log(y), lambda x, y: np.log(y)),
    ('sqrt(y)', lambda x, y: jnp.sqrt(y), lambda x, y: np.sqrt(y)),
    ('rsqrt(y)', lambda x, y: jax.lax.rsqrt(y), lambda x, y: 1 / np.sqrt(y)),
    ('tanh(x)', lambda x, y: jnp.tanh(x), lambda x, y: np.tanh(x)),
    ('sigmoid(x)', lambda x, y: jax.nn.sigmoid(x), lambda x, y: 1 / (1 + np.exp(-x))),
    ('sin(x)', lambda x, y: jnp.sin(x), lambda x, y: np.sin(x)),
)

def ulp_error(result: np.ndarray, expected: np.ndarray) -> int:
    """result 与 float64 参考舍入到 f32 后相差多少个 f32 可表示数（同号时按位型之差计算）。"""
    target = expected.astype(np.float32)
    assert np.all(np.sign(result) == np.sign(target))
    return int(np.max(np.abs(result.view(np.int32).astype(np.int64) - target.view(np.int32).astype(np.int64))))

def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(-4, 4, (8, 128)).astype(np.float32)
    y = rng.uniform(0.25, 4, (8, 128)).astype(np.float32)
    for name, f, reference in OPS:
        try:
            result, counts, _ = elementwise_common.run(f, x, y)
        except Exception as error:
            print(f'{name}：编译失败：{str(error).splitlines()[0]}')
            continue
        expected = reference(x.astype(np.float64), y.astype(np.float64))
        absolute = np.max(np.abs(result - expected))
        print(f'{name}：最大误差 {ulp_error(result, expected)} ULP，最大绝对误差 {absolute:.1e}；{elementwise_common.describe(counts)}')

if __name__ == '__main__':
    main()
