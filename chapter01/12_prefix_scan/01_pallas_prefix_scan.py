"""前缀和：jnp.cumsum 沿 lane 与沿 sublane，以及用循环移位和掩码手写的 Hillis–Steele 扫描。"""
import tpu_init
tpu_init.initialise_one_chip()

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import ml_dtypes
import numpy as np

import scan_common

def hillis_steele_lanes(x: jax.Array) -> jax.Array:
    """沿 lane 的包含式前缀和：第 d 轮把 lane l 加上 lane l - 2^d 的值，共 7 轮。"""
    lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    for d in range(7):
        shift = 1 << d
        x = x + jnp.where(lane >= shift, pltpu.roll(x, shift, axis=1), 0)
    return x

def hillis_steele_sublanes(x: jax.Array) -> jax.Array:
    """沿 sublane 的包含式前缀和：第 d 轮把 sublane s 加上 sublane s - 2^d 的值，共 3 轮。"""
    sublane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 0)
    for d in range(3):
        shift = 1 << d
        x = x + jnp.where(sublane >= shift, pltpu.roll(x, shift, axis=0), 0)
    return x

def suffix_sublanes(x: jax.Array) -> jax.Array:
    """沿 sublane 的后缀和：第 d 轮把 sublane s 加上 sublane s + 2^d 的值；roll 位移取 8 - 2^d。"""
    sublane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 0)
    for d in range(3):
        shift = 1 << d
        x = x + jnp.where(sublane < 8 - shift, pltpu.roll(x, 8 - shift, axis=0), 0)
    return x

def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.integers(-100, 100, (8, 128)).astype(np.float32)
    cases = (
        ('jnp.cumsum(x, axis=1)', lambda x: jnp.cumsum(x, axis=1), x),
        ('jnp.cumsum(x, axis=0)', lambda x: jnp.cumsum(x, axis=0), x),
        ('手写 Hillis–Steele，沿 lane', hillis_steele_lanes, x),
        ('手写 Hillis–Steele，沿 sublane', hillis_steele_sublanes, x),
        ('手写后缀和，沿 sublane', suffix_sublanes, x),
        ('手写 Hillis–Steele，沿 lane，bf16 输入', lambda x: hillis_steele_lanes(x.astype(jnp.float32)), x.astype(ml_dtypes.bfloat16)),
    )
    for name, f, data in cases:
        try:
            result, listing = scan_common.run(f, data)
        except Exception as error:
            print(f'## {name}：编译失败：{str(error).splitlines()[0][:300]}')
            print()
            continue
        axis = 0 if 'sublane' in name or 'axis=0' in name else 1
        expected = np.cumsum(data.astype(np.float32), axis=axis)
        if '后缀和' in name:
            expected = np.cumsum(data[::-1], axis=0)[::-1]
        print(f'## {name}：与 NumPy 结果一致 {bool(np.array_equal(result, expected))}')
        print(f'  {scan_common.describe(listing)}')
        print()

if __name__ == '__main__':
    main()
