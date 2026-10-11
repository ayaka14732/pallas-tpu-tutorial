"""前缀和：jnp.cumsum 沿 lane 与沿 sublane，以及用循环移位和掩码手写的 Hillis–Steele 扫描。"""
import tpu_init
tpu_init.initialize_one_chip()

import jax.numpy as jnp
import ml_dtypes
import numpy as np

import scan_common
from scan_common import hillis_steele_lanes, hillis_steele_sublanes, suffix_sublanes

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
