"""沿 lane 的前缀和写成矩阵乘法：y = x @ U，U[k, l] = 1（k ≤ l）。比较默认精度与 HIGHEST 精度在整数输入和随机浮点输入上的误差，以及清单中的指令。"""
import tpu_init
tpu_init.initialise_one_chip()

import jax
import jax.numpy as jnp
import numpy as np

import scan_common

def scan_by_matmul(precision):
    def f(x: jax.Array) -> jax.Array:
        # 上三角全 1 矩阵：第 l 列把第 0 到 l 行加起来。
        row = jax.lax.broadcasted_iota(jnp.int32, (128, 128), 0)
        column = jax.lax.broadcasted_iota(jnp.int32, (128, 128), 1)
        upper = (row <= column).astype(jnp.float32)
        return jnp.dot(x, upper, precision=precision, preferred_element_type=jnp.float32)

    return f

def main() -> None:
    rng = np.random.default_rng(0)
    inputs = {
        '−100 到 100 的整数': rng.integers(-100, 100, (8, 128)).astype(np.float32),
        '[0, 1) 的随机浮点数': rng.random((8, 128), dtype=np.float32),
    }
    for precision in (None, jax.lax.Precision.HIGHEST):
        print(f'## precision={precision}')
        for name, x in inputs.items():
            result, listing = scan_common.run(scan_by_matmul(precision), x)
            exact = np.cumsum(x.astype(np.float64), axis=1)
            error = np.abs(result - exact)
            print(f'  {name}：与 float64 cumsum 相比最大绝对误差 {error.max():.3g}，最大相对误差 {np.max(error / np.maximum(np.abs(exact), 1e-30)):.3g}')
        print('  计算指令：' + scan_common.describe(listing))

if __name__ == '__main__':
    main()
