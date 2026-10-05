"""随机比特 → 指数分布 −log u 与 Gumbel 分布 −log(−log u)：指令、与主机 float64 的误差，以及均匀数的分辨率给分布尾部设的上限。"""
import tpu_init
tpu_init.initialize_one_chip()

import math

import jax
import jax.numpy as jnp
import numpy as np

from distribution_common import bits, compile_and_run, excerpt, per_tile

def open_uniform() -> jax.Array:
    # (0, 1] 中 2^-23 的整数倍：1 − [0, 1)，避免 log(0)。
    return 2.0 - jax.lax.bitcast_convert_type((bits() >> 9) | jnp.uint32(0x3F800000), jnp.float32)

def exponential() -> tuple[jax.Array, ...]:
    u = open_uniform()
    return u, -jnp.log(u)

def gumbel() -> tuple[jax.Array, ...]:
    u = open_uniform()
    return u, -jnp.log(-jnp.log(u))

def main() -> None:
    _, baseline = compile_and_run(lambda: (bits(),), (jnp.uint32,))
    for name, sample, mean, reference in (
        ('指数分布 −log u', exponential, 1.0, lambda u: -np.log(u)),
        ('Gumbel 分布 −log(−log u)', gumbel, 0.5772156649, lambda u: -np.log(-np.log(u))),
    ):
        (u, values), listing = compile_and_run(sample, (jnp.float32, jnp.float32))
        exact = reference(u.astype(np.float64))
        finite = np.isfinite(exact)
        error = np.abs(values[finite] - exact[finite])
        print(f'## {name}')
        print(f'  每个 TC VREG 的变换指令（含生成 (0, 1] 均匀数的 3 条）{per_tile(listing, baseline)}')
        print(f'  u 的最小值 {u.min():.6g}，最大值 {u.max():.6g}；结果的最小 {values[finite].min():.4f}，最大 {values[finite].max():.4f}，均值 {values[finite].mean():.4f}（理论值 {mean:.4f}）')
        print(f'  与同一 u 的 float64 公式相比：最大绝对误差 {error.max():.2e}，最大相对误差 {np.max(error / np.maximum(np.abs(exact[finite]), 1e-30)):.2e}；非有限值 {int(np.sum(~np.isfinite(values)))} 个')
        print(excerpt(listing, 'vlog2', 4))
    print(f'u ≥ 2^-23 时 −log u 的上限 23 · ln 2 = {23 * math.log(2):.4f}；u = 1 时 Gumbel 为 +∞，概率 2^-23')

if __name__ == '__main__':
    main()
