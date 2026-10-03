"""随机比特 → 伯努利分布：stateful_bernoulli 与整数阈值两种写法，概率实际被量化成什么；以及把结果直接当作掩码使用时的指令。"""
import tpu_init
tpu_init.initialise_one_chip()

from fractions import Fraction
import math

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

from distribution_common import ROWS, bits, compile_and_run, excerpt, per_tile

def threshold(p: float) -> int:
    """整数阈值：高 24 位小于 round(p · 2^24) 的概率是 round(p · 2^24) / 2^24。"""
    return round(p * 2**24)

def by_integer(p: float):
    # 右移 8 位后是非负的 int32，可以用有符号比较。
    return lambda: (((bits() >> 8).astype(jnp.int32) < threshold(p)).astype(jnp.int32),)

def dropout(x: jax.Array) -> tuple[jax.Array, ...]:
    # 以 1/4 的概率置 0，其余放大 4/3：掩码不写回内存，直接作为 vsel 的条件。
    keep = (bits() >> 8).astype(jnp.int32) >= threshold(0.25)
    return (jnp.where(keep, x * (4.0 / 3.0), 0.0),)

def main() -> None:
    _, baseline = compile_and_run(lambda: (bits(),), (jnp.uint32,))
    for p in (0.25, 0.3):
        # stateful_uniform 的结果是 k · 2^-23，u < p 成立的 k 有 ceil(p · 2^23) 个。
        exact = {'stateful_bernoulli': Fraction(math.ceil(Fraction(p) * 2**23), 2**23), '整数阈值': Fraction(threshold(p), 2**24)}
        cases = {'stateful_bernoulli': lambda: (pltpu.stateful_bernoulli(p, (ROWS, 128)).astype(jnp.int32),), '整数阈值': by_integer(p)}
        for name, sample in cases.items():
            (values,), listing = compile_and_run(sample, (jnp.int32,))
            rate = values.mean()
            print(f'## p = {p}，{name}')
            print(f'  每个 TC VREG 的变换指令 {per_tile(listing, baseline)}')
            print(f'  实际概率 {float(exact[name]):.12f}（与 p 相差 {float(exact[name] - Fraction(p)):.2e}）；{values.size} 个样本中为真的比例 {rate:.4f}，标准误差 {math.sqrt(p * (1 - p) / values.size):.4f}')
            if p == 0.25:
                print(excerpt(listing, 'vrng', 5))
    x = np.full((ROWS, 128), 3.0, np.float32)
    (values,), listing = compile_and_run(dropout, (jnp.float32,), (jnp.asarray(x),))
    print('## 掩码直接用于 vsel：jnp.where(keep, x · 4/3, 0)')
    print(f'  每个 TC VREG 的变换指令 {per_tile(listing, compile_and_run(lambda x: (bits(),), (jnp.uint32,), (jnp.asarray(x),))[1])}')
    print(f'  取值：{sorted(set(values.ravel().tolist()))}，为 0 的比例 {np.mean(values == 0):.4f}，均值 {values.mean():.4f}（期望 3）')
    print(excerpt(listing, 'vrng', 6))

if __name__ == '__main__':
    main()
