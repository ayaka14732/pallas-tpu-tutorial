"""f32 → bf16 的随机舍入：pltpu.stochastic_round 在 TPU v4 上能否编译；用整数加法手写，检查向上舍入的概率、对各种数值是否无偏，以及无穷大、NaN、最大有限值等特殊输入。"""
import tpu_init
tpu_init.initialise_one_chip()

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from distribution_common import ROWS, bits, compile_and_run, excerpt, per_tile

def by_hand(x: jax.Array) -> tuple[jax.Array, ...]:
    # bf16 是 f32 的高 16 位：把随机的 16 位加到被舍去的低 16 位上，进位的概率正好等于被舍去部分的比例，再截断。
    raw = jax.lax.bitcast_convert_type(x, jnp.uint32)
    rounded = (raw + (bits() & jnp.uint32(0xFFFF))) & jnp.uint32(0xFFFF0000)
    return (jax.lax.bitcast_convert_type(rounded, jnp.float32),)

def api(x: jax.Array) -> tuple[jax.Array, ...]:
    return (pltpu.stochastic_round(x, bits(), target_dtype=jnp.bfloat16).astype(jnp.float32),)

def lanes(values: np.ndarray) -> jnp.ndarray:
    """每个 lane 一个输入值，每一行是对这 128 个值的一次独立舍入。"""
    return jnp.asarray(np.broadcast_to(values.astype(np.float32), (ROWS, 128)))

def main() -> None:
    one = lanes(np.full(128, 1.0 + 2.0**-10))
    try:
        compile_and_run(api, (jnp.float32,), (one,))
    except Exception as error:
        print(f'## pltpu.stochastic_round：编译失败：{str(error).splitlines()[0].split(": ")[-1]}')
    _, baseline = compile_and_run(lambda x: (bits(),), (jnp.uint32,), (one,))
    (values,), listing = compile_and_run(by_hand, (jnp.float32,), (one,))
    print('## 手写：1 + 2^-10 介于 bf16 的 1 与 1 + 2^-7 之间，向上的概率应为 1/8')
    print(f'  每个 TC VREG 的变换指令 {per_tile(listing, baseline)}')
    print(f'  向上的比例 {np.mean(values > 1.0):.4f}，其余都是 1.0：{bool(np.all((values == 1.0) | (values == 1.0 + 2.0**-7)))}')
    print(excerpt(listing, 'vrng', 5))
    # 128 个随机的数：量级 2^-20 到 2^20，正负各半。
    rng = np.random.default_rng(0)
    x = (rng.uniform(1, 2, 128) * 2.0 ** rng.integers(-20, 21, 128) * rng.choice([-1, 1], 128)).astype(np.float32)
    (values,), _ = compile_and_run(by_hand, (jnp.float32,), (lanes(x),))
    mean = values.astype(np.float64).mean(axis=0)
    nearest = x.astype(ml_dtypes.bfloat16).astype(np.float64)
    print(f'## 128 个随机的数，每个独立舍入 {ROWS} 次')
    print(f'  {ROWS} 次的平均值与原值的最大相对误差 {np.max(np.abs(mean / x - 1)):.2e}；舍入到最近的 bf16 的最大相对误差 {np.max(np.abs(nearest / x - 1)):.2e}')
    lower = x.view(np.uint32) & np.uint32(0xFFFF0000)
    upper = lower + np.uint32(0x10000)
    neighbours = (values.view(np.uint32) == lower) | (values.view(np.uint32) == upper)
    print(f'  每个结果都是原值两侧相邻的两个 bf16 之一：{bool(np.all(neighbours))}')
    special = np.array([np.inf, -np.inf, np.nan, np.float32(3.4028235e38), 0.0, -0.0, 1e-40, -1e-40], np.float32)
    payload_nan = np.array([0x7F800001], np.uint32).view(np.float32)
    inputs = np.concatenate([special, payload_nan, np.ones(128 - len(special) - 1, np.float32)])
    (values,), _ = compile_and_run(by_hand, (jnp.float32,), (lanes(inputs),))
    print('## 特殊输入')
    names = ['+inf', '-inf', 'NaN', '最大有限值', '+0', '-0', '1e-40（非正规数）', '-1e-40', 'NaN（位型 0x7f800001）']
    for lane, name in enumerate(names):
        column = values[:, lane]
        outcomes = {}
        for value in column:
            key = 'NaN' if np.isnan(value) else f'{value:.7g}'
            outcomes[key] = outcomes.get(key, 0) + 1
        print(f'  {name}：' + '，'.join(f'{key} × {count}' for key, count in sorted(outcomes.items())))

if __name__ == '__main__':
    main()
