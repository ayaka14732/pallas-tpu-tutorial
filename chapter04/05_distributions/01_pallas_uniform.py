"""随机比特 → [0, 1) 的均匀分布：stateful_uniform、两种手写方法，以及 minval/maxval。比较每个 TC VREG 的指令、取值的间隔和能取到的最小、最大值。"""
import tpu_init
tpu_init.initialize_one_chip()

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

from distribution_common import ROWS, bits, compile_and_run, excerpt, per_tile

def mantissa() -> tuple[jax.Array, ...]:
    # 高 23 位放进 [1, 2) 的尾数，减 1 得到 [0, 1) 中 2^-23 的整数倍。
    return (jax.lax.bitcast_convert_type((bits() >> 9) | jnp.uint32(0x3F800000), jnp.float32) - 1.0,)

def convert() -> tuple[jax.Array, ...]:
    # 高 24 位转成整数再转成 f32（24 位以内的整数在 f32 中精确），乘以 2^-24。
    return ((bits() >> 8).astype(jnp.int32).astype(jnp.float32) * 2.0**-24,)

CASES = {
    'stateful_uniform': lambda: (pltpu.stateful_uniform((ROWS, 128), jnp.float32),),
    '手写：尾数': mantissa,
    '手写：整数转浮点': convert,
    'stateful_uniform(minval=-1, maxval=1)': lambda: (pltpu.stateful_uniform((ROWS, 128), jnp.float32, minval=-1.0, maxval=1.0),),
}

def main() -> None:
    (raw,), baseline = compile_and_run(lambda: (bits(),), (jnp.uint32,))
    results = {}
    for name, sample in CASES.items():
        (values,), listing = compile_and_run(sample, (jnp.float32,))
        results[name] = values
        lowest = values.min() if values.min() >= 0 else -1.0
        steps = (values.astype(np.float64) - lowest) / 2.0**-23
        print(f'## {name}')
        print(f'  每个 TC VREG 的变换指令 {per_tile(listing, baseline)}')
        print(f'  最小 {values.min():.9g}，最大 {values.max():.9g}，不同取值 {len(np.unique(values))} 个 / {values.size}，均值 {values.mean():.5f}')
        print(f'  全部是 2^-23 的整数倍：{bool(np.all(steps == np.round(steps)))}；全部是 2^-24 的整数倍：{bool(np.all(steps * 2 == np.round(steps * 2)))}')
        print(excerpt(listing, 'vrng', 6))
    # 两种 stateful_uniform 与手写尾数版本消耗同样的比特。
    print(f'stateful_uniform 与手写尾数版本逐位相同：{bool(np.array_equal(results["stateful_uniform"], results["手写：尾数"]))}')
    expected = ((raw >> 9).astype(np.float64)) * 2.0**-23
    print(f'手写尾数版本等于 (比特 >> 9) × 2^-23：{bool(np.array_equal(results["手写：尾数"], expected))}')

if __name__ == '__main__':
    main()
