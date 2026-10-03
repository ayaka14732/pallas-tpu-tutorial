"""随机比特 → 标准正态分布：stateful_normal（反误差函数）与手写的 Box–Muller。比较每个随机数的指令、用到的 EUP 指令，以及落在 ±1σ、±2σ、±3σ 内的比例。"""
import tpu_init
tpu_init.initialise_one_chip()

import math

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

from distribution_common import ROWS, TILES, bits, compile_and_run, compute_counts, excerpt

def uniform() -> jax.Array:
    return jax.lax.bitcast_convert_type((bits() >> 9) | jnp.uint32(0x3F800000), jnp.float32) - 1.0

def box_muller() -> tuple[jax.Array, ...]:
    # 两个均匀数得到两个独立的正态数：r = sqrt(−2 ln u1)，θ = 2π u2。u1 取 (0, 1]，避免 log(0)。
    u1 = 1.0 - uniform()
    u2 = uniform()
    r = jnp.sqrt(-2.0 * jnp.log(u1))
    theta = 2.0 * math.pi * u2
    return r * jnp.cos(theta), r * jnp.sin(theta)

EUP = ('vpow2', 'vlog2', 'vrcp', 'vrsqrt', 'vtanh')

def report(name: str, samples: list[np.ndarray], listing: str, baseline: str) -> None:
    """两种方法都是每生成一个 TC VREG 的正态数消耗一个 TC VREG 的随机比特。"""
    values = np.concatenate([sample.ravel() for sample in samples]).astype(np.float64)
    counts = compute_counts(listing) - compute_counts(baseline)
    per_vreg = len(samples) * TILES
    total = sum(counts.values()) / per_vreg
    eup = {mnemonic: count / per_vreg for mnemonic, count in counts.items() if mnemonic.startswith(EUP)}
    print(f'## {name}')
    print(f'  每个正态数 TC VREG：变换指令 {total:.1f} 条，其中 EUP ' + '，'.join(f'{mnemonic} {count:g}' for mnemonic, count in sorted(eup.items())))
    print('  ' + '，'.join(f'{mnemonic} {count / per_vreg:g}' for mnemonic, count in counts.most_common(10)))
    within = [np.mean(np.abs(values) < k) for k in (1, 2, 3)]
    expected = [math.erf(k / math.sqrt(2)) for k in (1, 2, 3)]
    print(f'  {values.size} 个样本：均值 {values.mean():.4f}，标准差 {values.std():.4f}，最小 {values.min():.3f}，最大 {values.max():.3f}')
    print('  落在 ±1σ、±2σ、±3σ 内：' + '，'.join(f'{got:.4f}（{want:.4f}）' for got, want in zip(within, expected)))

def main() -> None:
    _, baseline = compile_and_run(lambda: (bits(),), (jnp.uint32,))
    samples, listing = compile_and_run(lambda: (pltpu.stateful_normal((ROWS, 128), jnp.float32),), (jnp.float32,))
    report('stateful_normal', samples, listing, baseline)
    print(excerpt(listing, 'vlog2', 12))
    samples, listing = compile_and_run(box_muller, (jnp.float32, jnp.float32))
    report('Box–Muller', samples, listing, baseline)
    print(f'  两个输出的相关系数 {np.corrcoef(samples[0].ravel(), samples[1].ravel())[0, 1]:.4f}')

if __name__ == '__main__':
    main()
