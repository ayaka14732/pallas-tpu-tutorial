"""从随机比特到分布：均匀分布、伯努利、正态、Gumbel，以及随机舍入。每种都在 kernel 中用硬件 vrng 生成 u32[64,128] 个比特再变换，统计每个 TC VREG 的指令数并检查结果。"""
import tpu_init
tpu_init.initialise_one_chip()

from collections.abc import Callable

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

import tpuasm_tools

ROWS = 64
TILES = ROWS // 8

def build(sample: Callable[[], tuple[jax.Array, ...]], dtypes: tuple):
    """kernel 先 prng_seed(1)，再由 sample() 得到若干个 [ROWS,128] 的数组，依次写出。"""
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=(),
        out_specs=tuple(P() for _ in dtypes),
        check_vma=False,
    )
    def draw():
        @pl.kernel(
            out_type=tuple(jax.ShapeDtypeStruct((ROWS, 128), dtype) for dtype in dtypes),
            mesh=tc_mesh,
            scratch_types=(*(pltpu.VMEM((ROWS, 128), dtype) for dtype in dtypes), pltpu.SemaphoreType.DMA),
            name='draw',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(*refs: Ref) -> None:
            outputs, buffers, sem = refs[:len(dtypes)], refs[len(dtypes):-1], refs[-1]
            pltpu.prng_seed(1)
            for buffer, value in zip(buffers, sample(), strict=True):
                buffer[...] = value
            for buffer, output in zip(buffers, outputs, strict=True):
                pltpu.async_copy(buffer, output, sem).wait()

        return kernel()

    return mesh, draw

def bits() -> jax.Array:
    return pltpu.prng_random_bits((ROWS, 128)).astype(jnp.uint32)

def uniform_by_hand() -> tuple[jax.Array, ...]:
    # 高 23 位放进 [1, 2) 的尾数，减 1 得到 [0, 1) 中 2^-23 的整数倍。
    u = jax.lax.bitcast_convert_type((bits() >> 9) | jnp.uint32(0x3F800000), jnp.float32) - 1.0
    return (u,)

def bernoulli_by_integer() -> tuple[jax.Array, ...]:
    # 概率 1/4 写成比特的阈值：高 24 位小于 2^22 的概率恰好是 2^22 / 2^24。
    return (((bits() >> 8).astype(jnp.int32) < (1 << 22)).astype(jnp.int32),)

def gumbel() -> tuple[jax.Array, ...]:
    # 均匀数取 (0, 1]：1 − [0, 1)，避免 log(0)。
    u = 1.0 - uniform_by_hand()[0]
    return u, -jnp.log(-jnp.log(u))

# 1 + 2^-10 介于两个相邻的 bf16 之间（1 与 1 + 2^-7），随机舍入时向上的概率应为 1/8。
ROUND_INPUT = 1.0 + 2.0**-10

def stochastic_round_api() -> tuple[jax.Array, ...]:
    x = jnp.full((ROWS, 128), ROUND_INPUT, jnp.float32)
    return (pltpu.stochastic_round(x, bits(), target_dtype=jnp.bfloat16).astype(jnp.float32),)

def stochastic_round_by_hand() -> tuple[jax.Array, ...]:
    # bf16 是 f32 的高 16 位：把随机的 16 位加到被舍去的低 16 位上，进位的概率正好等于被舍去部分的比例，再截断。
    x = jnp.full((ROWS, 128), ROUND_INPUT, jnp.float32)
    raw = jax.lax.bitcast_convert_type(x, jnp.uint32)
    rounded = (raw + (bits() & jnp.uint32(0xFFFF))) & jnp.uint32(0xFFFF0000)
    return (jax.lax.bitcast_convert_type(rounded, jnp.float32),)

SAMPLERS = {
    'stateful_uniform': (lambda: (pltpu.stateful_uniform((ROWS, 128), jnp.float32),), (jnp.float32,)),
    '手写均匀分布': (uniform_by_hand, (jnp.float32,)),
    'stateful_bernoulli(p=0.25)': (lambda: (pltpu.stateful_bernoulli(0.25, (ROWS, 128)).astype(jnp.int32),), (jnp.int32,)),
    '整数阈值伯努利(p=1/4)': (bernoulli_by_integer, (jnp.int32,)),
    'stateful_normal': (lambda: (pltpu.stateful_normal((ROWS, 128), jnp.float32),), (jnp.float32,)),
    'Gumbel：−log(−log u)': (gumbel, (jnp.float32, jnp.float32)),
    'pltpu.stochastic_round → bf16': (stochastic_round_api, (jnp.float32,)),
    '手写随机舍入 → bf16': (stochastic_round_by_hand, (jnp.float32,)),
}

def main() -> None:
    for name, (sample, dtypes) in SAMPLERS.items():
        mesh, draw = build(sample, dtypes)
        try:
            compiled = tpuasm_tools.compile(draw, mesh=mesh)
        except Exception as error:
            print(f'## {name}：编译失败：{str(error).splitlines()[0].split(": ")[-1]}')
            continue
        values = [np.asarray(value) for value in compiled()]
        counts = tpuasm_tools.count_mnemonics(tpuasm_tools.kernel_listing(compiled, pallas_only=True))
        # 与只生成比特的 kernel 相比，多出来的计算指令就是变换的代价。
        compute = {mnemonic: count for mnemonic, count in counts.items() if mnemonic.startswith(('va', 'vm', 'vs', 'vx', 'vo', 've', 'vn', 'vc', 'vp', 'vl', 'vr', 'vw', 'vf', 'vt')) and not mnemonic.startswith(('vld', 'vst', 'vsync', 'vwait', 'vtrace', 'vnop', 'vrng', 'vsettm'))}
        eup = sum(count for mnemonic, count in counts.items() if mnemonic.startswith(('vpow2', 'vlog2', 'vrcp', 'vrsqrt', 'vtanh')))
        print(f'## {name}：EUP 指令 {eup}，vrng {counts.get("vrng.8x128.u32", 0)}')
        print('  ' + '，'.join(f'{mnemonic} {count}' for mnemonic, count in sorted(compute.items(), key=lambda item: -item[1])[:10]))
        first = values[0].astype(np.float64)
        print(f'  均值 {first.mean():.4f}，标准差 {first.std():.4f}，最小 {first.min():.6g}，最大 {first.max():.6g}')
        if name == '手写均匀分布':
            print(f'  全部是 2^-23 的整数倍：{bool(np.all(values[0] * 2.0**23 == np.round(values[0] * 2.0**23)))}')
        if name.startswith('Gumbel'):
            reference = -np.log(-np.log(values[0].astype(np.float64)))
            finite = np.isfinite(reference)
            print(f'  Gumbel：与主机 float64 公式的最大绝对误差 {np.max(np.abs(values[1][finite] - reference[finite])):.2e}，均值 {values[1].mean():.4f}（理论值 0.5772）')
        if name.startswith('手写随机舍入'):
            print(f'  向上舍入到 1 + 2^-7 的比例 {np.mean(values[0] > 1.0):.4f}（理论值 0.125），其余都是 1.0：{bool(np.all((values[0] == 1.0) | (values[0] == 1.0 + 2.0**-7)))}')

if __name__ == '__main__':
    main()
