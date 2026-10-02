"""一个 f32[8,128] tile 内的数据重排：按索引的 lane gather、按索引的 sublane gather、固定位移的 lane/sublane 循环移位。"""
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

SKIPPED = ('vld', 'vst', 'vsync', 'vwait', 'vtrace', 'cfence', 'dma.', 's', 'p')

def run(f: Callable[[jax.Array, jax.Array], jax.Array], x: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, str]:
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=(P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    def function(x: jax.Array, indices: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.VMEM(indices.shape, indices.dtype), pltpu.SemaphoreType.DMA),
            name='permute',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, i_hbm: Ref, o_hbm: Ref, x_vmem: Ref, i_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            pltpu.async_copy(i_hbm, i_vmem, sem).wait()
            x_vmem[...] = f(x_vmem[...], i_vmem[...])
            pltpu.async_copy(x_vmem, o_hbm, sem).wait()

        return kernel(x, indices)

    compiled = tpuasm_tools.compile(function, x, indices, mesh=mesh)
    return np.asarray(compiled(x, indices)), tpuasm_tools.kernel_listing(compiled)

def sublane_gather_by_rotation(x: jax.Array, indices: jax.Array) -> jax.Array:
    """用 8 次“比较并选择”合成 sublane gather：第 k 轮的候选是 x 向上循环移动 k 行的结果。"""
    sublane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 0)
    result = jnp.zeros_like(x)
    rotated = x
    for k in range(8):
        # 此时 rotated[s, l] = x[(s + k) % 8, l]。
        result = jnp.where(indices == (sublane + k) % 8, rotated, result)
        rotated = pltpu.roll(rotated, 7, axis=0)
    return result

def main() -> None:
    rng = np.random.default_rng(0)
    x = np.arange(8 * 128, dtype=np.float32).reshape(8, 128)
    lane_indices = rng.integers(0, 128, (8, 128)).astype(np.int32)
    sublane_indices = rng.integers(0, 8, (8, 128)).astype(np.int32)
    cases = (
        ('lane gather：y[s,l] = x[s, i[s,l]]', lambda x, i: jnp.take_along_axis(x, i, axis=1), lane_indices, np.take_along_axis(x, lane_indices, axis=1)),
        ('sublane gather：y[s,l] = x[i[s,l], l]', lambda x, i: jnp.take_along_axis(x, i, axis=0), sublane_indices, np.take_along_axis(x, sublane_indices, axis=0)),
        ('用循环移位与选择合成 sublane gather', sublane_gather_by_rotation, sublane_indices, np.take_along_axis(x, sublane_indices, axis=0)),
        ('lane 循环移位 5 位：pltpu.roll(x, 5, axis=1)', lambda x, i: pltpu.roll(x, 5, axis=1), lane_indices, np.roll(x, 5, axis=1)),
        ('sublane 循环移位 3 位：pltpu.roll(x, 3, axis=0)', lambda x, i: pltpu.roll(x, 3, axis=0), lane_indices, np.roll(x, 3, axis=0)),
    )
    for name, f, indices, expected in cases:
        try:
            result, listing = run(f, x, indices)
        except Exception as error:
            print(f'## {name}：编译失败')
            print(str(error).splitlines()[0][:300])
            print()
            continue
        np.testing.assert_array_equal(result, expected)
        counts = tpuasm_tools.count_mnemonics(listing)
        print(f'## {name}：数值检查通过')
        print('计算指令：', '、'.join(f'{mnemonic}×{count}' for mnemonic, count in sorted(counts.items()) if not mnemonic.startswith(SKIPPED)))
        print('\n'.join(line for line in listing.splitlines() if any(f'{slot}: ' in line for slot in ('va0', 'va1', 'vx0', 'vx1', 'vr0', 'vr1'))))
        print()

if __name__ == '__main__':
    main()
