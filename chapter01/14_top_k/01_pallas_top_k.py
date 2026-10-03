"""每行取最大的 k 个值及其下标：argmax、top-1、top-8，跨两个 lane tile 的 top-8，bf16 输入，以及有效值少于 k 个的边界情况。"""
import tpu_init
tpu_init.initialise_one_chip()

from collections.abc import Callable

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import ml_dtypes
import numpy as np

import tpuasm_tools

SKIPPED = ('vld', 'vst', 'vsync', 'vwait', 'vtrace', 'cfence', 'dma.', 's', 'p')

def run(f: Callable[[jax.Array], tuple[jax.Array, jax.Array]], x: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], str]:
    """f(x) 返回 (values, indices)；kernel 有两个输出。"""
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)
    values, indices = jax.eval_shape(f, jax.ShapeDtypeStruct(x.shape, x.dtype))

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=(P(), P()),
        check_vma=False,
    )
    def function(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        @pl.kernel(
            out_type=(pltpu.HBM(values.shape, values.dtype), pltpu.HBM(indices.shape, indices.dtype)),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.VMEM(values.shape, values.dtype), pltpu.VMEM(indices.shape, indices.dtype), pltpu.SemaphoreType.DMA),
            name='top_k',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, v_hbm: Ref, i_hbm: Ref, x_vmem: Ref, v_vmem: Ref, i_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            v_vmem[...], i_vmem[...] = f(x_vmem[...])
            pltpu.async_copy(v_vmem, v_hbm, sem).wait()
            pltpu.async_copy(i_vmem, i_hbm, sem).wait()

        return kernel(x)

    compiled = tpuasm_tools.compile(function, x, mesh=mesh)
    values, indices = compiled(x)
    return (np.asarray(values), np.asarray(indices)), tpuasm_tools.kernel_listing(compiled, pallas_only=True)

def argmax(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return jnp.max(x, axis=1, keepdims=True), jnp.argmax(x, axis=1, keepdims=True).astype(jnp.int32)

def top_k_by_hand(k: int) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
    """每轮取未选位置中的最大值，再在等于它的未选位置中取最小的下标；已选位置用单独的掩码排除，不靠把值改成 -inf。"""
    def f(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1).astype(jnp.float32)
        taken = jnp.zeros(x.shape, jnp.bool_)
        values, indices = [], []
        for _ in range(k):
            best = jnp.max(jnp.where(taken, -jnp.inf, x), axis=1, keepdims=True)
            index = jnp.min(jnp.where(~taken & (x == best), lane, float(x.shape[1])), axis=1, keepdims=True)
            taken = taken | (lane == index)
            values.append(best)
            indices.append(index.astype(jnp.int32))
        return jnp.concatenate(values, axis=1), jnp.concatenate(indices, axis=1)

    return f

def main() -> None:
    rng = np.random.default_rng(0)
    # 每行是互不相同的值的随机排列，避免并列。
    x128 = np.stack([rng.permutation(128) for _ in range(8)]).astype(np.float32)
    x256 = np.stack([rng.permutation(256) for _ in range(8)]).astype(np.float32)
    sparse = np.full((8, 128), -np.inf, np.float32)
    sparse[:, :3] = x128[:, :3]
    cases = (
        ('max 与 argmax，f32[8,128]', argmax, x128, 1),
        ('lax.top_k(x, 1)，f32[8,128]', lambda x: jax.lax.top_k(x, 1), x128, 1),
        ('lax.top_k(x, 1, is_stable=False)，f32[8,128]', lambda x: jax.lax.top_k(x, 1, is_stable=False), x128, 1),
        ('lax.top_k(x, 8, is_stable=False)，f32[8,128]', lambda x: jax.lax.top_k(x, 8, is_stable=False), x128, 8),
        ('lax.top_k(x, 8, is_stable=False)，f32[8,256]', lambda x: jax.lax.top_k(x, 8, is_stable=False), x256, 8),
        ('lax.top_k(x, 8, is_stable=False)，bf16[8,128]', lambda x: jax.lax.top_k(x, 8, is_stable=False), x128.astype(ml_dtypes.bfloat16), 8),
        ('lax.top_k(x, 8, is_stable=False)，每行只有 3 个有限值，其余为 -inf', lambda x: jax.lax.top_k(x, 8, is_stable=False), sparse, 8),
        ('手写 top-8，用掩码排除已选位置，f32[8,128]', top_k_by_hand(8), x128, 8),
        ('手写 top-8，用掩码排除已选位置，每行只有 3 个有限值，其余为 -inf', top_k_by_hand(8), sparse, 8),
    )
    for name, f, x, k in cases:
        try:
            (values, indices), listing = run(f, x)
        except Exception as error:
            print(f'## {name}：编译失败：{str(error).splitlines()[0][:250]}')
            print()
            continue
        expected_values, expected_indices = (np.asarray(array) for array in jax.jit(jax.lax.top_k, static_argnums=1, backend='cpu')(x, k))
        print(f'## {name}')
        print(f'  值与 CPU 结果一致：{bool(np.array_equal(values, expected_values))}；下标与 CPU 结果一致：{bool(np.array_equal(indices, expected_indices))}')
        if not np.array_equal(indices, expected_indices):
            print(f'  第 0 行下标：{indices[0].tolist()}；CPU：{expected_indices[0].tolist()}')
        counts = tpuasm_tools.count_mnemonics(listing)
        print('  计算指令：' + '、'.join(f'{mnemonic}×{count}' for mnemonic, count in sorted(counts.items()) if not mnemonic.startswith(SKIPPED)))
        print()

if __name__ == '__main__':
    main()
