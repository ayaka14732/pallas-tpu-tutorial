"""按行 scatter：把 8 行更新写进 HBM 中 f32[64,128] 的指定行，行号在 SMEM 中，每行一次 DMA；数组作为 jax Ref 原地修改。检查行号重复时谁的写入留下，以及在 TC VREG 内按 lane 做 scatter 能否编译。"""
import tpu_init
tpu_init.initialize_one_chip()

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

import tpuasm_tools

PARAMS = pltpu.CompilerParams(
    disable_bounds_checks=True,
    disable_semaphore_checks=True,
)

def build_row_scatter():
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @pl.kernel(
        out_type=(),
        mesh=tc_mesh,
        scratch_types=(pltpu.VMEM((8, 128), jnp.float32), pltpu.SMEM((8,), jnp.int32), pltpu.SemaphoreType.DMA),
        name='scatter_rows',
        compiler_params=PARAMS,
    )
    def kernel(u_hbm: Ref, r_hbm: Ref, t_hbm: Ref, u_vmem: Ref, r_smem: Ref, sem: Ref) -> None:
        pltpu.async_copy(u_hbm, u_vmem, sem).wait()
        pltpu.async_copy(r_hbm, r_smem, sem).wait()
        # 与按行 gather 对称：源是更新的第 j 行，目的地是表的第 r[j] 行。8 次 DMA 先全部发出，再一起等待。
        copies = [pltpu.async_copy(u_vmem.at[pl.ds(j, 1)], t_hbm.at[pl.ds(r_smem[j], 1)], sem) for j in range(8)]
        for copy in copies:
            copy.wait()

    def scatter_rows(table: Ref, updates: jax.Array, rows: jax.Array) -> None:
        kernel(updates, rows, table)

    return scatter_rows

def lane_kernel(scatter_by_set: bool):
    """在 TC VREG 内按 lane 重排。scatter_by_set 为 True 时直接写 y[s, i[s,l]] = x[s,l]；否则按传入的逆排列 gather。"""
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=(P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    def scatter(x: jax.Array, i: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.VMEM(i.shape, i.dtype), pltpu.SemaphoreType.DMA),
            name='lane_scatter',
            compiler_params=PARAMS,
        )
        def kernel(x_hbm: Ref, i_hbm: Ref, o_hbm: Ref, x_vmem: Ref, i_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            pltpu.async_copy(i_hbm, i_vmem, sem).wait()
            if scatter_by_set:
                rows = jax.lax.broadcasted_iota(jnp.int32, x.shape, 0)
                x_vmem[...] = jnp.zeros(x.shape, x.dtype).at[rows, i_vmem[...]].set(x_vmem[...])
            else:
                x_vmem[...] = jnp.take_along_axis(x_vmem[...], i_vmem[...], axis=1)
            pltpu.async_copy(x_vmem, o_hbm, sem).wait()

        return kernel(x, i)

    return mesh, scatter

def main() -> None:
    scatter_rows = build_row_scatter()
    updates = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128) + 1000
    print('## 按行 scatter：行号互不相同')
    rows = jnp.array([5, 3, 60, 0, 17, 42, 9, 63], jnp.int32)
    table = jax.new_ref(jnp.zeros((64, 128), jnp.float32))
    compiled = tpuasm_tools.compile(scatter_rows, table, updates, rows)
    compiled(table, updates, rows)
    expected = np.zeros((64, 128), np.float32)
    expected[np.asarray(rows)] = np.asarray(updates)
    print(f'  行号 {np.asarray(rows).tolist()}：数值检查 {bool(np.array_equal(np.asarray(table[...]), expected))}')
    listing = tpuasm_tools.kernel_listing(compiled, pallas_only=True)
    print('\n'.join('  ' + line.split('#')[0].strip() for line in listing.splitlines() if 'dma.' in line or 'vwait' in line))
    print('## 按行 scatter：第 4、5 个更新都写到第 17 行，运行 20 次')
    rows = jnp.array([5, 3, 60, 0, 17, 17, 9, 63], jnp.int32)
    winners = {}
    for _ in range(20):
        table = jax.new_ref(jnp.zeros((64, 128), jnp.float32))
        compiled(table, updates, rows)
        row = np.asarray(table[...])[17]
        winner = next((j for j in (4, 5) if np.array_equal(row, np.asarray(updates)[j])), '混合')
        winners[winner] = winners.get(winner, 0) + 1
    print(f'  第 17 行最终等于哪个更新：{winners}')
    print('## 在 TC VREG 内按 lane scatter：每行的索引是 0–127 的一个排列')
    x = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
    permutation = np.stack([np.random.default_rng(row).permutation(128) for row in range(8)]).astype(np.int32)
    expected = np.zeros((8, 128), np.float32)
    np.put_along_axis(expected, permutation, np.asarray(x), axis=1)
    mesh, scatter = lane_kernel(True)
    try:
        tpuasm_tools.compile(scatter, x, jnp.asarray(permutation), mesh=mesh)
        print('  直接写 .at[].set()：编译成功')
    except Exception as error:
        print(f'  直接写 .at[].set()：编译失败：{str(error).splitlines()[0][:120]}')
    # 排列没有冲突：inverse[s, i[s,l]] = l，scatter 变成按 inverse 的 gather。
    inverse = np.argsort(permutation, axis=1).astype(np.int32)
    mesh, gather = lane_kernel(False)
    compiled = tpuasm_tools.compile(gather, x, jnp.asarray(inverse), mesh=mesh)
    print(f'  按逆排列 gather：与 NumPy 的 scatter 结果相同 {bool(np.array_equal(np.asarray(compiled(x, jnp.asarray(inverse))), expected))}')
    counts = tpuasm_tools.count_mnemonics(tpuasm_tools.kernel_listing(compiled, pallas_only=True))
    print('  计算指令：' + '，'.join(f'{name} {count}' for name, count in sorted(counts.items()) if name.startswith(('vperm', 'vsetperm', 'vpop', 'vsel', 'vlt', 'vadd'))))

if __name__ == '__main__':
    main()
