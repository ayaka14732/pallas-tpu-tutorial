"""拼接两个数组：在 TC VMEM 中用 jnp.concatenate，或不经 TC VMEM、让两次 DMA 直接写进输出的两个窗口；对齐与不对齐的 shape 各一组。"""
import tpu_init
tpu_init.initialize_one_chip()

import re

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
SKIPPED = ('vsync', 'vwait', 'vtrace', 'cfence', 's', 'p')

def build(style: str, shape: tuple[int, int], axis: int):
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)
    out_shape = (shape[0] * 2, shape[1]) if axis == 0 else (shape[0], shape[1] * 2)

    @jax.shard_map(
        mesh=mesh,
        in_specs=(P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    def in_vmem(a: jax.Array, b: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct(out_shape, a.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM(shape, a.dtype), pltpu.VMEM(shape, a.dtype), pltpu.VMEM(out_shape, a.dtype), pltpu.SemaphoreType.DMA),
            name='concatenate',
            compiler_params=PARAMS,
        )
        def kernel(a_hbm: Ref, b_hbm: Ref, o_hbm: Ref, a_vmem: Ref, b_vmem: Ref, o_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(a_hbm, a_vmem, sem).wait()
            pltpu.async_copy(b_hbm, b_vmem, sem).wait()
            o_vmem[...] = jnp.concatenate([a_vmem[...], b_vmem[...]], axis=axis)
            pltpu.async_copy(o_vmem, o_hbm, sem).wait()

        return kernel(a, b)

    @jax.shard_map(
        mesh=mesh,
        in_specs=(P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    def by_dma(a: jax.Array, b: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct(out_shape, a.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.SemaphoreType.DMA,),
            name='concatenate',
            compiler_params=PARAMS,
        )
        def kernel(a_hbm: Ref, b_hbm: Ref, o_hbm: Ref, sem: Ref) -> None:
            rows, columns = shape
            if axis == 0:
                windows = (o_hbm.at[pl.ds(0, rows), :], o_hbm.at[pl.ds(rows, rows), :])
            else:
                windows = (o_hbm.at[:, pl.ds(0, columns)], o_hbm.at[:, pl.ds(columns, columns)])
            # 两次 DMA 直接从输入 HBM 写到输出 HBM 的两个窗口，然后一起等待。
            copies = [pltpu.async_copy(a_hbm, windows[0], sem), pltpu.async_copy(b_hbm, windows[1], sem)]
            for copy in copies:
                copy.wait()

        return kernel(a, b)

    return mesh, in_vmem if style == 'TC VMEM 中 jnp.concatenate' else by_dma

def main() -> None:
    cases = (((8, 128), 0), ((8, 128), 1), ((9, 128), 0), ((8, 129), 1))
    for shape, axis in cases:
        a = jnp.arange(shape[0] * shape[1], dtype=jnp.float32).reshape(shape)
        b = -a - 1
        for style in ('TC VMEM 中 jnp.concatenate', 'DMA 写入输出窗口'):
            name = f'f32{list(shape)} × 2，axis={axis}，{style}'
            mesh, function = build(style, shape, axis)
            try:
                compiled = tpuasm_tools.compile(function, a, b, mesh=mesh)
            except Exception as error:
                print(f'## {name}：编译失败：{str(error).splitlines()[0][:250]}')
                print()
                continue
            np.testing.assert_array_equal(np.asarray(compiled(a, b)), np.concatenate([np.asarray(a), np.asarray(b)], axis=axis))
            listing = tpuasm_tools.kernel_listing(compiled, pallas_only=True)
            counts = tpuasm_tools.count_mnemonics(listing)
            vector = '、'.join(f'{mnemonic}×{count}' for mnemonic, count in sorted(counts.items()) if mnemonic.startswith('v') and not mnemonic.startswith(SKIPPED[:3]))
            print(f'## {name}：数值检查通过')
            print(f'  向量指令：{vector or "无"}')
            print('\n'.join('  ' + line.split('#')[0].strip() for line in listing.splitlines() if 'dma.' in line))
            if shape != (8, 128) and style.startswith('TC VMEM'):
                # 不对齐时，列出含向量运算、读写 TC VMEM 的 bundle。
                text = re.sub(r'\s*;\s*\.encoding \{[^}]*\}', '', ' '.join(line.split('#')[0].strip() for line in listing.splitlines() if not line.startswith('#')))
                print('  计算部分的清单：')
                print('\n'.join('    { ' + body.strip() + ' }' for body in re.findall(r'\{(.*?)\}', text) if re.search(r'\b(va0|va1|vld|vst|vx0|vx1|vr0|vr1):', body)))
            print()

if __name__ == '__main__':
    main()
