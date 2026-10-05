"""按行 gather：用 SMEM 中的 8 个行号，从 HBM 中的 f32[64,128] 各取一行，拼成 f32[8,128]；每行一次 DMA。"""
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

def main() -> None:
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=(P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    def gather_rows(table: jax.Array, rows: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct((8, 128), table.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM((8, 128), table.dtype), pltpu.SMEM(rows.shape, rows.dtype), pltpu.SemaphoreType.DMA),
            name='gather_rows',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(t_hbm: Ref, r_hbm: Ref, o_hbm: Ref, o_vmem: Ref, r_smem: Ref, sem: Ref) -> None:
            pltpu.async_copy(r_hbm, r_smem, sem).wait()
            # 8 次 DMA 先全部发出，再一次等待：它们共用一个信号量，等待阈值合计 8 个 granule。
            copies = [pltpu.async_copy(t_hbm.at[pl.ds(r_smem[j], 1)], o_vmem.at[pl.ds(j, 1)], sem) for j in range(8)]
            for copy in copies:
                copy.wait()
            pltpu.async_copy(o_vmem, o_hbm, sem).wait()

        return kernel(table, rows)

    table = jnp.arange(64 * 128, dtype=jnp.float32).reshape(64, 128)
    rows = jnp.array([5, 3, 60, 0, 17, 17, 42, 9], jnp.int32)
    compiled = tpuasm_tools.compile(gather_rows, table, rows, mesh=mesh)
    np.testing.assert_array_equal(np.asarray(compiled(table, rows)), np.asarray(table)[np.asarray(rows)])
    print('行号', np.asarray(rows).tolist(), '：数值检查通过')
    listing = tpuasm_tools.kernel_listing(compiled)
    print('\n'.join(line for line in listing.splitlines() if any(key in line for key in ('dma.simple', 'vwait', 'sld'))))

if __name__ == '__main__':
    main()
