"""把 TC VMEM 中的 bf16[16,128] 按位重新解释为 u32[8,128]，确定两个 bf16 元素怎样共用一个 32 bit 位置。"""
import tpu_init
tpu_init.initialise_one_chip()

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
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def reinterpret(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct((8, 128), jnp.uint32),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM((16, 128), jnp.bfloat16), pltpu.VMEM((8, 128), jnp.uint32), pltpu.SemaphoreType.DMA),
            name='reinterpret',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, o_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            o_vmem[...] = pltpu.bitcast(x_vmem[...], jnp.uint32)
            pltpu.async_copy(o_vmem, o_hbm, sem).wait()

        return kernel(x)

    # 元素值编码自己的行号和列号：第 r 行第 c 列的 bf16 位型为 r * 256 + c（c < 128），互不相同。
    rows, columns = np.meshgrid(np.arange(16), np.arange(128), indexing='ij')
    bits = (rows * 256 + columns).astype(np.uint16)
    x = jnp.asarray(bits.view(jnp.bfloat16))
    compiled = tpuasm_tools.compile(reinterpret, x, mesh=mesh)
    words = np.asarray(compiled(x))
    low = (words & 0xffff).astype(np.uint16)
    high = (words >> 16).astype(np.uint16)
    print('u32[s, l] 的低 16 bit 来自 bf16 的第几行：', sorted(set((low // 256)[s, 0] for s in range(8))), '按 sublane s =', [int(low[s, 0] // 256) for s in range(8)])
    print('u32[s, l] 的高 16 bit 来自 bf16 的第几行：按 sublane s =', [int(high[s, 0] // 256) for s in range(8)])
    print('列号是否不变：', bool(np.all(low % 256 == columns[:8]) and np.all(high % 256 == columns[:8])))
    listing = tpuasm_tools.kernel_listing(compiled)
    tpuasm_tools.print_mnemonic_counts(listing)
    print(listing)

if __name__ == '__main__':
    main()
