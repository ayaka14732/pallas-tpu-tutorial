"""一颗芯片的两个 TensorCore 交换数据：每个 TensorCore 把自己 TC VMEM 中的 f32[8,128] 用 remote DMA 直接写进对方的 TC VMEM。"""
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
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=2)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def exchange(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=pltpu.HBM(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(
                pltpu.VMEM((8, 128), x.dtype),
                pltpu.VMEM((8, 128), x.dtype),
                pltpu.SemaphoreType.DMA((4,)),
            ),
            name='exchange',
            compiler_params=pltpu.CompilerParams(
                collective_id=1,
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, send_vmem: Ref, recv_vmem: Ref, sems: Ref) -> None:
            core = jax.lax.axis_index('tc')
            pltpu.async_copy(x_hbm.at[core], send_vmem, sems.at[0]).wait()

            # 一：两个 TensorCore 都准备好接收之后，才允许向对方写。
            ready = pltpu.get_barrier_semaphore()
            pl.semaphore_signal(ready, 1, device_id={'tc': 1 - core})
            pl.semaphore_wait(ready, 1)

            # 二：把自己的 send_vmem 写进对方的 recv_vmem。
            transfer = pltpu.make_async_remote_copy(send_vmem, recv_vmem, sems.at[1], sems.at[2], device_id={'tc': 1 - core}, device_id_type=pl.DeviceIdType.MESH)
            transfer.start()
            transfer.wait_send()
            transfer.wait_recv()
            pltpu.async_copy(recv_vmem, o_hbm.at[core], sems.at[3]).wait()

        return kernel(x)

    x = jnp.arange(2 * 8 * 128, dtype=jnp.float32).reshape(2, 8, 128)
    compiled = tpuasm_tools.compile(exchange, x, mesh=mesh)
    np.testing.assert_array_equal(np.asarray(compiled(x)), np.asarray(x)[::-1])
    print('数值检查通过：输出的第 0 块来自 TensorCore 1，第 1 块来自 TensorCore 0')
    listing = tpuasm_tools.kernel_listing(compiled, pallas_only=True)
    counts = tpuasm_tools.count_mnemonics(listing)
    print(f'dma.general {counts["dma.general"]}、dma.simple {counts["dma.simple"]}、vsyncadd.remote {counts["vsyncadd.remote.s32"]}、vwait.ge {counts["vwait.ge"]}')
    print(listing)

if __name__ == '__main__':
    main()
