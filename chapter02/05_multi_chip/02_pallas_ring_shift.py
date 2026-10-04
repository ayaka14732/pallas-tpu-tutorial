"""四颗芯片组成环，每颗芯片的 TensorCore 0 把 f32[8,128] 用 remote DMA 发给环上的下一颗。比较按 jax.make_mesh 顺序成环与按物理相邻顺序成环。"""
import tpu_init
tpu_init.initialise_local_chips()

import re

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

import tpuasm_tools

def build(mesh: jax.sharding.Mesh, rounds: int):
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P('device'),
        out_specs=P('device'),
        check_vma=False,
    )
    def shift(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=pltpu.HBM(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM((8, 128), x.dtype), pltpu.VMEM((8, 128), x.dtype), pltpu.SemaphoreType.DMA((3,))),
            name='shift',
            compiler_params=pltpu.CompilerParams(
                collective_id=1,
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, send_vmem: Ref, recv_vmem: Ref, sems: Ref) -> None:
            # 在 shard_map 中，axis_index('device') 是本芯片在 mesh 中的位置。
            me = jax.lax.axis_index('device')
            pltpu.async_copy(x_hbm.at[0], send_vmem, sems.at[0]).wait()
            ready = pltpu.get_barrier_semaphore()

            @pl.loop(0, rounds)
            def _(_: jax.Array) -> None:
                # 每一轮先让四颗芯片汇合，确认上一轮的数据都已取走，再发送。
                for rank in range(4):
                    pl.semaphore_signal(ready, 1, device_id={'device': rank, 'tc': 0})
                pl.semaphore_wait(ready, 4)
                transfer = pltpu.make_async_remote_copy(send_vmem, recv_vmem, sems.at[1], sems.at[2], device_id={'device': (me + 1) % 4, 'tc': 0}, device_id_type=pl.DeviceIdType.MESH)
                transfer.start()
                transfer.wait_send()
                transfer.wait_recv()

            pltpu.async_copy(recv_vmem, o_hbm.at[0], sems.at[0]).wait()

        return kernel(x)

    return shift

def round_cycles(clock: tpuasm_tools.KernelClock, mesh: jax.sharding.Mesh, x: jax.Array) -> float:
    """每一轮的周期数：kernel 内部 64 轮与 32 轮，各用 LCC 读出 kernel 在 device 0 上的周期数（第三章第 3 节的 KernelClock），相减除以 32。"""
    cycles = {}
    for rounds in (32, 64):
        compiled = tpuasm_tools.compile(build(mesh, rounds), x, mesh=mesh)
        cycles[rounds] = clock.kernel_cycles(compiled, lambda timed: jax.block_until_ready(timed(x)))
    return (cycles[64] - cycles[32]) / 32

def main() -> None:
    devices = {device.id: device for device in jax.devices()}
    orders = {
        'jax.make_mesh 的顺序': [device.id for device in jax.make_mesh((4,), ('device',)).devices.flat],
        '物理环的顺序': [0, 1, 3, 2],
    }
    clock = tpuasm_tools.KernelClock(num_cores=1)
    for name, order in orders.items():
        mesh = jax.sharding.Mesh(np.array([devices[i] for i in order]), ('device',))
        coords = [tuple(devices[i].coords[:2]) for i in order]
        hops = [abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(coords, coords[1:] + coords[:1])]
        x = jax.device_put(jnp.arange(4 * 8 * 128, dtype=jnp.float32).reshape(4, 8, 128), jax.NamedSharding(mesh, P('device')))
        compiled = tpuasm_tools.compile(build(mesh, 1), x, mesh=mesh)
        np.testing.assert_array_equal(np.asarray(compiled(x)), np.roll(np.asarray(x), 1, axis=0))
        print(f'## {name}：mesh 中依次为 device {order}，坐标 {coords}；环上每一步的跳数 {hops}')
        print(f'  一轮数值检查通过；每轮 {round_cycles(clock, mesh, x):.0f} 个周期')
    print('## 物理环、1 轮时的 kernel 段清单（去掉源码注释与编码约束）')
    listing = tpuasm_tools.kernel_listing(compiled, pallas_only=True)
    text = re.sub(r'\s*;\s*\.encoding \{[^}]*\}', '', '\n'.join(line.split('#')[0].rstrip() for line in listing.splitlines()))
    print('\n'.join(line for line in text.splitlines() if line.strip()))

if __name__ == '__main__':
    main()
