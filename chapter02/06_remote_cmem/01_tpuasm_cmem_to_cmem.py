"""跨芯片的 Megacore Shared CMEM → Megacore Shared CMEM：以两颗相邻芯片交换 TC VMEM 数据的 kernel 为载体，用 tpuasm 把 remote DMA 的两端改成 CMEM，并修改 ici_dest 中的 TensorCore 字段。"""
import tpu_init
tpu_init.initialise_local_chips()

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

import tpuasm_tools

# 一：发送前，把要发送的 TC VMEM 数据（s9）搬到本芯片 CMEM 地址 0；CMEM 地址 64 作为接收区。
STAGE = '''{ s0: simm.s32 s25, 0 ; s1: simm.s32 s26, 64 }
{ s0: dma.simple [cmem:s25], [vmem:s9], length=8, dst_flag=[sflag:52] }
{ misc: vwait.ge [sflag:52], 8 }
{ misc: vsyncadd.s32 [sflag:52], -8 }'''
# 三：收到之后，把 CMEM 接收区搬回原来的 TC VMEM 接收 buffer（s23），沿原路径写回 HBM。
UNSTAGE = '''{ s0: dma.simple [vmem:s23], [cmem:s26], length=8, dst_flag=[sflag:52] }
{ misc: vwait.ge [sflag:52], 8 }
{ misc: vsyncadd.s32 [sflag:52], -8 }'''

def build():
    """载体：两颗相邻芯片（device 0 与 1，第 5 节）交换 TC VMEM 中的 f32[8,128]。"""
    devices = {device.id: device for device in jax.devices()}
    mesh = jax.sharding.Mesh(np.array([devices[0], devices[1]]), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P('device'),
        out_specs=P('device'),
        check_vma=False,
    )
    def exchange(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=pltpu.HBM(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM((8, 128), x.dtype), pltpu.VMEM((8, 128), x.dtype), pltpu.SemaphoreType.DMA((3,))),
            name='exchange',
            compiler_params=pltpu.CompilerParams(
                collective_id=1,
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, send_vmem: Ref, recv_vmem: Ref, sems: Ref) -> None:
            me = jax.lax.axis_index('device')
            pltpu.async_copy(x_hbm.at[0], send_vmem, sems.at[0]).wait()
            ready = pltpu.get_barrier_semaphore()
            pl.semaphore_signal(ready, 1, device_id={'device': 1 - me, 'tc': 0})
            pl.semaphore_wait(ready, 1)
            transfer = pltpu.make_async_remote_copy(send_vmem, recv_vmem, sems.at[1], sems.at[2], device_id={'device': 1 - me, 'tc': 0}, device_id_type=pl.DeviceIdType.MESH)
            transfer.start()
            transfer.wait_send()
            transfer.wait_recv()
            pltpu.async_copy(recv_vmem, o_hbm.at[0], sems.at[0]).wait()

        return kernel(x)

    return exchange, mesh

def main() -> None:
    exchange, mesh = build()
    x = jax.device_put(jnp.arange(2 * 8 * 128, dtype=jnp.float32).reshape(2, 8, 128), jax.NamedSharding(mesh, P('device')))
    compiled = tpuasm_tools.compile(exchange, x, mesh=mesh)
    np.testing.assert_array_equal(np.asarray(compiled(x)), np.asarray(x)[::-1])
    serialized = tpuasm_tools.serialize(compiled)
    listing = tpuasm_tools.kernel_listing(serialized, pallas_only=True)
    print('## 载体：两颗芯片交换 TC VMEM 中的数据，数值检查通过')
    print('\n'.join(line.split('#')[0].rstrip() for line in listing.splitlines() if 'dma.general' in line or '0x88008000' in line))

    remote_pc, = tpuasm_tools.find_bundles(serialized, 'dma.general')
    barrier_pc = tpuasm_tools.find_bundles(serialized, 'vsyncadd.remote.s32')[0]
    output_pc, = tpuasm_tools.find_bundles(serialized, 'dma.simple [hbm:s1], [vmem:s23]')
    ici_pc, = tpuasm_tools.find_bundles(serialized, 'sor.u32 s21, 0x88008000, s17')
    # 二：remote DMA 的两端改为 CMEM；ici_dest 中的 TensorCore 字段（bits 28:26）清零，表示目的地不是某个 TensorCore。
    patched = tpuasm_tools.edit_bundles(serialized, {
        remote_pc: ('dma.general [vmem:s23], [vmem:s9]', 'dma.general [cmem:s26], [cmem:s25]'),
        ici_pc: ('sor.u32 s21, 0x88008000, s17', 'sor.u32 s21, 0x80008000, s17'),
    })
    patched = tpuasm_tools.insert_bundles(patched, {barrier_pc: STAGE, output_pc: UNSTAGE})
    function = tpuasm_tools.load(patched, compiled)
    for trial in range(20):
        value = x + trial
        np.testing.assert_array_equal(np.asarray(function(value)), np.asarray(value)[::-1])
    print('\n## 改写后：数据经 TC VMEM → 本芯片 CMEM → 对方 CMEM → 对方 TC VMEM，20 组输入的数值检查全部通过')
    print('\n'.join(line.split('#')[0].rstrip() for line in tpuasm_tools.kernel_listing(patched, pallas_only=True).splitlines() if 'dma.' in line or '0x80008000' in line))

if __name__ == '__main__':
    main()
