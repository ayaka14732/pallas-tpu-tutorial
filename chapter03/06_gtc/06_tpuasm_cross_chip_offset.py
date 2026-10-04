"""不同芯片之间的 GTC 偏移：在 TPU v4 (2x2x2) 上，一颗芯片（发起方）给另一颗（应答方）发信号，应答方收到后回信，两边在发出之前、收到之后各读一次 GTC。由两个方向读数差的最小值估计偏移，并与只用因果顺序得到的区间比较；device 0 与每颗芯片 t 各做两遍，分别由 device 0 和 t 发起。运行方式：podrun -- /srv/workspace/venv/bin/python 本文件。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
from jax import Ref
from jax.experimental import multihost_utils
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

RUNS = 300
CHIPS = 8

def build(mesh: jax.sharding.Mesh, initiator: int, responder: int):
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P('device'),
        out_specs=P('device'),
        check_vma=False,
    )
    def exchange(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.REGULAR),
            name='exchange',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
                collective_id=1,
            ),
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, vmem: Ref, sem: Ref, ping: Ref) -> None:
            me = jax.lax.axis_index('device')
            pltpu.async_copy(x_hbm, vmem, sem).wait()
            ready = pltpu.get_barrier_semaphore()
            for rank in range(CHIPS):
                pl.semaphore_signal(ready, 1, device_id={'device': rank, 'tc': 0})
            pl.semaphore_wait(ready, CHIPS)

            @pl.when(me == initiator)
            def _() -> None:
                # 先停 20 µs，让应答方已经在等待；再发信号，等回信。
                pl.delay(20000)
                pl.semaphore_signal(ping, 1, device_id={'device': responder, 'tc': 0})
                pl.semaphore_wait(ping, 1)
                # 只用来标出这个分支的末尾，供插入读数时定位。
                pl.delay(8)

            @pl.when(me == responder)
            def _() -> None:
                pl.semaphore_wait(ping, 1)
                pl.semaphore_signal(ping, 1, device_id={'device': initiator, 'tc': 0})

            pltpu.async_copy(vmem, o_hbm, sem).wait()

        return kernel(x)

    return exchange

def main() -> None:
    jax.distributed.initialize()
    import tpuasm_tools

    mesh = jax.sharding.Mesh(np.array(jax.devices()), ('device',))
    sharding = jax.NamedSharding(mesh, P('device'))
    x = jax.make_array_from_process_local_data(sharding, np.zeros((len(jax.local_devices()), 8, 128), np.float32))
    clocks = [tpuasm_tools.KernelClock(num_cores=1, device=index) for index in range(len(jax.local_devices()))]
    # readings[本进程的第几颗芯片, 由谁发起（0：device 0，1：t）, t, 第几次运行, 读数 0/1 的低、高 32 位]
    readings = np.zeros((len(clocks), 2, CHIPS, RUNS, 4), np.uint32)
    for target in range(1, CHIPS):
        for swapped, (initiator, responder) in enumerate(((0, target), (target, 0))):
            compiled = tpuasm_tools.compile(build(mesh, initiator, responder), x, mesh=mesh)
            serialized = tpuasm_tools.serialize(compiled)
            remote = tpuasm_tools.find_bundles(serialized, 'vsyncadd.remote.s32')
            delays = tpuasm_tools.find_bundles(serialized, 'vdelay')
            # kernel 中的 vsyncadd.remote 依次是：汇合的若干条、发起方的信号、应答方的回信、出口汇合的一条。
            send, reply = remote[-3], remote[-2]
            gtc = lambda index: tpuasm_tools.clock_read(index, 'gtc')
            # 发起方：发信号之前读 A0，等到回信之后读 A3。应答方：等到信号之后读 B1，回信之前读 B2。
            timed = tpuasm_tools.load(tpuasm_tools.insert_bundles(serialized, {send: gtc(0), delays[1]: gtc(1), reply - 1: gtc(0), reply: gtc(1)}), compiled, jax.devices())
            jax.block_until_ready(timed(x))
            for run in range(RUNS):
                jax.block_until_ready(timed(x))
                for index, clock in enumerate(clocks):
                    values = clock.read(2)[0]
                    readings[index, swapped, target, run] = [values[0] & 0xFFFFFFFF, values[0] >> 32, values[1] & 0xFFFFFFFF, values[1] >> 32]
    everything = np.asarray(multihost_utils.process_allgather(jax.make_array_from_process_local_data(sharding, readings), tiled=True)).astype(np.int64)
    if jax.process_index() != 0:
        return
    first = everything[..., 0] | (everything[..., 1] << 32)
    second = everything[..., 2] | (everything[..., 3] << 32)
    coordinates = {device.id: tuple(device.coords) for device in jax.devices()}
    print(f'每种配置 {RUNS} 次交换；o = t 的 GTC − device 0 的 GTC，单位是 GTC 计数（11.2 个计数 / ns）')
    for target in range(1, CHIPS):
        estimates = []
        print(f'device {target} {coordinates[target]}')
        for swapped, (initiator, responder) in enumerate(((0, target), (target, 0))):
            a0, a3 = first[initiator, swapped, target], second[initiator, swapped, target]
            b1, b2 = first[responder, swapped, target], second[responder, swapped, target]
            # 去程：应答方收到时的读数 − 发起方发出前的读数 = 时延 + 偏移；回程：时延 − 偏移。偏移 = 应答方的 GTC − 发起方的 GTC。
            forward, backward = b1 - a0, a3 - b2
            sign = 1 if initiator == 0 else -1
            estimates.append(sign * (forward.min() - backward.min()) / 2)
            bounds = sorted((sign * -backward.min(), sign * forward.min()))
            print(f'  device {initiator} 发起：去程读数差最小 {forward.min()}，中位数 {int(np.median(forward))}；回程最小 {backward.min()}，中位数 {int(np.median(backward))}；只用因果顺序 o ∈ [{bounds[0]}, {bounds[1]}]；假设两个方向最小时延相等 o = {estimates[-1]:+.1f}')
        mean = sum(estimates) / 2
        print(f'  两种发起方式的平均：o = {mean:+.1f}（{mean / 11.2:+.1f} ns）；两者之差的一半 {abs(estimates[0] - estimates[1]) / 2:.1f}')

if __name__ == '__main__':
    main()
