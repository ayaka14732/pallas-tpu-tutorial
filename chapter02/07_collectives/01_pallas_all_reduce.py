"""四颗芯片的 all-reduce（求和），每颗芯片一个 f32[R,128]：一次交换（每颗芯片把整份数据直接发给其余三颗）、环形（reduce-scatter 加 all-gather，只和物理相邻的芯片通信）与双向环形。比较不同 R 下每次 all-reduce 的时间。"""
import tpu_init
tpu_init.initialise_local_chips()

import statistics
import time

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

import tpuasm_tools

CHIPS = 4
PARAMS = pltpu.CompilerParams(
    collective_id=1,
    disable_bounds_checks=True,
    disable_semaphore_checks=True,
)

def barrier() -> None:
    """四颗芯片汇合：给每颗芯片的 barrier 信号量加 1，再等自己的达到 4。"""
    ready = pltpu.get_barrier_semaphore()
    for rank in range(CHIPS):
        pl.semaphore_signal(ready, 1, device_id={'device': rank, 'tc': 0})
    pl.semaphore_wait(ready, CHIPS)

def one_shot(mesh: jax.sharding.Mesh, rows: int, repeats: int):
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P('device'),
        out_specs=P('device'),
        check_vma=False,
    )
    def all_reduce(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=pltpu.HBM(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(
                pltpu.VMEM((rows, 128), x.dtype),
                pltpu.VMEM((CHIPS - 1, rows, 128), x.dtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((CHIPS - 1,)),
                pltpu.SemaphoreType.DMA((CHIPS - 1,)),
            ),
            name='one_shot',
            compiler_params=PARAMS,
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, acc: Ref, inbox: Ref, sem: Ref, send_sems: Ref, recv_sems: Ref) -> None:
            me = jax.lax.axis_index('device')
            pltpu.async_copy(x_hbm.at[0], acc, sem).wait()

            @pl.loop(0, repeats)
            def _(_: jax.Array) -> None:
                # 汇合：确认每颗芯片都已用完上一次收到的数据，inbox 可以再次写入。
                barrier()
                # 第 k 个 DMA 把整份数据发给 me + k + 1，写进对方 inbox 的第 k 格；三个 DMA 同时进行。
                copies = [pltpu.make_async_remote_copy(acc, inbox.at[k], send_sems.at[k], recv_sems.at[k], device_id={'device': (me + k + 1) % CHIPS, 'tc': 0}, device_id_type=pl.DeviceIdType.MESH) for k in range(CHIPS - 1)]
                for copy in copies:
                    copy.start()
                for copy in copies:
                    copy.wait_send()
                    copy.wait_recv()
                # 对方 inbox 的第 k 格来自 me − k − 1；三份相加。重复多次时不改变 acc，便于计时。
                inbox[0] = acc[...] + inbox[0] + inbox[1] + inbox[2]

            pltpu.async_copy(inbox.at[0], o_hbm.at[0], sem).wait()

        return kernel(x)

    return all_reduce

def ring(mesh: jax.sharding.Mesh, rows: int, repeats: int, directions: int = 1):
    """环形 all-reduce。directions=2 时把数据分成两半，一半沿环向右、一半沿环向左，同时进行，用上每颗芯片的两条链路。"""
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)
    half = rows // directions
    chunk = half // CHIPS

    @jax.shard_map(
        mesh=mesh,
        in_specs=P('device'),
        out_specs=P('device'),
        check_vma=False,
    )
    def all_reduce(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=pltpu.HBM(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(
                pltpu.VMEM((rows, 128), x.dtype),
                pltpu.VMEM((directions, chunk, 128), x.dtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((directions, 2)),
                pltpu.SemaphoreType.REGULAR((directions,)),
            ),
            name='ring',
            compiler_params=PARAMS,
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, acc: Ref, inbox: Ref, sem: Ref, sems: Ref, credits: Ref) -> None:
            me = jax.lax.axis_index('device')
            # 方向 d = 0 向右（me + 1），d = 1 向左（me − 1）；sign 把向左的环写成向右的镜像。
            signs = (1, -1)[:directions]

            def part(direction: int, index: jax.Array) -> Ref:
                return acc.at[pl.ds(direction * half + index * chunk, chunk)]

            def step(sources: list[Ref], destinations: list[Ref], accumulate: list[jax.Array] | None, first: bool) -> None:
                transfers = []
                for d, sign in enumerate(signs):
                    # 向前发送之前，等下游的芯片表示它已经用完上一次收到的数据（第一步之前已有汇合）。
                    if not first:
                        pl.semaphore_wait(credits.at[d], 1)
                    transfers.append(pltpu.make_async_remote_copy(sources[d], destinations[d], sems.at[d, 0], sems.at[d, 1], device_id={'device': (me + sign) % CHIPS, 'tc': 0}, device_id_type=pl.DeviceIdType.MESH))
                for transfer in transfers:
                    transfer.start()
                for transfer in transfers:
                    transfer.wait_send()
                    transfer.wait_recv()
                for d, sign in enumerate(signs):
                    if accumulate is not None:
                        part(d, accumulate[d])[...] = part(d, accumulate[d])[...] + inbox[d]
                    # 用完了上游发来的数据，通知上游的芯片可以再发。
                    pl.semaphore_signal(credits.at[d], 1, device_id={'device': (me - sign) % CHIPS, 'tc': 0})

            @pl.loop(0, repeats)
            def _(_: jax.Array) -> None:
                pltpu.async_copy(x_hbm.at[0], acc, sem).wait()
                barrier()
                # reduce-scatter：第 s 步把第 me − s 块发给下游，收到上游的第 me − s − 1 块并累加；三步后第 me + 1 块是完整的和（向左的环取镜像）。
                for s in range(CHIPS - 1):
                    sources = [part(d, (me - sign * s) % CHIPS) for d, sign in enumerate(signs)]
                    targets = [(me - sign * (s + 1)) % CHIPS for d, sign in enumerate(signs)]
                    step(sources, [inbox.at[d] for d in range(directions)], targets, s == 0)
                # all-gather：第 s 步把完整的第 me + 1 − s 块直接写进下游 acc 的同一位置。
                for s in range(CHIPS - 1):
                    indices = [(me + sign * (1 - s)) % CHIPS for d, sign in enumerate(signs)]
                    parts = [part(d, indices[d]) for d in range(directions)]
                    step(parts, parts, None, False)
                # 最后一步之后，上游还会给自己发一个许可，消耗掉它，使每一轮开始时计数为 0。
                for d in range(directions):
                    pl.semaphore_wait(credits.at[d], 1)

            pltpu.async_copy(acc, o_hbm.at[0], sem).wait()

        return kernel(x)

    return all_reduce

def bidirectional(mesh: jax.sharding.Mesh, rows: int, repeats: int):
    return ring(mesh, rows, repeats, directions=2)

def time_per_call(build, mesh: jax.sharding.Mesh, rows: int, x: jax.Array) -> float:
    """每次 all-reduce 的时间（微秒）：kernel 内部重复 64 次与 32 次的主机计时之差除以 32。"""
    times = {}
    for repeats in (32, 64):
        compiled = tpuasm_tools.compile(build(mesh, rows, repeats), x, mesh=mesh)
        jax.block_until_ready(compiled(x))
        samples = []
        for _ in range(20):
            start = time.perf_counter()
            jax.block_until_ready(compiled(x))
            samples.append(time.perf_counter() - start)
        times[repeats] = statistics.median(samples)
    return (times[64] - times[32]) / 32 * 1e6

def main() -> None:
    devices = {device.id: device for device in jax.devices()}
    # 物理环 0 → 1 → 3 → 2（第 5 节）。
    mesh = jax.sharding.Mesh(np.array([devices[i] for i in (0, 1, 3, 2)]), ('device',))
    for rows in (32, 256, 2048):
        host = np.random.default_rng(rows).integers(-100, 100, (CHIPS, rows, 128)).astype(np.float32)
        x = jax.device_put(jnp.asarray(host), jax.NamedSharding(mesh, P('device')))
        expected = np.broadcast_to(host.sum(axis=0), host.shape)
        print(f'## 每颗芯片 f32[{rows},128]（{rows * 512 // 1024} KiB）')
        for name, build in (('一次交换', one_shot), ('环形', ring), ('双向环形', bidirectional)):
            compiled = tpuasm_tools.compile(build(mesh, rows, 1), x, mesh=mesh)
            np.testing.assert_array_equal(np.asarray(compiled(x)), expected)
            print(f'  {name}：数值检查通过；每次约 {time_per_call(build, mesh, rows, x):.1f} µs')

if __name__ == '__main__':
    main()
