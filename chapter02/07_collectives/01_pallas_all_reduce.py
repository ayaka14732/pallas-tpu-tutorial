"""四颗芯片的 all-reduce（求和），每颗芯片一个 f32[R,128]：一次交换（每颗芯片把整份数据直接发给其余三颗）、环形（reduce-scatter 加 all-gather，只和物理相邻的芯片通信）与双向环形。比较不同 R 下每次 all-reduce 的时间；再单独测汇合，并把一次交换拆成只发给相邻芯片、只发给对角线芯片等几种，看它的时间花在哪里。"""
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

def one_shot(mesh: jax.sharding.Mesh, rows: int, repeats: int, offsets: tuple[int, ...] = (1, 2, 3)):
    """每颗芯片把整份数据同时发给 mesh 中 me + o 的芯片（o 取 offsets 中的每个值），再把收到的几份相加。offsets = (1, 2, 3) 就是 all-reduce；在物理环的 mesh 中，o = 1、3 是相邻芯片，o = 2 是对角线芯片。"""
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
                pltpu.VMEM((len(offsets), rows, 128), x.dtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((len(offsets),)),
                pltpu.SemaphoreType.DMA((len(offsets),)),
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
                # 第 k 个 DMA 把整份数据发给 me + offsets[k]，写进对方 inbox 的第 k 格；几个 DMA 同时进行。
                copies = [pltpu.make_async_remote_copy(acc, inbox.at[k], send_sems.at[k], recv_sems.at[k], device_id={'device': (me + offset) % CHIPS, 'tc': 0}, device_id_type=pl.DeviceIdType.MESH) for k, offset in enumerate(offsets)]
                for copy in copies:
                    copy.start()
                for copy in copies:
                    copy.wait_send()
                    copy.wait_recv()
                # 自己 inbox 的第 k 格来自 me − offsets[k]；几份相加。重复多次时不改变 acc，便于计时。
                total = acc[...]
                for k in range(len(offsets)):
                    total = total + inbox[k]
                inbox[0] = total

            pltpu.async_copy(inbox.at[0], o_hbm.at[0], sem).wait()

        return kernel(x)

    return all_reduce

def barrier_only(mesh: jax.sharding.Mesh, rows: int, repeats: int):
    """只有四方汇合，不传数据：每次汇合的时间。"""
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P('device'),
        out_specs=P('device'),
        check_vma=False,
    )
    def wait_all(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=pltpu.HBM(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM((rows, 128), x.dtype), pltpu.SemaphoreType.DMA),
            name='barrier_only',
            compiler_params=PARAMS,
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, acc: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm.at[0], acc, sem).wait()

            @pl.loop(0, repeats)
            def _(_: jax.Array) -> None:
                barrier()

            pltpu.async_copy(acc, o_hbm.at[0], sem).wait()

        return kernel(x)

    return wait_all

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

def cycles_per_call(clock: tpuasm_tools.KernelClock, build, mesh: jax.sharding.Mesh, rows: int, x: jax.Array) -> float:
    """每次 all-reduce 的周期数：kernel 内部重复 64 次与 32 次，各用 LCC 读出 kernel 在 device 0 上的周期数（第三章第 3 节的 KernelClock），相减除以 32。"""
    cycles = {}
    for repeats in (32, 64):
        compiled = tpuasm_tools.compile(build(mesh, rows, repeats), x, mesh=mesh)
        cycles[repeats] = clock.kernel_cycles(compiled, lambda timed: jax.block_until_ready(timed(x)))
    return (cycles[64] - cycles[32]) / 32

def main() -> None:
    devices = {device.id: device for device in jax.devices()}
    # 物理环 0 → 1 → 3 → 2（第 5 节）。
    mesh = jax.sharding.Mesh(np.array([devices[i] for i in (0, 1, 3, 2)]), ('device',))
    clock = tpuasm_tools.KernelClock(num_cores=1)
    for rows in (32, 256, 2048):
        host = np.random.default_rng(rows).integers(-100, 100, (CHIPS, rows, 128)).astype(np.float32)
        x = jax.device_put(jnp.asarray(host), jax.NamedSharding(mesh, P('device')))
        expected = np.broadcast_to(host.sum(axis=0), host.shape)
        print(f'## 每颗芯片 f32[{rows},128]（{rows * 512 // 1024} KiB）')
        last = lambda mesh, rows, repeats: one_shot(mesh, rows, repeats, (1, 3, 2))
        for name, build in (('一次交换', one_shot), ('一次交换，对角线最后发起', last), ('环形', ring), ('双向环形', bidirectional)):
            compiled = tpuasm_tools.compile(build(mesh, rows, 1), x, mesh=mesh)
            np.testing.assert_array_equal(np.asarray(compiled(x)), expected)
            print(f'  {name}：数值检查通过；每次 {cycles_per_call(clock, build, mesh, rows, x):.0f} 个周期')
    print(f'## 只有四方汇合：每次 {cycles_per_call(clock, barrier_only, mesh, rows, x):.0f} 个周期')
    # 物理环的 mesh 中，me + 1、me + 3 是相邻芯片，me + 2 是对角线芯片。
    # 偏移的顺序就是 DMA 发起的顺序。
    variants = (
        ('一颗相邻芯片', (1,)),
        ('两颗相邻芯片', (1, 3)),
        ('对角线芯片', (2,)),
        ('同一颗相邻芯片发两份', (1, 1)),
        ('相邻芯片与对角线芯片，先发相邻', (3, 2)),
        ('相邻芯片与对角线芯片，先发对角线', (2, 3)),
        ('全部三颗，按 1、2、3 发起', (1, 2, 3)),
        ('全部三颗，对角线最后', (1, 3, 2)),
        ('全部三颗，对角线最先', (2, 1, 3)),
    )
    for rows in (256, 2048):
        host = np.random.default_rng(rows).integers(-100, 100, (CHIPS, rows, 128)).astype(np.float32)
        x = jax.device_put(jnp.asarray(host), jax.NamedSharding(mesh, P('device')))
        print(f'## 拆开一次交换：每颗芯片 f32[{rows},128]（{rows * 512 // 1024} KiB）')
        for name, offsets in variants:
            build = lambda mesh, rows, repeats, offsets=offsets: one_shot(mesh, rows, repeats, offsets)
            compiled = tpuasm_tools.compile(build(mesh, rows, 1), x, mesh=mesh)
            expected = host + sum(np.roll(host, offset, axis=0) for offset in offsets)
            np.testing.assert_array_equal(np.asarray(compiled(x)), expected)
            print(f'  {name}（偏移 {list(offsets)}）：数值检查通过；每次 {cycles_per_call(clock, build, mesh, rows, x):.0f} 个周期')

if __name__ == '__main__':
    main()
