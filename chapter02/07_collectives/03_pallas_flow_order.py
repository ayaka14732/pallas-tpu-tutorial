"""找出一次交换中“发起顺序”起作用的原因：只让指定的几颗芯片发出指定的几条 remote DMA（流），比较不同的流组合与发起顺序下每次的时间。每条流写成 (源, 目的)，都是物理环 mesh 中的位置；位置 0、1、2、3 依次是 device 0、1、3、2，相差 2 的两个位置在对角线上。"""
import tpu_init
tpu_init.initialize_local_chips()

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

import tpuasm_tools

CHIPS = 4
ROWS = 2048  # 每条流 f32[2048,128]，1 MiB
Flow = tuple[int, int]
# (说明, 流的列表)；同一个源的流按列表中的顺序发起。
CASES: tuple[tuple[str, list[Flow]], ...] = (
    ('一条对角线流', [(0, 2)]),
    ('对角线流与发往位置 1 的相邻流，先发相邻', [(0, 1), (0, 2)]),
    ('同上，先发对角线', [(0, 2), (0, 1)]),
    ('对角线流与发往位置 3 的相邻流，先发相邻', [(0, 3), (0, 2)]),
    ('同上，先发对角线', [(0, 2), (0, 3)]),
    ('位置 1 的对角线流与位置 0 发往位置 3 的相邻流', [(1, 3), (0, 3)]),
    ('再加位置 1 发往位置 0 的相邻流，位置 1 先发对角线', [(0, 3), (1, 3), (1, 0)]),
    ('同上，位置 1 先发相邻', [(0, 3), (1, 0), (1, 3)]),
)

def build(mesh: jax.sharding.Mesh, flows: list[Flow], repeats: int):
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
            scratch_types=(
                pltpu.VMEM((ROWS, 128), x.dtype),
                pltpu.VMEM((len(flows), ROWS, 128), x.dtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((len(flows),)),
                pltpu.SemaphoreType.DMA((len(flows),)),
            ),
            name='flows',
            compiler_params=pltpu.CompilerParams(
                collective_id=1,
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, acc: Ref, inbox: Ref, sem: Ref, send_sems: Ref, recv_sems: Ref) -> None:
            me = jax.lax.axis_index('device')
            pltpu.async_copy(x_hbm.at[0], acc, sem).wait()

            def flow(k: int):
                # 第 k 条流把源芯片的 acc 写进目的芯片 inbox 的第 k 格。
                return pltpu.make_async_remote_copy(acc, inbox.at[k], send_sems.at[k], recv_sems.at[k], device_id={'device': flows[k][1], 'tc': 0}, device_id_type=pl.DeviceIdType.MESH)

            @pl.loop(0, repeats)
            def _(_: jax.Array) -> None:
                ready = pltpu.get_barrier_semaphore()
                for rank in range(CHIPS):
                    pl.semaphore_signal(ready, 1, device_id={'device': rank, 'tc': 0})
                pl.semaphore_wait(ready, CHIPS)
                # 每颗芯片只执行属于自己的那一段：发起以自己为源的流，再等以自己为目的的流到达。
                for position in range(CHIPS):
                    @pl.when(me == position)
                    def _() -> None:
                        outgoing = [flow(k) for k, (source, _) in enumerate(flows) if source == position]
                        for copy in outgoing:
                            copy.start()
                        for copy in outgoing:
                            copy.wait_send()
                        for k, (_, destination) in enumerate(flows):
                            if destination == position:
                                flow(k).wait_recv()

            # 结果是自己的数组加上收到的每一份，用于数值检查。
            for position in range(CHIPS):
                @pl.when(me == position)
                def _() -> None:
                    for k, (_, destination) in enumerate(flows):
                        if destination == position:
                            acc[...] = acc[...] + inbox[k]

            pltpu.async_copy(acc, o_hbm.at[0], sem).wait()

        return kernel(x)

    return exchange

def cycles_per_call(clock: tpuasm_tools.KernelClock, mesh: jax.sharding.Mesh, flows: list[Flow], x: jax.Array) -> float:
    """每次交换的周期数：kernel 内部重复 64 次与 32 次，各用 LCC 读出 kernel 在 device 0 上的周期数，相减除以 32。"""
    cycles = {}
    for repeats in (32, 64):
        compiled = tpuasm_tools.compile(build(mesh, flows, repeats), x, mesh=mesh)
        cycles[repeats] = clock.kernel_cycles(compiled, lambda timed: jax.block_until_ready(timed(x)))
    return (cycles[64] - cycles[32]) / 32

def main() -> None:
    devices = {device.id: device for device in jax.devices()}
    mesh = jax.sharding.Mesh(np.array([devices[i] for i in (0, 1, 3, 2)]), ('device',))
    host = np.random.default_rng(0).integers(-100, 100, (CHIPS, ROWS, 128)).astype(np.float32)
    x = jax.device_put(jnp.asarray(host), jax.NamedSharding(mesh, P('device')))
    clock = tpuasm_tools.KernelClock(num_cores=1)
    for name, flows in CASES:
        expected = host.copy()
        for source, destination in flows:
            expected[destination] += host[source]
        np.testing.assert_array_equal(np.asarray(tpuasm_tools.compile(build(mesh, flows, 1), x, mesh=mesh)(x)), expected)
        print(f'{name}，流 {flows}：数值检查通过；每次 {cycles_per_call(clock, mesh, flows, x):.0f} 个周期')

if __name__ == '__main__':
    main()
