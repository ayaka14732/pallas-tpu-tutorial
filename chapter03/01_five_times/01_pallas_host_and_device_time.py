"""同一个 kernel 的五种“时间”：Python 调用返回的时间、调用并等到结果的时间、连续调用时每次调用的时间、XProf 中设备上的时间，以及从清单读出的设备工作量。kernel 用 pl.delay 让 TensorCore 停住已知的时间。"""
import tpu_init
tpu_init.initialise_one_chip()

from pathlib import Path
import statistics
import time

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

import tpuasm_tools
import xprof_tools

DELAYS_NS = (10_000, 1_000_000)
CALLS = 64

def build(delay_ns: int):
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def wait(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.SemaphoreType.DMA),
            name='wait',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            # 让 TensorCore 停住 delay_ns 纳秒。
            pl.delay(delay_ns)
            x_vmem[...] = x_vmem[...] + 1.0
            pltpu.async_copy(x_vmem, o_hbm, sem).wait()

        return kernel(x)

    return mesh, wait

def main() -> None:
    x = jnp.zeros((8, 128), jnp.float32)
    for delay_ns in DELAYS_NS:
        mesh, wait = build(delay_ns)
        compiled = tpuasm_tools.compile(wait, x, mesh=mesh)
        compiled(x).block_until_ready()
        print(f'## pl.delay({delay_ns})')
        listing = tpuasm_tools.kernel_listing(compiled, pallas_only=True)
        # pl.delay 按 1.05 GHz 把纳秒换算成周期，写成 vdelay 的立即数或寄存器。
        cycles = str(delay_ns * 105 // 100)
        print('清单：' + '；'.join(' '.join(line.split('#')[0].split()) for line in listing.splitlines() if 'vdelay' in line or cycles in line))
        dispatch, synchronized = [], []
        for _ in range(20):
            start = time.perf_counter()
            result = compiled(x)
            returned = time.perf_counter()
            result.block_until_ready()
            done = time.perf_counter()
            dispatch.append(returned - start)
            synchronized.append(done - start)
        print(f'1. 调用返回：中位数 {statistics.median(dispatch) * 1e6:.1f} µs')
        print(f'2. 调用并等到结果：中位数 {statistics.median(synchronized) * 1e6:.1f} µs')
        # 连续调用而不等结果：每次调用的输入都是上一次的输出。
        calls = []
        result = x
        begin = time.perf_counter()
        for _ in range(CALLS):
            start = time.perf_counter()
            result = compiled(result)
            calls.append(time.perf_counter() - start)
        result.block_until_ready()
        total = time.perf_counter() - begin
        print(f'3. 连续调用 {CALLS} 次：第 1–30 次每次中位数 {statistics.median(calls[:30]) * 1e6:.1f} µs，第 35–64 次每次中位数 {statistics.median(calls[34:]) * 1e6:.1f} µs；从开始到等到最后一个结果共 {total * 1e6:.0f} µs，平均每次 {total / CALLS * 1e6:.1f} µs')
        events = xprof_tools.device_events(xprof_tools.capture(lambda: [compiled(x).block_until_ready() for _ in range(8)], Path('/tmp/pallas_tpu_tutorial/xprof')))
        for device in sorted({event['device'] for event in events}):
            for track in ('XLA Modules', 'XLA Ops'):
                durations = [xprof_tools.duration_us(event) for event in events if event['device'] == device and event['track'] == track]
                print(f'4. XProf {device} {track}：{len(durations)} 个事件，中位数 {statistics.median(durations):.2f} µs')

if __name__ == '__main__':
    main()
