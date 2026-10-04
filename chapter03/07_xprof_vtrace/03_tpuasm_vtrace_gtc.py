"""vtrace 记录的时间就是 GTC：在计算部分的前后各插入一条 vtrace，从 XPlane 的两个时间字段换回 GTC 的差值；再在同样的两个位置改为读 GTC、读 LCC，比较三者。"""
import tpu_init
tpu_init.initialise_one_chip()

from collections import Counter
import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import tpuasm_tools
from tpuasm_tools import bundle
import xprof_tools

spec = importlib.util.spec_from_file_location('regions', Path(__file__).with_name('01_pallas_trace_regions.py'))
regions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regions)
RUNS = 48

def counts(values: list[int]) -> str:
    return '、'.join(f'{value}（{count} 次）' for value, count in sorted(Counter(values).items()))

def main() -> None:
    x = jnp.arange(1024 * 128, dtype=jnp.float32).reshape(1024, 128) / 1024
    mesh, affine = regions.build()
    compiled = tpuasm_tools.compile(affine, x, mesh=mesh, compiler_options={'xla_enable_custom_call_region_trace': 'false'})
    serialized = tpuasm_tools.serialize(compiled)
    first = tpuasm_tools.find_bundles(serialized, 'vld.8x128')[0]
    last = tpuasm_tools.find_bundles(serialized, 'vst.8x128')[-1]

    print(f'## 一对 vtrace（bundle {first} 之前、bundle {last} 之后），{RUNS} 次调用，每次的事件')
    traced = tpuasm_tools.load(tpuasm_tools.insert_bundles(serialized, {first: bundle('misc: vtrace 0xb0000005'), last + 1: bundle('misc: vtrace 0xc0000005')}), compiled)
    traced(x).block_until_ready()
    events = xprof_tools.device_events(xprof_tools.capture(lambda: [traced(x).block_until_ready() for _ in range(RUNS)], Path('/tmp/pallas_tpu_tutorial/xprof')))
    events = [event for event in events if event['device'] == '/device:TPU:0' and event['track'] == 'XLA TraceMe']
    print(f'  duration_ps：{counts([event["duration_ps"] for event in events])}')
    print(f'  duration_ps × 11.2 / 1000，即 GTC 之差 ΔG：{counts([xprof_tools.gtc_delta(event) for event in events])}')
    print(f'  device_duration_ps：{counts([event["stats"]["device_duration_ps"] for event in events])}')
    print(f'  device_duration_ps × 0.7 / 1000，即 GTC 高 60 位之差 ΔT：{counts([xprof_tools.gtc_ticks(event) for event in events])}')

    clock = tpuasm_tools.KernelClock(num_cores=1)
    for counter in ('gtc', 'lcc'):
        timed = clock.instrument(compiled, [first, last + 1], counter)
        readings = []
        for _ in range(RUNS):
            timed(x).block_until_ready()
            readings.append(clock.read(2).astype(np.int64)[0])
        readings = np.array(readings)
        print(f'## 同样的两个位置改为读 {counter.upper()}，{RUNS} 次调用（两次读数之间有 20 个周期是读数自身的开销）')
        print(f'  两次读数之差：{counts((readings[:, 1] - readings[:, 0]).tolist())}')
        if counter == 'gtc':
            print(f'  高 60 位之差：{counts(((readings[:, 1] >> 4) - (readings[:, 0] >> 4)).tolist())}')

if __name__ == '__main__':
    main()
