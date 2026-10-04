"""不用 named_scope，直接用 tpuasm 在计算部分的前后各插入一条 vtrace：比较三种操作数在 XProf 中得到的事件，再在同样的位置换成 LCC 读数。"""
import tpu_init
tpu_init.initialise_one_chip()

from collections import defaultdict
import importlib.util
from pathlib import Path
import statistics

import jax.numpy as jnp
import numpy as np

import tpuasm_tools
from tpuasm_tools import bundle
import xprof_tools

spec = importlib.util.spec_from_file_location('regions', Path(__file__).with_name('01_pallas_trace_regions.py'))
regions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regions)
# (开始标记的操作数, 结束标记的操作数)
MARKERS = (('0xb8000000', '0xc8000000'), ('0xb0000005', '0xc0000005'), ('0x80000007', '0x90000007'))

def device_medians(function, x) -> str:
    events = xprof_tools.device_events(xprof_tools.capture(lambda: [function(x).block_until_ready() for _ in range(16)], Path('/tmp/pallas_tpu_tutorial/xprof')))
    durations = defaultdict(list)
    for event in events:
        if event['device'] == '/device:TPU:0' and event['track'] != 'XLA Modules':
            durations[(event['track'], event['name'])].append(xprof_tools.device_cycles(event))
    return '，'.join(f'{track} {name} {statistics.median(values):.0f} 个周期' for (track, name), values in sorted(durations.items()))

def main() -> None:
    x = jnp.arange(1024 * 128, dtype=jnp.float32).reshape(1024, 128) / 1024
    # 载体是上一个实验的 kernel，关闭 region trace：named_scope 不产生任何 vtrace。
    mesh, affine = regions.build()
    compiled = tpuasm_tools.compile(affine, x, mesh=mesh, compiler_options={'xla_enable_custom_call_region_trace': 'false'})
    serialized = tpuasm_tools.serialize(compiled)
    first = tpuasm_tools.find_bundles(serialized, 'vld.8x128')[0]
    last = tpuasm_tools.find_bundles(serialized, 'vst.8x128')[-1]
    print(f'计算部分是 bundle {first}–{last}，共 {last - first + 1} 个；程序中的 vtrace 标记：{tpuasm_tools.hlo_ops(compiled)}')
    print(f'未改写：{device_medians(compiled, x)}')
    for start, stop in MARKERS:
        patched = tpuasm_tools.load(tpuasm_tools.insert_bundles(serialized, {first: bundle(f'misc: vtrace {start}'), last + 1: bundle(f'misc: vtrace {stop}')}), compiled)
        np.testing.assert_array_equal(np.asarray(patched(x)), np.asarray(x) * 2.0 + 1.0)
        print(f'vtrace {start} … vtrace {stop}：{device_medians(patched, x)}')
    clock = tpuasm_tools.KernelClock(num_cores=1)
    timed = clock.instrument(compiled, [first, last + 1])
    timed(x).block_until_ready()
    readings = clock.read(2).astype(np.int64)[0]
    print(f'同样的位置换成 LCC 读数：两次读数相差 {readings[1] - readings[0]} 个周期，其中 20 个是读数自身的开销')

if __name__ == '__main__':
    main()
