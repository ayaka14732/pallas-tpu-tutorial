"""XProf 的逐指令事件从哪里来：打开 xla_xprof_enable_custom_call_tracing 后，编译器每隔若干个 bundle 放一条 0xa 类型的 vtrace，主机把相邻两个标记之间的指令事件按 bundle 编号均匀插值。列出标记的位置，以及一次执行中各 bundle 的事件起始时刻。"""
import tpu_init
tpu_init.initialise_one_chip()

import importlib.util
from pathlib import Path
import re

import jax.numpy as jnp
import numpy as np

import tpuasm_tools
import xprof_tools

spec = importlib.util.spec_from_file_location('regions', Path(__file__).with_name('01_pallas_trace_regions.py'))
regions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regions)

def main() -> None:
    x = jnp.arange(1024 * 128, dtype=jnp.float32).reshape(1024, 128) / 1024
    mesh, affine = regions.build()
    compiled = tpuasm_tools.compile(affine, x, mesh=mesh, compiler_options={'xla_xprof_enable_custom_call_tracing': 'true'})
    np.testing.assert_array_equal(np.asarray(compiled(x)), np.asarray(x) * 2.0 + 1.0)
    bundles = regions.joined_bundles(tpuasm_tools.kernel_listing(compiled, pallas_only=True))
    markers = [(index, re.search(r'vtrace (0x[0-9a-f]+)', text).group(1)) for index, text in enumerate(bundles) if 'vtrace' in text]
    print(f'kernel 段 {len(bundles)} 个 bundle；vtrace（清单中的位置, 操作数）：{markers}')
    print(f'0xa 类型标记的低 28 位：{[int(operand, 16) & 0xFFFFFFF for _, operand in markers if operand.startswith("0xa")]}')

    events = xprof_tools.device_events(xprof_tools.capture(lambda: compiled(x).block_until_ready(), Path('/tmp/pallas_tpu_tutorial/xprof')))
    instructions = [event for event in events if event['device'] == '/device:TPU:0' and event['track'].endswith('Instructions')]
    print(f'TensorCore 0 的轨道：{sorted({event["track"] for event in events if event["device"] == "/device:TPU:0"})}')
    # 每个 bundle 编号取它最早的一个事件；同一个 bundle 中的几条指令起始时刻相同。
    starts: dict[int, tuple[int, str]] = {}
    for event in sorted(instructions, key=lambda event: event['start_ps']):
        starts.setdefault(int(event['stats']['bundle_number']), (event['start_ps'], event['name']))
    print('bundle 编号、其中一条指令、与上一个 bundle 的起始时刻之差（ns）：')
    previous = None
    for number in sorted(starts)[:26]:
        start, name = starts[number]
        delta = '' if previous is None else f'{(start - previous) / 1000:.2f}'
        print(f'  {number:3d}  {name:<14} {delta}')
        previous = start

if __name__ == '__main__':
    main()
