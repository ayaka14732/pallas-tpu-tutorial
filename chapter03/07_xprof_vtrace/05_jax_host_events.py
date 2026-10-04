"""只有 XProf 能给出的信息：一次“调用并等到结果”在主机上花在哪里。采集主机一侧的事件，按事件名统计每次调用中的时间，并与设备上的 module、kernel 事件对照。kernel 是第 1 节的 pl.delay(100000)。"""
import tpu_init
tpu_init.initialise_one_chip()

from collections import defaultdict
import importlib.util
from pathlib import Path
import statistics

import jax
import jax.numpy as jnp

import tpuasm_tools
import xprof_tools

spec = importlib.util.spec_from_file_location('five_times', Path(__file__).resolve().parents[1] / '01_five_times' / '01_pallas_host_and_device_time.py')
five_times = importlib.util.module_from_spec(spec)
spec.loader.exec_module(five_times)
CALLS = 8
EVENTS = (
    'PjitFunction(jit(wait))',
    'PjRtCApiLoadedExecutable::Execute',
    'CommonPjRtLoadedExecutable::ExecutePrepare',
    'TpuLoadedExecutable::ExecuteLaunch',
    'tpu::System::Execute',
    'ReadSyncFlag',
    'CompleteCallbacks',
    'tpu::System::Execute=>Done',
)

def main() -> None:
    mesh, wait = five_times.build(100_000)
    x = jnp.zeros((8, 128), jnp.float32)
    compiled = tpuasm_tools.compile(wait, x, mesh=mesh)
    compiled(x).block_until_ready()

    def calls() -> None:
        for _ in range(CALLS):
            with jax.profiler.TraceAnnotation('call'):
                compiled(x).block_until_ready()

    events = xprof_tools.capture(calls, Path('/tmp/pallas_tpu_tutorial/xprof'))
    # 主机事件的时间以皮秒记录，这里换成微秒。
    host = [(event['name'], event['start_ps'] / 1e6, event['duration_ps'] / 1e6, event['stats'].get('core_id')) for event in events if event['plane'].startswith('/host')]
    windows = sorted((start, start + duration) for name, start, duration, _ in host if name == 'call')
    # 每次调用中，同一个名字可能嵌套出现（外层包住内层），也可能每个 TensorCore 各一次。
    occurrences = defaultdict(list)
    for begin, end in windows:
        seen = defaultdict(int)
        for name, start, duration, core in sorted(host, key=lambda item: item[1]):
            if begin <= start <= end and name in EVENTS:
                # 带 core_id 的事件每个 TensorCore 一次，按 core_id 归类；其余按出现的先后归类。
                occurrences[(name, 0 if core is not None else seen[name], core)].append((start - begin, duration))
                seen[name] += 1
    print(f'## 主机：{CALLS} 次调用，每次的中位数（开始时刻相对调用开始）')
    print(f'  整个调用（call 区间）：{statistics.median(end - begin for begin, end in windows):.1f} µs')
    for (name, index, core), values in sorted(occurrences.items(), key=lambda item: statistics.median(value[0] for value in item[1])):
        note = f' 第 {index + 1} 次' if core is None else f'（core_id={core}）'
        print(f'  {name}{note}：开始 {statistics.median(value[0] for value in values):.1f} µs，持续 {statistics.median(value[1] for value in values):.1f} µs')
    print('## 设备')
    for track in ('XLA Modules', 'XLA Ops'):
        durations = [xprof_tools.device_ns(event) / 1000 for event in xprof_tools.device_events(events) if event['device'] == '/device:TPU:0' and event['track'] == track]
        print(f'  TensorCore 0 {track}：{statistics.median(durations):.1f} µs')

if __name__ == '__main__':
    main()
