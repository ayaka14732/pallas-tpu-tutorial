"""一次“调用并等到结果”在主机上花在哪里：用 XProf 采集主机一侧的事件，按事件名统计每次调用中的时间，并与设备上的 module、kernel 时间对照。kernel 与第 1 个实验相同，pl.delay(100000)。"""
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

spec = importlib.util.spec_from_file_location('five_times', Path(__file__).with_name('01_pallas_host_and_device_time.py'))
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

    raw = xprof_tools.capture(calls, Path('/tmp/pallas_tpu_tutorial/xprof'))
    processes = {event['pid']: event['args']['name'] for event in raw if event.get('ph') == 'M' and event.get('name') == 'process_name'}
    host = [event for event in raw if event.get('ph') == 'X' and processes.get(event['pid'], '').startswith('/host')]
    windows = sorted((event['ts'], event['ts'] + event['dur']) for event in host if event['name'] == 'call')
    # 每次调用中，同名事件按出现的先后编号；同一个名字可能嵌套（外层包住内层）或每个 TensorCore 各一次。
    occurrences = defaultdict(list)
    for start, end in windows:
        seen = defaultdict(int)
        for event in sorted(host, key=lambda event: event['ts']):
            if start <= event['ts'] <= end and event['name'] in EVENTS:
                occurrences[(event['name'], seen[event['name']])].append((event['ts'] - start, event['dur']))
                seen[event['name']] += 1
    print(f'## 主机：{CALLS} 次调用，每次的中位数（开始时刻相对调用开始）')
    print(f'  整个调用（call 区间）：{xprof_tools.host_cycles(statistics.median(end - start for start, end in windows) / 1e6)}')
    for (name, index), values in sorted(occurrences.items(), key=lambda item: statistics.median(value[0] for value in item[1])):
        print(f'  {name} 第 {index + 1} 次：开始 {xprof_tools.host_cycles(statistics.median(value[0] for value in values) / 1e6)}，持续 {xprof_tools.host_cycles(statistics.median(value[1] for value in values) / 1e6)}')
    device = xprof_tools.device_events(raw)
    print('## 设备')
    for track in ('XLA Modules', 'XLA Ops'):
        durations = [xprof_tools.duration_cycles(event) for event in device if event['device'] == '/device:TPU:0' and event['track'] == track]
        print(f'  TensorCore 0 {track}：{statistics.median(durations):.0f} 个周期')

if __name__ == '__main__':
    main()
