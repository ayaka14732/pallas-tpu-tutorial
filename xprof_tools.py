"""教程脚本共用的 XProf 辅助函数：采集 trace，按 TensorCore 和轨道读出设备事件。"""
from collections.abc import Callable
import gzip
import json
from pathlib import Path
import shutil

import jax

def capture(function: Callable[[], object], directory: Path) -> list[dict]:
    """在 jax.profiler.trace 中执行 function，返回 trace 中的全部事件。"""
    shutil.rmtree(directory, ignore_errors=True)
    with jax.profiler.trace(str(directory)):
        function()
    path, = directory.rglob('*.trace.json.gz')
    with gzip.open(path, 'rt') as file:
        return json.load(file)['traceEvents']

def device_events(events: list[dict]) -> list[dict]:
    """设备上的完整事件，每个附上 device（如 /device:TPU:0）和 track（如 XLA Modules）。"""
    processes = {event['pid']: event['args']['name'] for event in events if event.get('ph') == 'M' and event.get('name') == 'process_name'}
    threads = {(event['pid'], event['tid']): event['args']['name'] for event in events if event.get('ph') == 'M' and event.get('name') == 'thread_name'}
    result = []
    for event in events:
        if event.get('ph') == 'X' and processes.get(event['pid'], '').startswith('/device:TPU'):
            result.append({**event, 'device': processes[event['pid']], 'track': threads.get((event['pid'], event['tid']), '')})
    return result

def duration_us(event: dict) -> float:
    """事件在设备上的持续时间（µs），优先使用皮秒精度的 device_duration_ps。"""
    picoseconds = event.get('args', {}).get('device_duration_ps')
    return float(picoseconds) / 1e6 if picoseconds is not None else float(event['dur'])
