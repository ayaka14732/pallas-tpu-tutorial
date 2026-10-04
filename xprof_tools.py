"""教程脚本共用的 XProf 辅助函数：用 jax.profiler 采集，直接解析采集结果中的 XPlane 文件。"""
from collections.abc import Callable
from pathlib import Path
import shutil
import struct

import jax

# XPlane 文件是一个 protobuf。下面是本教程读取的字段：消息名 → {字段编号: (字段名, 类型, 是否重复)}；类型为另一个消息名时递归解析。
SCHEMA = {
    'XSpace': {1: ('planes', 'XPlane', True)},
    'XPlane': {2: ('name', 'string', False), 3: ('lines', 'XLine', True), 4: ('event_metadata', 'EventEntry', True), 5: ('stat_metadata', 'StatEntry', True)},
    'EventEntry': {1: ('key', 'int', False), 2: ('value', 'XEventMetadata', False)},
    'StatEntry': {1: ('key', 'int', False), 2: ('value', 'XStatMetadata', False)},
    'XEventMetadata': {2: ('name', 'string', False), 4: ('display_name', 'string', False), 5: ('stats', 'XStat', True)},
    'XStatMetadata': {2: ('name', 'string', False)},
    'XLine': {2: ('name', 'string', False), 3: ('timestamp_ns', 'int', False), 4: ('events', 'XEvent', True)},
    'XEvent': {1: ('metadata_id', 'int', False), 2: ('offset_ps', 'int', False), 3: ('duration_ps', 'int', False), 4: ('stats', 'XStat', True)},
    'XStat': {1: ('metadata_id', 'int', False), 2: ('double_value', 'double', False), 3: ('uint64_value', 'int', False), 4: ('int64_value', 'int', False), 5: ('str_value', 'string', False), 6: ('bytes_value', 'bytes', False), 7: ('ref_value', 'int', False)},
}

def _varint(data: bytes, position: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[position]
        position += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return value, position

def decode(data: bytes, message: str) -> dict:
    """按 SCHEMA 解析一个 protobuf 消息：逐个读出“字段编号 + 线型”，再按线型读出值；未声明的字段跳过。"""
    fields = SCHEMA[message]
    result: dict = {name: [] for name, _, repeated in fields.values() if repeated}
    position = 0
    while position < len(data):
        key, position = _varint(data, position)
        number, wire = key >> 3, key & 7
        if wire == 0:  # varint
            value, position = _varint(data, position)
        elif wire == 1:  # 64 位
            value, position = data[position:position + 8], position + 8
        elif wire == 2:  # 带长度的字节串：字符串、bytes 或嵌套的消息
            length, position = _varint(data, position)
            value, position = data[position:position + length], position + length
        else:  # 32 位
            value, position = data[position:position + 4], position + 4
        if number not in fields:
            continue
        name, kind, repeated = fields[number]
        if kind in SCHEMA:
            value = decode(value, kind)
        elif kind == 'string':
            value = value.decode()
        elif kind == 'double':
            value = struct.unpack('<d', value)[0]
        if repeated:
            result[name].append(value)
        else:
            result[name] = value
    return result

def read_xplane(path: Path) -> list[dict]:
    """把一个 .xplane.pb 文件展开成事件列表。每个事件：plane（如 /device:TPU:0、/host:CPU）、line（设备上是轨道名，主机上是线程名）、name、start_ps、duration_ps，以及 stats（事件自身与事件 metadata 上的属性）。"""
    events = []
    for plane in decode(Path(path).read_bytes(), 'XSpace')['planes']:
        metadata = {entry.get('key', 0): entry['value'] for entry in plane['event_metadata']}
        stat_names = {entry.get('key', 0): entry['value'].get('name', '') for entry in plane['stat_metadata']}

        def stats(rows: list[dict]) -> dict:
            result = {}
            for stat in rows:
                name = stat_names.get(stat.get('metadata_id', 0))
                for kind, value in stat.items():
                    if kind != 'metadata_id':
                        # ref_value 指向 stat_metadata 中的一项，那一项的名字才是字符串值。
                        result[name] = stat_names.get(value) if kind == 'ref_value' else value
            return result

        for line in plane['lines']:
            for event in line['events']:
                meta = metadata[event.get('metadata_id', 0)]
                events.append({
                    'plane': plane.get('name', ''),
                    'line': line.get('name', ''),
                    'name': meta.get('display_name') or meta.get('name', ''),
                    'start_ps': line.get('timestamp_ns', 0) * 1000 + event.get('offset_ps', 0),
                    'duration_ps': event.get('duration_ps', 0),
                    'stats': stats([*meta['stats'], *event['stats']]),
                })
    return events

def capture(function: Callable[[], object], directory: Path) -> list[dict]:
    """在 jax.profiler.trace 中执行 function，返回采集到的全部事件（read_xplane 的格式）。"""
    shutil.rmtree(directory, ignore_errors=True)
    with jax.profiler.trace(str(directory)):
        function()
    path, = directory.rglob('*.xplane.pb')
    return read_xplane(path)

def device_events(events: list[dict]) -> list[dict]:
    """TensorCore 上的事件，每个附上 device（如 /device:TPU:0）和 track（如 XLA Modules）。"""
    return [{**event, 'device': event['plane'], 'track': event['line']} for event in events if event['plane'].startswith('/device:TPU')]

def device_cycles(event: dict) -> float:
    """事件在设备上的持续时间，按 TensorCore 的周期计数器约 1.05 GHz 换算成周期数。优先使用由设备计数直接换算的 device_duration_ps。"""
    return event['stats'].get('device_duration_ps', event['duration_ps']) * 1.05e-3
