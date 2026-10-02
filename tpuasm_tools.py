"""教程脚本共用的 tpuasm 辅助函数：编译、读取机器清单、统计指令、改写并运行 executable。

所有编译产物证据都来自 executable 中实际的程序映像，经 tpuasm 反汇编得到带物理槽位和 Pallas 源码注释的清单。完整清单包含 runtime 的 prologue、例程和 epilogue；`kernel_listing` 只保留编译器归属到 HLO 指令（Pallas kernel 或 XLA fusion）的函数段。
"""
import atexit
from collections import Counter
from collections.abc import Callable
import contextlib
from pathlib import Path
import re
from typing import Any, cast

import jax
from jax.stages import Compiled
from jaxlib.xla_client import LoadedExecutable

from tpuasm import BundleInsertion, assemble_listing, compiler_source_mapping, executable_programs, executable_source_maps, format_assembly, insert_executable_bundles, load_executable, replace_executable_programs

TARGET = 'tpu-v4-tc'
ROOT = Path(__file__).resolve().parent
# 源码映射需要在 TPU 初始化之前装入编译器钩子；导入本模块时进入一次，整个进程保持，退出时恢复。
_SOURCE_MAPPING = contextlib.ExitStack()
_SOURCE_MAPPING.enter_context(compiler_source_mapping())
atexit.register(_SOURCE_MAPPING.close)
_INSTRUCTION = re.compile(r'^\s*\{?\s*([a-z]+[0-9]*): (?:@!?p[0-9]+ )?([a-z][\w.]*)')

def compile(function: Callable[..., Any], *arguments: Any, mesh: jax.sharding.Mesh | None = None, compiler_options: dict[str, str] | None = None) -> Compiled:
    """编译并保留源码映射；mesh 是 shard_map 使用的 mesh，没有时省略。"""
    jax.config.update('jax_enable_compilation_cache', False)
    # 设置 abstract mesh 后，kernel 中重复调用的 jnp 函数不会复用第一次追踪的 jaxpr，来源注释的行号才准确。
    abstract_mesh = jax.sharding.use_abstract_mesh(mesh.abstract_mesh) if mesh is not None else contextlib.nullcontext()
    with abstract_mesh:
        return jax.jit(function, compiler_options=compiler_options).lower(*arguments).compile()

def serialize(compiled: Compiled) -> bytes:
    return bytes(cast(LoadedExecutable, compiled.runtime_executable()).serialize())

def full_listing(serialized: bytes) -> str:
    """唯一一份 TC 程序映像的完整清单，源码路径改为相对仓库根目录。"""
    (_, _, image), = executable_programs(serialized)
    source_map, = executable_source_maps(serialized)
    return format_assembly(image, target=TARGET, source_map=source_map).replace(f'{ROOT}/', '')

def kernel_listing(compiled: Compiled | bytes, *, pallas_only: bool = False) -> str:
    """只保留编译器归属到 HLO 指令的代码：Pallas kernel 的函数段，以及 XLA fusion 的 entry 到 exit 之间；中间被省略的 runtime 代码记为一行注释。pallas_only=True 时只保留 Pallas kernel 的函数段。"""
    serialized = compiled if isinstance(compiled, bytes) else serialize(compiled)
    lines: list[str] = []
    in_function = False
    in_hlo = False
    skipped = 0
    for line in full_listing(serialized).splitlines():
        if line.startswith('# function '):
            in_function = True
        if line.startswith('# entry bundle') and not pallas_only:
            in_hlo = True
        if in_function or in_hlo or line.startswith('# exit bundle') and not pallas_only:
            if lines and skipped:
                lines.append(f'# ... 省略 runtime 代码 {skipped} 个 bundle')
            skipped = 0
            lines.append(line)
        elif line.startswith('{'):
            skipped += 1
        if line.startswith('# end function'):
            in_function = False
        if line.startswith('# exit bundle'):
            in_hlo = False
    return '\n'.join(lines)

def listing_outline(compiled: Compiled | bytes) -> str:
    """完整清单的结构概览：每个结构注释（runtime 例程、函数段、HLO entry/exit）所在的 bundle 编号，以及清单总 bundle 数。"""
    serialized = compiled if isinstance(compiled, bytes) else serialize(compiled)
    lines: list[str] = []
    pc = 0
    for line in full_listing(serialized).splitlines():
        if line.startswith('# ') and not line.startswith(('# source mapping', '# no tpuasm')):
            lines.append(f'bundle {pc:4d}: {line[2:]}')
        elif line.startswith('{'):
            pc += 1
    lines.append(f'bundle {pc:4d}: 清单结束（共 {pc} 个 bundle）')
    return '\n'.join(lines)

def print_kernel_listing(compiled: Compiled | bytes) -> None:
    print(kernel_listing(compiled))

def count_mnemonics(listing: str) -> Counter[str]:
    """按助记符统计清单中的指令条数。"""
    return Counter(match[2] for line in listing.splitlines() if (match := _INSTRUCTION.match(line)) and match[1] != 'encoding')

def print_mnemonic_counts(listing: str) -> None:
    counts = count_mnemonics(listing)
    for mnemonic in sorted(counts):
        print(f'{mnemonic}: {counts[mnemonic]}')

def replace_listing(serialized: bytes, edit: Callable[[str], str], *, encoding: str = 'exact') -> bytes:
    """对完整清单做文本改写后重新汇编，替换原程序映像（新旧映像长度必须相同）。

    encoding='exact' 的清单带有逐字节还原所需的 `.encoding` 约束，适合只改操作数的数值；把一条指令换成另一条时，原指令的约束可能不再适用，此时用 encoding='canonical'，由汇编器重新选择编码。
    """
    (record, index, image), = executable_programs(serialized)
    source = format_assembly(image, target=TARGET, encoding=encoding)
    return replace_executable_programs(serialized, {(record, index): assemble_listing(edit(source))})

def _bundle_spans(lines: list[str]) -> list[tuple[int, int]]:
    """清单中每个 bundle 占据的行区间（含两端）；bundle 从行首的 `{` 开始，到以 `}` 结尾的行结束。"""
    spans = []
    start = None
    for number, line in enumerate(lines):
        if start is None and line.startswith('{'):
            start = number
        if start is not None and line.rstrip().endswith('}'):
            spans.append((start, number))
            start = None
    return spans

def pallas_bundles(serialized: bytes) -> list[int]:
    """编译器归属到 Pallas kernel 的全部 bundle 编号。"""
    source_map, = executable_source_maps(serialized)
    return sorted({pc for function in source_map.functions for span in function.ranges for pc in range(span.image_start, span.image_limit)})

def bundle_text(serialized: bytes, pc: int, *, encoding: str = 'canonical') -> str:
    """第 pc 个 bundle 的清单文本。"""
    (_, _, image), = executable_programs(serialized)
    lines = format_assembly(image, target=TARGET, encoding=encoding).splitlines()
    start, end = _bundle_spans(lines)[pc]
    return '\n'.join(lines[start:end + 1])

def find_bundles(serialized: bytes, text: str, *, encoding: str = 'canonical') -> list[int]:
    """Pallas kernel 中文本包含 text 的 bundle 编号，按顺序排列。"""
    (_, _, image), = executable_programs(serialized)
    lines = format_assembly(image, target=TARGET, encoding=encoding).splitlines()
    spans = _bundle_spans(lines)
    return [pc for pc in pallas_bundles(serialized) if text in '\n'.join(lines[spans[pc][0]:spans[pc][1] + 1])]

def edit_bundles(serialized: bytes, edits: dict[int, tuple[str, str]], *, encoding: str = 'canonical') -> bytes:
    """只在指定编号的 bundle 内做文本替换：edits[pc] = (原文本, 新文本)，原文本在该 bundle 中必须恰好出现一次。"""
    (record, index, image), = executable_programs(serialized)
    lines = format_assembly(image, target=TARGET, encoding=encoding).splitlines()
    spans = _bundle_spans(lines)
    # 从后往前改，前面 bundle 的行号不受影响。
    for pc, (old, new) in sorted(edits.items(), reverse=True):
        start, end = spans[pc]
        bundle = '\n'.join(lines[start:end + 1])
        assert bundle.count(old) == 1, (pc, bundle)
        lines[start:end + 1] = bundle.replace(old, new).split('\n')
    return replace_executable_programs(serialized, {(record, index): assemble_listing('\n'.join(lines))})

def insert_bundles(serialized: bytes, insertions: dict[int, str]) -> bytes:
    """在原 bundle 编号 pc 之前插入一段清单（不含 `.target` 行），分支与元数据由 tpuasm 重定位。"""
    (record, index, _), = executable_programs(serialized)
    return insert_executable_bundles(serialized, {(record, index): [BundleInsertion(pc, f'.target {TARGET}\n{text}') for pc, text in sorted(insertions.items())]})

def load(serialized: bytes, template: Compiled) -> Compiled:
    """按 template 的调用约定装载改写后的 executable。"""
    return load_executable(serialized, template)
