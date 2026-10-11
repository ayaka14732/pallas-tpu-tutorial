"""教程脚本共用的 tpuasm 辅助函数：编译、读取机器清单、统计指令、改写并运行 executable，以及用设备上的周期计数器 LCC 计时。

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
from jax import Ref
from jax._src.state import primitives as state
from jax.experimental import pallas as pl
from jax.experimental.layout import Layout, with_layout_constraint
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.stages import Compiled
from jaxlib.xla_client import LoadedExecutable
import numpy as np

from tpuasm import BundleInsertion, assemble_listing, compiler_source_mapping, executable_programs, executable_source_maps, format_assembly, insert_executable_bundles, load_executable, replace_executable_programs

TARGET = 'tpu-v4-tc'
ROOT = Path(__file__).resolve().parent
# 源码映射需要在 TPU 初始化之前装入编译器钩子；导入本模块时进入一次，整个进程保持，退出时恢复。
_SOURCE_MAPPING = contextlib.ExitStack()
_SOURCE_MAPPING.enter_context(compiler_source_mapping())
atexit.register(_SOURCE_MAPPING.close)
_INSTRUCTION = re.compile(r'^\s*\{?\s*([a-z]+[0-9]*): (?:@!?p[0-9]+ )?([a-z][\w.]*)')

def compile(function: Callable[..., Any], *arguments: Any, mesh: jax.sharding.Mesh | None = None, compiler_options: dict[str, str] | None = None, **jit_options: Any) -> Compiled:
    """编译并保留源码映射；mesh 是 shard_map 使用的 mesh，没有时省略；其余关键字参数（如 out_shardings）原样传给 jax.jit。"""
    jax.config.update('jax_enable_compilation_cache', False)
    # 设置 abstract mesh 后，kernel 中重复调用的 jnp 函数不会复用第一次追踪的 jaxpr，来源注释的行号才准确。
    abstract_mesh = jax.sharding.use_abstract_mesh(mesh.abstract_mesh) if mesh is not None else contextlib.nullcontext()
    with abstract_mesh:
        return jax.jit(function, compiler_options=compiler_options, **jit_options).lower(*arguments).compile()

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

def find_bundles(serialized: bytes, text: str, *, encoding: str = 'canonical', whole_program: bool = False) -> list[int]:
    """文本包含 text 的 bundle 编号，按顺序排列；默认只在 Pallas kernel 中查找，whole_program=True 时在整个程序（包括 runtime 代码）中查找。"""
    (_, _, image), = executable_programs(serialized)
    lines = format_assembly(image, target=TARGET, encoding=encoding).splitlines()
    spans = _bundle_spans(lines)
    candidates = range(len(spans)) if whole_program else pallas_bundles(serialized)
    # 只在指令文本中查找，不匹配行尾的源码注释。
    return [pc for pc in candidates if text in '\n'.join(line.split('#')[0] for line in lines[spans[pc][0]:spans[pc][1] + 1])]

def edit_bundles(serialized: bytes, edits: dict[int, tuple[str, str]]) -> bytes:
    """只在指定编号的 bundle 内做文本替换：edits[pc] = (原文本, 新文本)，原文本在该 bundle 中必须恰好出现一次。

    使用精确编码的清单，其余 bundle 逐字节保持不变；被修改的 bundle 去掉 `.encoding` 约束，由汇编器为它重新选择编码。
    """
    (record, index, image), = executable_programs(serialized)
    lines = format_assembly(image, target=TARGET).splitlines()
    spans = _bundle_spans(lines)
    # 从后往前改，前面 bundle 的行号不受影响。
    for pc, (old, new) in sorted(edits.items(), reverse=True):
        start, end = spans[pc]
        bundle = '\n'.join(lines[start:end + 1])
        assert bundle.count(old) == 1, (pc, bundle)
        bundle = re.sub(r'\s*;\s*\.encoding \{[^}]*\}', '', bundle.replace(old, new))
        lines[start:end + 1] = bundle.split('\n')
    return replace_executable_programs(serialized, {(record, index): assemble_listing('\n'.join(lines))})

def insert_bundles(serialized: bytes, insertions: dict[int, str]) -> bytes:
    """在原 bundle 编号 pc 之前插入一段清单（不含 `.target` 行），分支与元数据由 tpuasm 重定位。"""
    (record, index, _), = executable_programs(serialized)
    return insert_executable_bundles(serialized, {(record, index): [BundleInsertion(pc, f'.target {TARGET}\n{text}') for pc, text in sorted(insertions.items())]})

def load(serialized: bytes, template: Compiled, devices: list[jax.Device] | None = None) -> Compiled:
    """按 template 的调用约定装载改写后的 executable。程序跨越多个 host 时，devices 给出全部 device（jax.devices()），每个进程都要装载。"""
    return load_executable(serialized, template, devices=devices)

def bundle(text: str = '') -> str:
    """一个 bundle 的清单文本；text 为空时是空 bundle。"""
    return '{ ' + text + ' }\n'

def read_lcc(low: int) -> str:
    """在同一个 bundle 中读 LCC 的低、高 32 位：低位写入 s{low}，高位写入 s{low + 5}。"""
    return bundle(f's0: srdreg.lcclo s{low} ; s1: srdreg.lcchi s{low + 5}')

def read_gtc(low: int) -> str:
    """在同一个 bundle 中读 GTC 的低、高 32 位：低位写入 s{low}，高位写入 s{low + 5}。"""
    return bundle(f's0: srdreg.gtclo s{low} ; s1: srdreg.gtchi s{low + 5}')

SAVED = range(20, 31)

def compute_section(listing: str) -> list[str]:
    """kernel 清单中从输入 DMA 等待之后到下一次 DMA 等待之前的 bundle（每个是一段不带花括号的文本），去掉标量指令与信号量指令；只剩这些指令的 bundle 变成空 bundle。"""
    lines = [line.split('#')[0].rstrip() for line in listing.splitlines()]
    text = re.sub(r'\s*;\s*\.encoding \{[^}]*\}', '', ' '.join(line for line in lines if line.strip()))
    bundles = [body.strip() for body in re.findall(r'\{(.*?)\}', text)]
    first, second = [index for index, body in enumerate(bundles) if 'vwait' in body][:2]
    return [' ; '.join(item.strip() for item in body.split(';') if not item.strip().startswith(('s0:', 's1:', 'misc:'))) for body in bundles[first + 1:second]]

def section_program(section: list[str]) -> str:
    """把 compute_section 的结果包成 LccProbe 的片段：R0、这些 bundle、R1、sfence、R2。"""
    return read_lcc(20) + ''.join(bundle(text) for text in section) + read_lcc(21) + bundle('s0: sfence') + read_lcc(22)

def spill_saved(address: int = 0x80) -> str:
    """LccProbe 的 setup 用：把保存着载体标量寄存器的 v20–v30 写进 TC VMEM 从 address 起的 11 个 tile，供会改写这些 TC VREG 的片段使用。"""
    return ''.join(bundle(f'vst: vst.8x128 [vmem:0x{address + 8 * index:x}], v{register}') for index, register in enumerate(SAVED))

def reload_saved(address: int = 0x80) -> str:
    """接在片段最后一次读数之后：从 TC VMEM 读回 spill_saved 写出的 v20–v30。"""
    return ''.join(bundle(f'vld: vld.8x128 v{register}, [vmem:0x{address + 8 * index:x}]') for index, register in enumerate(SAVED)) + bundle('misc: vnop') * 16

class LccProbe:
    """在一个载体 kernel 中插入手写片段，用 LCC（或 GTC）读数计时（第一章第 3 节；参照 tpu-v4-latency-numbers 第 0 节）。

    片段约定：第 i 次读数用 read_lcc(20 + i) 或 read_gtc(20 + i)，i 从 0 起，最多 4 次；片段不得改写存放读数的 s20–s23、s25–s28；v20–v30 保存着载体的标量寄存器，片段若要改写，须在 setup 中用 spill_saved() 存起来、在最后一次读数之后用 reload_saved() 读回。读数经载体的输出 DMA 返回。片段之前已把 TC VMEM 地址 0 起的输入读进 v10；输入是 u32[256,128] 的随机数，片段可以把它当作数据。setup 在第一次读数之前执行，之后由一条 sfence 排空，不计入区间。num_cores=2 时，两个 TensorCore 执行同一段片段，各自返回读数。进程打开多颗芯片时，device 选择载体在第几颗芯片上运行。
    """

    def __init__(self, num_cores: int = 1, device: int = 0) -> None:
        self.num_cores = num_cores
        mesh = jax.make_mesh((1,), ('device',), devices=[jax.local_devices()[device]])
        tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=num_cores)

        @jax.shard_map(
            mesh=mesh,
            in_specs=P(),
            out_specs=P(),
            check_vma=False,
        )
        def probe(x: jax.Array) -> jax.Array:
            @pl.kernel(
                out_type=jax.ShapeDtypeStruct((num_cores * 72, 128), jnp.uint32),
                mesh=tc_mesh,
                scratch_types=(pltpu.VMEM((256, 128), jnp.uint32), pltpu.SemaphoreType.DMA),
                name='lcc_probe',
                compiler_params=pltpu.CompilerParams(
                    disable_bounds_checks=True,
                    disable_semaphore_checks=True,
                ),
            )
            def kernel(x_hbm, out_hbm, data, sem) -> None:
                pltpu.async_copy(x_hbm, data, sem).wait()
                # 带唯一立即数的 vxor 标出插入位置。
                data[:8, :] = data[:8, :] ^ jnp.uint32(0x13579BDF)
                core = jax.lax.axis_index('tc')
                pltpu.async_copy(data.at[:72, :], out_hbm.at[pl.ds(core * 72, 72)], sem).wait()

            return kernel(x)

        self.host = np.random.default_rng(0).integers(0, 1 << 32, (256, 128), dtype=np.uint64).astype(np.uint32)
        self.x = jax.device_put(self.host, jax.local_devices()[device])
        self.compiled = compile(probe, self.x, mesh=mesh)
        serialized = serialize(self.compiled)
        marker, = find_bundles(serialized, '0x13579bdf')
        text = bundle_text(serialized, marker, encoding='exact')
        instruction, = [item.strip(' {}\n') for item in text.split(';') if '0x13579bdf' in item]
        destination = instruction.split()[2].rstrip(',')
        # 标记本身改为把 v10 原样写出，使输出的第一个 tile 等于输入。
        self.serialized = edit_bundles(serialized, {marker: (instruction, f'{instruction.split(":")[0]}: vmov.8x128 {destination}, v10')})
        self.marker = marker

    def _load(self, body: str, reads: int, setup: str) -> Callable[[jax.Array], jax.Array]:
        """把 setup、片段和写回读数的后缀插入载体，返回可调用的 executable。"""
        return load(self.program(body, reads, setup), self.compiled)

    def program(self, body: str, reads: int = 3, setup: str = '') -> bytes:
        """插入 setup、片段和写回读数的后缀之后的 executable（序列化形式），可以用 full_listing 查看。"""
        # 载体在插入点之后还要用到的标量寄存器可能落在 s20–s30 中：先广播进 v20–v30 保存，最后经 vpush/spop 恢复。
        save = ''.join(bundle(f'va0: vmov.8x128 v{register}, s{register}') for register in SAVED) + bundle('s0: sfence')
        restore = ''.join(bundle(f'vst: vpush v2sf, v{register}') + bundle(f's0: spop s{register}, v2sf') for register in SAVED)
        prefix = save + bundle('vld: vld.8x128 v10, [vmem:0x0]') + bundle('s0: simm.s32 s24, 0') + bundle('misc: vnop') * 16 + setup + bundle('s0: sfence')
        suffix = bundle('misc: vnop') * 16
        registers = [*range(20, 20 + reads), *range(25, 25 + reads)]
        for tile, register in enumerate(registers, 1):
            suffix += bundle(f'va0: vmov.8x128 v12, s{register}') + bundle('misc: vnop') * 8 + bundle(f'vst: vst.8x128 [vmem:0x{tile * 8:x}], v12')
        suffix += bundle('misc: vnop') * 16 + bundle('s0: sfence') + restore + bundle('s0: sfence')
        return insert_bundles(self.serialized, {self.marker: prefix + body + suffix})

    def run_raw(self, body: str, reads: int, repeats: int = 8, setup: str = '') -> np.ndarray:
        """返回 (repeats, num_cores, reads) 的 64 位读数。"""
        function = self._load(body, reads, setup)
        rows = [core * 72 + tile * 8 for core in range(self.num_cores) for tile in range(1, 2 * reads + 1)]
        samples = []
        for _ in range(repeats):
            halves = np.asarray(function(self.x))[rows, 0].astype(np.uint64).reshape(self.num_cores, 2, reads)
            samples.append(halves[:, 0] | (halves[:, 1] << np.uint64(32)))
        return np.array(samples)

    def run_tiles(self, body: str, repeats: int = 1, setup: str = '') -> np.ndarray:
        """片段自己用 vst 把结果写进 TC VMEM 的第 1–8 个 tile（地址 0x8 到 0x40）；返回 (repeats, num_cores, 8, 8, 128) 的 u32 数组。"""
        function = self._load(body, 0, setup)
        return np.array([np.asarray(function(self.x)).reshape(self.num_cores, 9, 8, 128)[:, 1:] for _ in range(repeats)])

    def time_section(self, section: list[str]) -> tuple[int, int]:
        """把一段编译器生成的计算 bundle 原样插进载体执行，返回 (R1 − R0, R2 − R0)。这段代码可以改写任何 TC VREG：v20–v30 先存进 TC VMEM，读数之后读回。"""
        (first, second), = {tuple(row) for row in self.run(section_program(section) + reload_saved(), setup=spill_saved()).tolist()}
        return first, second

    def run(self, body: str, repeats: int = 8, setup: str = '') -> np.ndarray:
        """返回 (repeats, 2) 的数组：每次运行的 R1 − R0 与 R2 − R0（R0、R1、R2 是第 0、1、2 次读数）。"""
        counters = self.run_raw(body, 3, repeats, setup)[:, 0].astype(np.int64)
        return counters[:, 1:] - counters[:, :1]

LCC_END = read_lcc(21) + bundle('s0: sfence') + read_lcc(22)  # 片段的结尾：R1、sfence、R2

def show_lcc(probe: LccProbe, label: str, body: str, setup: str = '') -> None:
    """运行片段 8 次，打印出现过的 (R1 − R0, R2 − R0)。"""
    print(f'  {label}：(R1 − R0, R2 − R0) = {sorted({tuple(row) for row in probe.run(body, setup=setup).tolist()})}')

def show_latency(probe: LccProbe, name: str, issue: str, fetch: str, setup: str = '', after: str = '', distances: tuple[int, ...] = (1, 2, 4, 6, 7, 8, 12, 40)) -> None:
    """测一条“提交—取回”通路的延迟：在提交指令 issue 与取回指令 fetch 之间插入 d − 1 个 vnop。取回要等结果时 R2 − R0 不随 d 变化，结果已经就绪后每次加 1。after 是读数之后还要执行的清单（例如取走队列中剩下的结果）。"""
    print(f'## {name}')
    for distance in distances:
        show_lcc(probe, f'd = {distance:2d}', read_lcc(20) + bundle(issue) + bundle('misc: vnop') * (distance - 1) + bundle(fetch) + LCC_END + after, setup)

CLOCK_BASE = 0x20000  # SMEM 中存放 KernelClock 读数的位置：第 i 个读数的低、高 32 位在 CLOCK_BASE + 2i 与 + 2i + 1
CLOCK_SAVE = 0x20100  # 读数时借用 s30、s31，原值暂存在这里
CLOCK_OVERHEAD = 20  # 相邻两次 clock_read 之间，读数自身占用的周期数（第一章第 3 节实测）

def clock_read(index: int, counter: str = 'lcc') -> str:
    """读一次 LCC 并存进 SMEM 的一段清单：先 sfence 等此前的向量工作全部发射，再在同一个 bundle 中读 LCC 的两半。借用的 s30、s31 读完后恢复。counter='gtc' 时改读 GTC。"""
    return (
        bundle(f's1: sst [smem:0x{CLOCK_SAVE:x}], s30')
        + bundle(f's1: sst [smem:0x{CLOCK_SAVE + 1:x}], s31')
        + bundle('s0: sfence')
        + bundle(f's0: srdreg.{counter}lo s30 ; s1: srdreg.{counter}hi s31')
        + bundle(f's1: sst [smem:0x{CLOCK_BASE + 2 * index:x}], s30')
        + bundle(f's1: sst [smem:0x{CLOCK_BASE + 2 * index + 1:x}], s31')
        + bundle(f's1: sld s30, [smem:0x{CLOCK_SAVE:x}]')
        + bundle(f's1: sld s31, [smem:0x{CLOCK_SAVE + 1:x}]')
    )

def hlo_bounds(serialized: bytes) -> dict[int, tuple[int, int]]:
    """程序中每条 HLO 指令的 vtrace 起止标记所在的 bundle 编号：{指令序号: (vtrace 0x8… 的 bundle, vtrace 0x9… 的 bundle)}。"""
    (_, _, image), = executable_programs(serialized)
    lines = format_assembly(image, target=TARGET, encoding='canonical').splitlines()
    marks: dict[int, dict[int, int]] = {}
    for pc, (start, end) in enumerate(_bundle_spans(lines)):
        for kind, number in re.findall(r'vtrace 0x([89])([0-9a-f]{7})\b', '\n'.join(lines[start:end + 1])):
            marks.setdefault(int(number, 16), {})[int(kind)] = pc
    return {number: (found[8], found[9]) for number, found in sorted(marks.items()) if len(found) == 2}

MODULE = 0xFFFFFFF  # 整个程序（XProf 中的 module）的 vtrace 标记所用的序号

NO_CODE = ('parameter', 'get-tuple-element', 'bitcast', 'tuple')  # 不生成代码、因而没有起止标记的 HLO 指令
_HLO_INSTRUCTION = re.compile(r'^\s*(?:ROOT )?%([\w.\-]+) = .*?\s([\w\-]+)\((?:%|\d|\))', re.MULTILINE)  # 编译后的 HLO 中一条指令的名称与操作码

def hlo_ops(compiled: Compiled) -> list[tuple[str, int, int]]:
    """按执行顺序列出 (名称, 起始标记的 bundle, 结束标记的 bundle)：每条有 vtrace 标记的 HLO 指令一项，整个程序一项（名称为 module）。

    名称优先取自清单的注释：XLA 的指令在它的第一个 bundle 前有一行 `# entry bundle: %名称 = ...`，Pallas kernel 等 custom call 的函数段以 `# function ... [symbol N, 名称]` 开头。没有注释的指令（如 copy-start）按标记中的序号取名：序号是它在编译后入口函数的指令中的次序，但 get-tuple-element、bitcast 这类不生成代码的指令不占序号。
    """
    serialized = serialize(compiled)
    notes, pc = [], 0
    for line in full_listing(serialized).splitlines():
        if line.startswith('{'):
            pc += 1
        elif line.startswith('# entry bundle: %'):
            notes.append((pc, line.split('%')[1].split(' ')[0]))
        elif line.startswith('# function '):
            notes.append((pc, cast(re.Match[str], re.search(r'\[symbol \d+, ([^\]]+)\]', line)).group(1)))
    text = compiled.as_text()
    entry = text[text.index('\nENTRY'):].split('\n}')[0]
    ordered = [name for name, opcode in _HLO_INSTRUCTION.findall(entry) if opcode not in NO_CODE]
    result = []
    for number, (start, end) in hlo_bounds(serialized).items():
        noted = next((name for at, name in notes if start <= at <= end), None)
        result.append(('module' if number == MODULE else noted or (ordered[number] if number < len(ordered) else f'op{number}'), start, end))
    return sorted(result, key=lambda item: item[1])

class KernelClock:
    """给任意已编译的程序计时：在指定的 bundle 之前插入 clock_read，读数留在每个 TensorCore 的 SMEM 中，程序运行之后再用一个读取程序取回。

    SMEM 的内容在程序之间保留，所以被计时的程序不需要为读数增加任何输出。每个 TensorCore 各有自己的 SMEM 和 LCC，读数只能在同一个 TensorCore 内相减。进程打开多颗芯片时，device 选择从第几颗芯片取回读数。插入点在循环中时，留下的是最后一次迭代的读数。
    """

    def __init__(self, num_cores: int = 2, device: int = 0) -> None:
        self.probe = LccProbe(num_cores, device)
        self.readers: dict[tuple[int, int], Callable[[jax.Array], jax.Array]] = {}

    def instrument(self, compiled: Compiled, points: list[int], counter: str = 'lcc') -> Compiled:
        """在原 bundle 编号 points[i] 之前插入第 i 次读数，返回装载后的 executable。同一个编号可以出现多次，读数按出现的顺序排列。counter='gtc' 时读 GTC。"""
        insertions: dict[int, str] = {}
        for index, pc in enumerate(points):
            insertions[pc] = insertions.get(pc, '') + clock_read(index, counter)
        return load(insert_bundles(serialize(compiled), insertions), compiled)

    def interval(self, compiled: Compiled, first: int, last: int, call: Callable[[Compiled], Any], samples: int = 1, core: int = 0) -> list[int]:
        """区间 [first, last) 的周期数：在这两个 bundle 之前各读一次，call(executable) 运行一次程序，读数差减去读数自身的 20 个周期。返回 samples 次运行各自的值。"""
        timed = self.instrument(compiled, [first, last])
        cycles = []
        for _ in range(samples):
            call(timed)
            readings = self.read(2)[core].astype(np.int64)
            cycles.append(int(readings[1] - readings[0]) - CLOCK_OVERHEAD)
        return cycles

    def time_ops(self, compiled: Compiled, call: Callable[[Compiled], Any], samples: int = 8) -> list[tuple[str, list[int]]]:
        """给整个程序和其中每条 HLO 指令计时：在各自 vtrace 起止标记的位置插入读数，call(executable) 运行一次程序，共运行 samples 次。返回 [(名称, 每个 TensorCore 的周期数)]，取各次运行的中位数，已扣除读数自身的开销（区间内每次读数 20 个周期）。"""
        ops = hlo_ops(compiled)
        # 读数按 bundle 编号的先后执行；起始标记之前读一次，结束标记之后读一次。
        points = sorted([(start, 2 * k) for k, (_, start, _) in enumerate(ops)] + [(end + 1, 2 * k + 1) for k, (_, _, end) in enumerate(ops)])
        order = {tag: position for position, (_, tag) in enumerate(points)}
        timed = self.instrument(compiled, [pc for pc, _ in points])
        runs = []
        for _ in range(samples):
            call(timed)
            runs.append(self.read(len(points)).astype(np.int64))
        readings = np.array(runs)
        result = []
        for k, (name, _, _) in enumerate(ops):
            first, last = order[2 * k], order[2 * k + 1]
            # 先对每次运行求差，再取耗时的中位数。绝对 LCC 随运行单调递增，先取中位数只会挑到中间一两次运行。
            durations = readings[:, :, last] - readings[:, :, first] - CLOCK_OVERHEAD * (last - first)
            result.append((name, np.median(durations, axis=0).astype(np.int64).tolist()))
        return result

    def kernel_cycles(self, compiled: Compiled, call: Callable[[Compiled], Any], samples: int = 8, core: int = 0) -> int:
        """程序中的 kernel 在第 core 个 TensorCore 上的周期数。kernel 指除整个程序之外第一条有起止标记的 HLO 指令；它之后若还有 XLA 追加的 copy，不计在内。"""
        return [cycles for name, cycles in self.time_ops(compiled, call, samples) if name != 'module'][0][core]

    def read(self, count: int) -> np.ndarray:
        """取回最近一次运行留下的前 count 个读数：返回 (TensorCore 数, count) 的 64 位周期计数。"""
        words = []
        for first in range(0, 2 * count, 8):
            last = min(first + 8, 2 * count)
            if (first, last) not in self.readers:
                body = ''
                for tile, word in enumerate(range(first, last), 1):
                    body += bundle(f's1: sld s24, [smem:0x{CLOCK_BASE + word:x}]') + bundle('va0: vmov.8x128 v12, s24') + bundle('misc: vnop') * 8 + bundle(f'vst: vst.8x128 [vmem:0x{tile * 8:x}], v12')
                # 读取程序只装载一次，之后重复使用。
                self.readers[(first, last)] = self.probe._load(body, 0, '')
            tiles = np.asarray(self.readers[(first, last)](self.probe.x)).reshape(self.probe.num_cores, 9, 8, 128)[:, 1:]
            words.append(tiles[:, :last - first, 0, 0].astype(np.uint64))
        halves = np.concatenate(words, axis=1).reshape(self.probe.num_cores, count, 2)
        return halves[:, :, 0] | (halves[:, :, 1] << np.uint64(32))

# 以下是第一章第 3 节的 TC VMEM 计时边界：输入和每个结果都固定在 TC VMEM，只测中间的计算段。

def row_layout(x: jax.Array) -> Layout:
    """Pallas kernel 在 TC VMEM 中的二维数组所用的布局：行优先，按 8 个子通道乘 128 个通道切成 tile，HLO 中写作 {1,0:T(8,128)}；16 bit、8 bit 的元素再按 (2,1)、(4,1) 打包。"""
    packing = 4 // x.dtype.itemsize
    return Layout(major_to_minor=(0, 1), tiling=((8, 128), (packing, 1)) if packing > 1 else ((8, 128),))

def pinned(x: jax.Array, constrain: bool = True) -> jax.Array:
    """把一个值固定在 TC VMEM（HLO 的 layout 中出现 S(1)）；constrain 为 True 时再把二维数组的布局约束为 row_layout。布局约束必须加在 Unpin 之后。"""
    y = state.unpin(state.pin(x, to='vmem'))
    return with_layout_constraint(y, row_layout(x)) if constrain and x.ndim == 2 else y

def vmem_kernel(f: Callable[..., Any], name: str = 'kernel') -> Callable[..., Any]:
    """把函数 f 包成一个操作数和结果都在 TC VMEM 的 Pallas kernel：没有 DMA，也没有信号量。f 的每个参数、每个结果各对应一个 Ref。"""
    def function(*arguments: jax.Array) -> Any:
        shapes = jax.eval_shape(f, *arguments)
        several = isinstance(shapes, (tuple, list))
        outputs = tuple(shapes) if several else (shapes,)

        @pl.kernel(
            out_type=tuple(pltpu.VMEM(output.shape, output.dtype) for output in outputs),
            mesh=pltpu.TensorCoreMesh(axis_name='tc', num_cores=1),
            name=name,
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(*refs: Ref) -> None:
            results = f(*(ref[...] for ref in refs[:len(arguments)]))
            for ref, value in zip(refs[len(arguments):], results if several else (results,)):
                ref[...] = value

        results = kernel(*(pltpu.with_memory_space_constraint(argument, pltpu.VMEM) for argument in arguments))
        return results if several else results[0]

    return function

def vmem_shell(function: Callable[..., Any], mesh: jax.sharding.Mesh, *, constrain: bool = True, barrier: bool = True) -> Callable[..., Any]:
    """外壳：每个输入和每个结果都经过 pinned；多个输入、多个结果各自共同经过 optimization_barrier。"""
    def program(*arguments: jax.Array) -> Any:
        inputs = tuple(pinned(argument, constrain) for argument in arguments)
        result = function(*(jax.lax.optimization_barrier(inputs) if barrier and len(inputs) > 1 else inputs))
        if isinstance(result, jax.Array):
            return pinned(result, constrain)
        outputs = tuple(pinned(item, constrain) for item in result)
        return jax.lax.optimization_barrier(outputs) if barrier else outputs

    # out_specs 只有一个 P() 时，它对结果的每个叶子都适用。
    return jax.shard_map(
        program,
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )

def vmem_interval(compiled: Compiled, inputs: int = 1) -> tuple[int, int, list[str], list[str], list[str], str]:
    """计时区间 [first, last)：从最晚的一个输入的 Unpin 结束，到最晚的一个结果的 Unpin 结束。同时返回各个端点在 HLO 中的类型与 layout、区间内有代码的 HLO 指令、区间内的 DMA，以及区间内的清单。"""
    text = compiled.as_text()
    entry = text[text.index('\nENTRY'):].split('\n}')[0]
    layouts = [f'{name} 是 {layout}' for name, layout in re.findall(r'%(unpin[\w.]*) = (\S+) custom-call', entry)]
    ops = hlo_ops(compiled)
    unpins = [end for name, _, end in ops if name.startswith('unpin')]
    first, last = max(unpins[:inputs]) + 1, max(unpins[inputs:]) + 1
    inside = [name for name, start, end in ops if first <= start and end < last and not name.startswith(('pin', 'unpin'))]
    moves, lines, pc = [], [], -1
    for line in full_listing(serialize(compiled)).splitlines():
        pc += line.startswith('{')
        if first <= pc < last:
            lines.append(line)
            moves += [f'{source} → {destination}' for destination, source in re.findall(r'dma\.\w+ \[(\w+):[^\]]*\], \[(\w+):', line.split('#')[0])]
    return first, last, layouts, inside, moves, '\n'.join(lines)

def vector_counts(listing: str) -> Counter[str]:
    """清单中向量指令的条数（不计 vtrace 与 vnop）。"""
    return Counter({mnemonic: count for mnemonic, count in count_mnemonics(listing).items() if mnemonic.startswith(('v', 'cld')) and not mnemonic.startswith(('vtrace', 'vnop'))})

class VmemTiming:
    """一段计算在 TC VMEM 边界内的计时结果：cycles 是每份输入的周期数（读数差减 20），bundles 是区间内的 bundle 数，listing 是区间内的清单，result 是最后一份输入的输出。"""

    def __init__(self, cycles: list[int], first: int, last: int, layouts: list[str], inside: list[str], moves: list[str], listing: str, result: Any) -> None:
        self.cycles, self.first, self.last, self.layouts, self.inside, self.moves, self.listing, self.result = cycles, first, last, layouts, inside, moves, listing, result
        self.bundles = last - first

    def summary(self) -> str:
        """周期数的写法：每次相同时只写一个值，否则写中位数与范围。"""
        low, high = min(self.cycles), max(self.cycles)
        return str(low) if low == high else f'{int(np.median(self.cycles))}（{low}–{high}）'

def time_in_vmem(clock: KernelClock, function: Callable[..., Any], runs: list[tuple[Any, ...]], check: Callable[..., None] | None = None, **options: bool) -> VmemTiming:
    """把 function 放进 vmem_shell 编译，在区间两端各读一次 LCC。runs 的每一项是一份输入（参数的 tuple），各运行一次；check(result, *arguments) 核对每次的输出。"""
    mesh = jax.make_mesh((1,), ('device',))
    compiled = compile(vmem_shell(function, mesh, **options), *runs[0], mesh=mesh)
    first, last, layouts, inside, moves, listing = vmem_interval(compiled, len(runs[0]))
    timed = clock.instrument(compiled, [first, last])
    cycles = []
    for arguments in runs:
        result = timed(*arguments)
        jax.block_until_ready(result)
        readings = clock.read(2)[0].astype(np.int64)
        cycles.append(int(readings[1] - readings[0]) - CLOCK_OVERHEAD)
        if check is not None:
            check(result, *arguments)
    return VmemTiming(cycles, first, last, layouts, inside, moves, listing, jax.tree.map(np.asarray, result))

def compare_in_vmem(clock: KernelClock, name: str, f: Callable[..., Any], runs: list[tuple[Any, ...]], check: Callable[..., None] | None = None, pallas: Callable[..., Any] | None = None) -> dict[str, VmemTiming]:
    """同一个运算分别由原生 XLA 和 Pallas kernel 实现，放进同一个 TC VMEM 边界计时，打印周期数与边界的证据。pallas 省略时用 vmem_kernel(f)；编译失败时打印报错的第一行。"""
    print(f'## {name}')
    timings = {}
    for kind, function in (('XLA', f), ('Pallas', pallas or vmem_kernel(f))):
        try:
            timing = time_in_vmem(clock, function, runs, check)
        except Exception as error:
            print(f'  {kind}：编译失败：{str(error).split(", metadata=")[0].splitlines()[0][:200]}')
            continue
        counts = vector_counts(timing.listing)
        print(f'  {kind}：{timing.summary()} 个周期；区间 {timing.bundles} 个 bundle，HLO 指令：{"、".join(timing.inside)}；DMA：{"、".join(sorted(set(timing.moves))) or "无"}')
        print(f'    端点：{"；".join(timing.layouts)}')
        print(f'    向量指令：{"，".join(f"{mnemonic} {counts[mnemonic]}" for mnemonic in sorted(counts))}')
        timings[kind] = timing
    if len(timings) == 2:
        same = all(np.array_equal(a, b, equal_nan=True) for a, b in zip(jax.tree.leaves(timings['XLA'].result), jax.tree.leaves(timings['Pallas'].result)))
        print(f'  Pallas 与 XLA 的结果逐元素相同：{same}')
    return timings
