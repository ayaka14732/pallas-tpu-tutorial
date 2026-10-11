"""循环的周期数：f32[512,128] 的 64 个 tile 逐个乘以 2，分别用 Python for（展开成直线代码）、pl.loop 和几种 unroll 写出，放进 TC VMEM 边界计时，并列出循环体的清单。"""
import tpu_init
tpu_init.initialize_one_core()

from collections.abc import Callable
import re

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import numpy as np

import tpuasm_tools

TILES = 64

def scale(style: str, unroll: int = 1) -> Callable[[jax.Array], jax.Array]:
    """操作数和结果都在 TC VMEM 的 kernel：逐个 tile 读入、乘以 2、写出。"""
    def function(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=pltpu.VMEM(x.shape, x.dtype),
            mesh=pltpu.TensorCoreMesh(axis_name='tc', num_cores=1),
            name='scale',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_vmem: Ref, o_vmem: Ref) -> None:
            if style == 'for':
                for i in range(TILES):
                    o_vmem[pl.ds(i * 8, 8)] = x_vmem[pl.ds(i * 8, 8)] * 2.0
            else:
                @pl.loop(0, TILES, unroll=unroll)
                def _(i: jax.Array) -> None:
                    start = i * 8
                    o_vmem[pl.ds(start, 8)] = x_vmem[pl.ds(start, 8)] * 2.0

        return kernel(pltpu.with_memory_space_constraint(x, pltpu.VMEM))

    return function

def check(result: jax.Array, x: np.ndarray) -> None:
    np.testing.assert_array_equal(np.asarray(result), x * 2.0)

def main() -> None:
    clock = tpuasm_tools.KernelClock(num_cores=1)
    x = np.random.default_rng(0).standard_normal((TILES * 8, 128), dtype=np.float32)
    cases = (
        ('Python for（展开）', scale('for')),
        ('pl.loop(0, 64)', scale('loop')),
        ('pl.loop(0, 64, unroll=2)', scale('loop', 2)),
        ('pl.loop(0, 64, unroll=4)', scale('loop', 4)),
        ('pl.loop(0, 64, unroll=8)', scale('loop', 8)),
    )
    for name, function in cases:
        timing = tpuasm_tools.time_in_vmem(clock, function, [(x,)] * 4, check)
        print(f'## {name}：{timing.summary()} 个周期；区间 {timing.bundles} 个 bundle')
        if name.startswith('pl.loop') and 'unroll=8' not in name:
            print('\n'.join('  ' + line for line in loop_body(timing.listing)))

def loop_body(listing: str) -> list[str]:
    """清单中的循环体，每个 bundle 写成一行：从跳转目标的标号到跳转指令之后的延迟槽。"""
    entries, current = [], ''
    for line in listing.splitlines():
        text = ' '.join(re.sub(r';?\s*\.encoding \{[^}]*\}', '', line.split('#')[0]).split())
        if not text:
            continue
        current = f'{current} {text}'.strip()
        if text.startswith('L_') or text.endswith('}'):
            entries.append(current.replace(' ; }', ' }'))
            current = ''
    start = next(index for index, entry in enumerate(entries) if entry.startswith('L_'))
    branch = next(index for index, entry in enumerate(entries) if 'sbr.rel' in entry and index > start)
    return entries[start:branch + 2]

if __name__ == '__main__':
    main()
