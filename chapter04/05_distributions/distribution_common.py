"""本节各实验共用：生成随机比特并做变换的 kernel，以及从清单中统计变换本身的指令。"""
from collections import Counter
from collections.abc import Callable

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

import tpuasm_tools

ROWS = 512  # 每次生成 u32[512,128]：64 个 TC VREG，65536 个随机数
TILES = ROWS // 8
NOT_COMPUTE = ('vld', 'vst', 'vsync', 'vwait', 'vtrace', 'vnop', 'vsettm', 'vdelay')

def bits() -> jax.Array:
    """u32[ROWS,128] 的硬件随机比特。"""
    return pltpu.prng_random_bits((ROWS, 128)).astype(jnp.uint32)

def build(sample: Callable[..., tuple[jax.Array, ...]], dtypes: tuple, inputs: tuple = ()):
    """kernel 先 prng_seed(1)，再调用 sample(*输入的值) 得到若干个 [ROWS,128] 的数组，依次写出。inputs 是若干个 f32[ROWS,128] 输入。"""
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=tuple(P() for _ in inputs),
        out_specs=tuple(P() for _ in dtypes),
        check_vma=False,
    )
    def draw(*arrays: jax.Array):
        @pl.kernel(
            out_type=tuple(jax.ShapeDtypeStruct((ROWS, 128), dtype) for dtype in dtypes),
            mesh=tc_mesh,
            scratch_types=(*(pltpu.VMEM((ROWS, 128), jnp.float32) for _ in inputs), *(pltpu.VMEM((ROWS, 128), dtype) for dtype in dtypes), pltpu.SemaphoreType.DMA),
            name='draw',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(*refs: Ref) -> None:
            count_in, count_out = len(inputs), len(dtypes)
            sources, outputs = refs[:count_in], refs[count_in:count_in + count_out]
            input_buffers = refs[count_in + count_out:2 * count_in + count_out]
            buffers, sem = refs[2 * count_in + count_out:-1], refs[-1]
            for source, buffer in zip(sources, input_buffers, strict=True):
                pltpu.async_copy(source, buffer, sem).wait()
            pltpu.prng_seed(1)
            for buffer, value in zip(buffers, sample(*(buffer[...] for buffer in input_buffers)), strict=True):
                buffer[...] = value
            for buffer, output in zip(buffers, outputs, strict=True):
                pltpu.async_copy(buffer, output, sem).wait()

        return kernel(*arrays)

    return mesh, draw

def compile_and_run(sample: Callable[..., tuple[jax.Array, ...]], dtypes: tuple, inputs: tuple = ()) -> tuple[list[np.ndarray], str]:
    """编译、运行，返回 (输出, kernel 段清单)。"""
    mesh, draw = build(sample, dtypes, inputs)
    compiled = tpuasm_tools.compile(draw, *inputs, mesh=mesh)
    return [np.asarray(value) for value in compiled(*inputs)], tpuasm_tools.kernel_listing(compiled, pallas_only=True)

def compute_counts(listing: str) -> Counter[str]:
    return Counter({name: count for name, count in tpuasm_tools.count_mnemonics(listing).items() if name.startswith('v') and not name.startswith(NOT_COMPUTE)})

def per_tile(listing: str, baseline: str) -> str:
    """变换本身每个 TC VREG 的计算指令：减去只生成比特的 kernel 的指令，再除以 TC VREG 的个数。"""
    difference = compute_counts(listing) - compute_counts(baseline)
    total = sum(difference.values()) / TILES
    return f'{total:.1f} 条：' + '，'.join(f'{name} {count / TILES:g}' for name, count in difference.most_common())

def excerpt(listing: str, first: str, count: int) -> str:
    """清单中从第一条包含 first 的 bundle 起的 count 个 bundle，去掉源码注释。"""
    lines = [line.split('#')[0].rstrip() for line in listing.splitlines()]
    bundles, current = [], []
    for line in lines:
        if line.startswith('{'):
            current = [line]
        elif current:
            current.append(line)
        if current and line.endswith('}'):
            bundles.append(' '.join(part.strip() for part in current))
            current = []
    start = next(index for index, text in enumerate(bundles) if first in text)
    return '\n'.join('  ' + text for text in bundles[start:start + count])
