"""本节实验共用：把逐元素函数 f(x, y) 编译成单 TC kernel，返回结果与 kernel 段中的计算指令。"""
from collections.abc import Callable

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P
import numpy as np

import tpuasm_tools

# 只统计计算指令，不统计 load/store、DMA、同步和标量指令。
SKIPPED = ('vld', 'vst', 'vsync', 'vwait', 'vtrace', 'cfence', 'dma.', 's', 'p')

def run(f: Callable[[jax.Array, jax.Array], jax.Array], x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[str, int], str]:
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)
    out = jax.eval_shape(f, jax.ShapeDtypeStruct(x.shape, x.dtype), jax.ShapeDtypeStruct(y.shape, y.dtype))

    @jax.shard_map(
        mesh=mesh,
        in_specs=(P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    def function(x: jax.Array, y: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=out,
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.VMEM(y.shape, y.dtype), pltpu.VMEM(out.shape, out.dtype), pltpu.SemaphoreType.DMA),
            name='elementwise',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, y_hbm: Ref, o_hbm: Ref, x_vmem: Ref, y_vmem: Ref, o_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            pltpu.async_copy(y_hbm, y_vmem, sem).wait()
            o_vmem[...] = f(x_vmem[...], y_vmem[...])
            pltpu.async_copy(o_vmem, o_hbm, sem).wait()

        return kernel(x, y)

    compiled = tpuasm_tools.compile(function, x, y, mesh=mesh)
    listing = tpuasm_tools.kernel_listing(compiled)
    counts = {mnemonic: count for mnemonic, count in sorted(tpuasm_tools.count_mnemonics(listing).items()) if not mnemonic.startswith(SKIPPED)}
    return np.asarray(compiled(x, y)), counts, listing

def describe(counts: dict[str, int]) -> str:
    return '、'.join(f'{mnemonic}×{count}' for mnemonic, count in counts.items()) or '（无计算指令）'
