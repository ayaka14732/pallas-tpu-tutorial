"""本节实验共用：单输入单 TC kernel，输出的 shape 与 dtype 由运算推出；统计清单中的计算指令。"""
from collections.abc import Callable

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P
import numpy as np

import tpuasm_tools

SKIPPED = ('vld', 'vst', 'vsync', 'vwait', 'vtrace', 'cfence', 'dma.', 's', 'p')

def run(f: Callable[[jax.Array], jax.Array], x: np.ndarray) -> tuple[np.ndarray, str]:
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)
    out = jax.eval_shape(f, jax.ShapeDtypeStruct(x.shape, x.dtype))

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def function(x: jax.Array) -> jax.Array:
        @pl.kernel(
            # 输出明确放在 HBM：否则 XLA 可能把窄的小输出放进 Megacore Shared CMEM。
            out_type=pltpu.HBM(out.shape, out.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.VMEM(out.shape, out.dtype), pltpu.SemaphoreType.DMA),
            name='reduce',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, o_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            o_vmem[...] = f(x_vmem[...])
            pltpu.async_copy(o_vmem, o_hbm, sem).wait()

        return kernel(x)

    compiled = tpuasm_tools.compile(function, x, mesh=mesh)
    # 窄输出在 HBM 中的 layout 与 kernel 写出的不同，XLA 会在 kernel 之后加一段重排；这里只看 kernel 本身。
    return np.asarray(compiled(x)), tpuasm_tools.kernel_listing(compiled, pallas_only=True)

def describe(listing: str) -> str:
    counts = tpuasm_tools.count_mnemonics(listing)
    return '、'.join(f'{mnemonic}×{count}' for mnemonic, count in sorted(counts.items()) if not mnemonic.startswith(SKIPPED)) or '（无计算指令）'
