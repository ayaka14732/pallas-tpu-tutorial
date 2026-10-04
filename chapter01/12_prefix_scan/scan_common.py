"""本节实验共用：单输入单 TC kernel，输出的 shape 与 dtype 由运算推出；几种手写的扫描；统计清单中的计算指令。"""
from collections.abc import Callable

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
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
            name='scan',
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

def hillis_steele_lanes(x: jax.Array) -> jax.Array:
    """沿 lane 的包含式前缀和：第 d 轮把 lane l 加上 lane l - 2^d 的值，共 7 轮。"""
    lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    for d in range(7):
        shift = 1 << d
        x = x + jnp.where(lane >= shift, pltpu.roll(x, shift, axis=1), 0)
    return x

def hillis_steele_sublanes(x: jax.Array) -> jax.Array:
    """沿 sublane 的包含式前缀和：第 d 轮把 sublane s 加上 sublane s - 2^d 的值，共 3 轮。"""
    sublane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 0)
    for d in range(3):
        shift = 1 << d
        x = x + jnp.where(sublane >= shift, pltpu.roll(x, shift, axis=0), 0)
    return x

def suffix_sublanes(x: jax.Array) -> jax.Array:
    """沿 sublane 的后缀和：第 d 轮把 sublane s 加上 sublane s + 2^d 的值；roll 位移取 8 - 2^d。"""
    sublane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 0)
    for d in range(3):
        shift = 1 << d
        x = x + jnp.where(sublane < 8 - shift, pltpu.roll(x, 8 - shift, axis=0), 0)
    return x

def run_row_serial(style: str, x: np.ndarray) -> tuple[np.ndarray, str]:
    """沿 sublane 逐行串行累加的 kernel：style 为 'broadcast_to' 或 'stride=0'，决定怎样把一行广播到 8 个 sublane。"""
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def scan(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.VMEM(x.shape, x.dtype), pltpu.SemaphoreType.DMA),
            name='scan',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, o_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            total = jnp.zeros(x_vmem.shape, x_vmem.dtype)
            for k in range(8):
                if style == 'broadcast_to':
                    row = jnp.broadcast_to(x_vmem[pl.ds(k, 1), :], x_vmem.shape)
                else:
                    # 从第 k 行起读 8 行、行距为 0：8 个 sublane 都是第 k 行。
                    row = x_vmem[pl.ds(k, 8, stride=0), :]
                total = total + row
                o_vmem[pl.ds(k, 1), :] = total[0:1, :]
            pltpu.async_copy(o_vmem, o_hbm, sem).wait()

        return kernel(x)

    compiled = tpuasm_tools.compile(scan, jnp.asarray(x), mesh=mesh)
    return np.asarray(compiled(jnp.asarray(x))), tpuasm_tools.kernel_listing(compiled, pallas_only=True)
