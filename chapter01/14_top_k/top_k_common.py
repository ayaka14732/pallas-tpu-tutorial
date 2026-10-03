"""本节各实验共用：有两个输出（值与下标）的 kernel，以及 argmax 与用掩码排除已选位置的手写 top-k。"""
from collections.abc import Callable

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

import tpuasm_tools

def run(f: Callable[[jax.Array], tuple[jax.Array, jax.Array]], x: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], str]:
    """f(x) 返回 (values, indices)；kernel 有两个输出。"""
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)
    values, indices = jax.eval_shape(f, jax.ShapeDtypeStruct(x.shape, x.dtype))

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=(P(), P()),
        check_vma=False,
    )
    def function(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        @pl.kernel(
            out_type=(pltpu.HBM(values.shape, values.dtype), pltpu.HBM(indices.shape, indices.dtype)),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.VMEM(values.shape, values.dtype), pltpu.VMEM(indices.shape, indices.dtype), pltpu.SemaphoreType.DMA),
            name='top_k',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, v_hbm: Ref, i_hbm: Ref, x_vmem: Ref, v_vmem: Ref, i_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            v_vmem[...], i_vmem[...] = f(x_vmem[...])
            pltpu.async_copy(v_vmem, v_hbm, sem).wait()
            pltpu.async_copy(i_vmem, i_hbm, sem).wait()

        return kernel(x)

    compiled = tpuasm_tools.compile(function, x, mesh=mesh)
    values, indices = compiled(x)
    return (np.asarray(values), np.asarray(indices)), tpuasm_tools.kernel_listing(compiled, pallas_only=True)

def argmax(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return jnp.max(x, axis=1, keepdims=True), jnp.argmax(x, axis=1, keepdims=True).astype(jnp.int32)

def top_k_by_hand(k: int) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
    """每轮取未选位置中的最大值，再在等于它的未选位置中取最小的下标；已选位置用单独的掩码排除，不靠把值改成 -inf。"""
    def f(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1).astype(jnp.float32)
        taken = jnp.zeros(x.shape, jnp.bool_)
        values, indices = [], []
        for _ in range(k):
            best = jnp.max(jnp.where(taken, -jnp.inf, x), axis=1, keepdims=True)
            index = jnp.min(jnp.where(~taken & (x == best), lane, float(x.shape[1])), axis=1, keepdims=True)
            taken = taken | (lane == index)
            values.append(best)
            indices.append(index.astype(jnp.int32))
        return jnp.concatenate(values, axis=1), jnp.concatenate(indices, axis=1)

    return f
