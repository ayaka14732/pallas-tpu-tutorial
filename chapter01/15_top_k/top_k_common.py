"""本节各实验共用：有两个输出（值与下标）的 kernel，以及 argmax 与三种手写 top-k（用掩码排除已选位置、抬高、秩计数）。"""
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

def top_k_lifted(k: int) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
    """抬高：lax.top_k 的写法之外，数出每行不等于 -inf 的元素个数 m；选完这 m 个之后，把原有的 -inf 抬到最小的有限值，已选的位置仍是 -inf。只多一次不在链上的归约。"""
    def f(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
        lowest = jnp.finfo(x.dtype).min
        original = x == -jnp.inf
        m = jnp.sum(jnp.where(original, 0.0, 1.0), axis=1, keepdims=True)
        # 每个原有的 -inf 在第几轮之后被抬高；m 为 0 时第 0 轮已经在 -inf 中选，从第 1 轮起抬高。
        lift_round = jnp.where(original, jnp.maximum(m, 1.0), -1.0)
        current = x
        values, indices = [], []
        for j in range(k):
            index = jnp.argmax(current, axis=1, keepdims=True).astype(jnp.int32)
            # 抬高之后选出的值是最小的有限值，输出时改回 -inf。
            values.append(jnp.where(m <= j, -jnp.inf, jnp.max(current, axis=1, keepdims=True)))
            indices.append(index)
            current = jnp.where(lane == index, -jnp.inf, jnp.where(lift_round == j + 1, lowest, current))
        return jnp.concatenate(values, axis=1), jnp.concatenate(indices, axis=1)

    return f

def top_k_by_rank(k: int) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
    """秩计数：把一行循环移动 1 到 n − 1 个通道，各与原来的行比较一次，数出每个元素前面有几个元素；秩小于 k 的就是结果。各次移位互不依赖。"""
    def f(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        n = x.shape[1]
        lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
        # f32 位型的整数键：按有符号整数比较的顺序就是浮点数的全序，NaN、-inf 都只是键的一个取值。
        bits = jax.lax.bitcast_convert_type(x, jnp.int32)
        key = jnp.where(bits < 0, bits ^ 0x7fffffff, bits)
        # 相等的元素规定下标小的在前：与下标更小的元素比较时用 key − 1。键是 INT_MIN 时减一会回绕，这些位置的秩用自己的下标预先补上。
        count = jnp.where(key == -2**31, lane, 0)
        lower = key - 1
        for distance in range(1, n):
            other = pltpu.roll(key, distance, axis=1)
            count += jnp.where(other > jnp.where(lane >= distance, lower, key), 1, 0)
        indices = jnp.concatenate([jnp.max(jnp.where(count == rank, lane.astype(jnp.float32), -1.0), axis=1, keepdims=True) for rank in range(k)], axis=1).astype(jnp.int32)
        return jnp.take_along_axis(x, indices, axis=1), indices

    return f
