"""前缀和在 TC VMEM 边界内的周期数：原生 XLA 的 jnp.cumsum 与本节的几种手写写法，沿通道与沿子通道各一组。"""
import tpu_init
tpu_init.initialize_one_core()

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

import scan_common
import tpuasm_tools

def by_matmul(x: jax.Array) -> jax.Array:
    """沿通道的前缀和写成 x @ U，U 是上三角全 1 的矩阵；默认精度。"""
    row = jax.lax.broadcasted_iota(jnp.int32, (128, 128), 0)
    column = jax.lax.broadcasted_iota(jnp.int32, (128, 128), 1)
    return jnp.dot(x, (row <= column).astype(jnp.float32), preferred_element_type=jnp.float32)

def row_serial(x: jax.Array) -> jax.Array:
    """沿子通道逐行累加：每次用行距为 0 的读把第 k 行广播到 8 个子通道。操作数和结果都在 TC VMEM。"""
    @pl.kernel(
        out_type=pltpu.VMEM(x.shape, x.dtype),
        mesh=pltpu.TensorCoreMesh(axis_name='tc', num_cores=1),
        name='scan',
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            disable_semaphore_checks=True,
        ),
    )
    def kernel(x_vmem: Ref, o_vmem: Ref) -> None:
        total = jnp.zeros(x_vmem.shape, x_vmem.dtype)
        for k in range(8):
            total = total + x_vmem[pl.ds(k, 8, stride=0), :]
            o_vmem[pl.ds(k, 1), :] = total[0:1, :]

    return kernel(pltpu.with_memory_space_constraint(x, pltpu.VMEM))

def main() -> None:
    clock = tpuasm_tools.KernelClock(num_cores=1)
    x = np.random.default_rng(0).integers(-100, 100, (8, 128)).astype(np.float32)
    cases = (
        ('沿通道：cumsum(x, axis=1)；Pallas 用 Hillis–Steele', 1, tpuasm_tools.vmem_kernel(scan_common.hillis_steele_lanes)),
        ('沿通道：cumsum(x, axis=1)；Pallas 用 MXU', 1, tpuasm_tools.vmem_kernel(by_matmul)),
        ('沿子通道：cumsum(x, axis=0)；Pallas 用 Hillis–Steele', 0, tpuasm_tools.vmem_kernel(scan_common.hillis_steele_sublanes)),
        ('沿子通道：cumsum(x, axis=0)；Pallas 逐行广播', 0, row_serial),
    )
    for name, axis, pallas in cases:
        tpuasm_tools.compare_in_vmem(clock, name, lambda x: jnp.cumsum(x, axis=axis), [(x,)] * 4, pallas=pallas)

if __name__ == '__main__':
    main()
