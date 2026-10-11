"""归约在 TC VMEM 边界内的周期数：沿通道、沿子通道、跨多个 tile；原生 XLA 与 Pallas kernel 各测一次。最后改两种写法：归约到一个数时不经过标量单元，跨 tile 的累加从一条链改成几条互不依赖的链。"""
import tpu_init
tpu_init.initialize_one_core()

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

import tpuasm_tools

def chained_sum(chains: int) -> Callable[[jax.Array], jax.Array]:
    """沿 axis=0 求和：各个 tile 轮流加到 chains 个累加器上，最后把累加器相加，再在一个 TC VREG 内归约。"""
    def f(x: jax.Array) -> jax.Array:
        tiles = [x[start:start + 8] for start in range(0, x.shape[0], 8)]
        sums = tiles[:chains]
        for index, tile in enumerate(tiles[chains:]):
            sums[index % chains] = sums[index % chains] + tile
        while len(sums) > 1:
            sums = [a + b for a, b in zip(sums[0::2], sums[1::2])]
        return jnp.sum(sums[0], axis=0, keepdims=True)

    return f

def main() -> None:
    clock = tpuasm_tools.KernelClock(num_cores=1)
    rng = np.random.default_rng(0)
    f32 = lambda *shape: rng.integers(-100, 100, shape).astype(np.float32)
    cases = (
        ('sum(x, axis=1)，f32[8,128]', lambda x: jnp.sum(x, axis=1, keepdims=True), f32(8, 128)),
        ('max(x, axis=1)，f32[8,128]', lambda x: jnp.max(x, axis=1, keepdims=True), f32(8, 128)),
        ('sum(x, axis=1)，f32[64,128]', lambda x: jnp.sum(x, axis=1, keepdims=True), f32(64, 128)),
        ('sum(x, axis=1)，f32[1024,128]', lambda x: jnp.sum(x, axis=1, keepdims=True), f32(1024, 128)),
        ('sum(x, axis=1)，f32[8,1024]', lambda x: jnp.sum(x, axis=1, keepdims=True), f32(8, 1024)),
        ('sum(x, axis=0)，f32[8,128]', lambda x: jnp.sum(x, axis=0, keepdims=True), f32(8, 128)),
        ('max(x, axis=0)，f32[8,128]', lambda x: jnp.max(x, axis=0, keepdims=True), f32(8, 128)),
        ('sum(x, axis=0)，f32[64,128]', lambda x: jnp.sum(x, axis=0, keepdims=True), f32(64, 128)),
        ('sum(x, axis=0)，f32[1024,128]', lambda x: jnp.sum(x, axis=0, keepdims=True), f32(1024, 128)),
        ('sum(x)，f32[8,128]', lambda x: jnp.sum(x, keepdims=True), f32(8, 128)),
    )
    for name, f, x in cases:
        tpuasm_tools.compare_in_vmem(clock, name, f, [(x,)] * 4)
    small = f32(8, 128)
    tpuasm_tools.compare_in_vmem(clock, 'sum(x)，f32[8,128]；Pallas 改为两步都保留维度的归约', lambda x: jnp.sum(x, keepdims=True), [(small,)] * 4, pallas=tpuasm_tools.vmem_kernel(lambda x: jnp.sum(jnp.sum(x, axis=1, keepdims=True), axis=0, keepdims=True)))
    tall = f32(1024, 128)
    for chains in (2, 4):
        tpuasm_tools.compare_in_vmem(clock, f'sum(x, axis=0)，f32[1024,128]；Pallas 改用 {chains} 条互不依赖的累加链', lambda x: jnp.sum(x, axis=0, keepdims=True), [(tall,)] * 4, pallas=tpuasm_tools.vmem_kernel(chained_sum(chains)))

if __name__ == '__main__':
    main()
