"""拼接在 TC VMEM 边界内的周期数：对齐与不对齐的 shape，沿行与沿列；原生 XLA 与 Pallas kernel 各测一次。"""
import tpu_init
tpu_init.initialize_one_core()

import jax.numpy as jnp
import numpy as np

import tpuasm_tools

def main() -> None:
    clock = tpuasm_tools.KernelClock(num_cores=1)
    rng = np.random.default_rng(0)
    f32 = lambda *shape: rng.standard_normal(shape, dtype=np.float32)
    cases = (
        ('沿行拼接，f32[8,128] 与 f32[8,128]', 0, (f32(8, 128), f32(8, 128))),
        ('沿列拼接，f32[8,128] 与 f32[8,128]', 1, (f32(8, 128), f32(8, 128))),
        ('沿行拼接，f32[9,128] 与 f32[9,128]', 0, (f32(9, 128), f32(9, 128))),
        ('沿列拼接，f32[8,129] 与 f32[8,129]', 1, (f32(8, 129), f32(8, 129))),
        ('沿行拼接，f32[64,128] 与 f32[64,128]', 0, (f32(64, 128), f32(64, 128))),
    )
    for name, axis, arguments in cases:
        tpuasm_tools.compare_in_vmem(clock, name, lambda a, b: jnp.concatenate([a, b], axis=axis), [arguments] * 4)

if __name__ == '__main__':
    main()
