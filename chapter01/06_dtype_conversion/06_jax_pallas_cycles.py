"""各种类型转换在 TC VMEM 边界内的周期数；原生 XLA 与 Pallas kernel 各测一次。"""
import tpu_init
tpu_init.initialize_one_core()

import jax.numpy as jnp
import ml_dtypes
import numpy as np

import tpuasm_tools

def main() -> None:
    clock = tpuasm_tools.KernelClock(num_cores=1)
    rng = np.random.default_rng(0)
    floats = (rng.standard_normal((64, 128)) * 100).astype(np.float32)
    cases = (
        ('bf16[64,128] → f32', floats.astype(ml_dtypes.bfloat16), jnp.float32),
        ('f32[64,128] → bf16', floats, jnp.bfloat16),
        ('f32[64,128] → int32', floats, jnp.int32),
        ('int32[64,128] → f32', np.round(floats).astype(np.int32), jnp.float32),
        ('int8[64,128] → int32', np.clip(floats, -128, 127).astype(np.int8), jnp.int32),
        ('int32[64,128] → int8', np.clip(floats, -128, 127).astype(np.int32), jnp.int8),
    )
    for name, x, dtype in cases:
        tpuasm_tools.compare_in_vmem(clock, name, lambda x: x.astype(dtype), [(x,)] * 4)

if __name__ == '__main__':
    main()
