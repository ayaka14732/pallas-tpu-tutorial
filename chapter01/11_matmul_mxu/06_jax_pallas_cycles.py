"""矩阵乘法在 TC VMEM 边界内的周期数：只改 M（LHS 的行数）、只改 dtype 与精度；原生 XLA 与 Pallas kernel 各测一次。"""
import tpu_init
tpu_init.initialize_one_core()

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

import tpuasm_tools

def main() -> None:
    clock = tpuasm_tools.KernelClock(num_cores=1)
    rng = np.random.default_rng(0)
    bf16 = lambda *shape: rng.standard_normal(shape, dtype=np.float32).astype(ml_dtypes.bfloat16)
    f32 = lambda *shape: rng.standard_normal(shape, dtype=np.float32)
    dot = lambda precision=None: lambda x, y: jnp.dot(x, y, preferred_element_type=jnp.float32, precision=precision)
    cases = (
        ('bf16[16,128] @ bf16[128,128]', dot(), (bf16(16, 128), bf16(128, 128))),
        ('bf16[128,128] @ bf16[128,128]', dot(), (bf16(128, 128), bf16(128, 128))),
        ('bf16[1024,128] @ bf16[128,128]', dot(), (bf16(1024, 128), bf16(128, 128))),
        ('f32[128,128] @ f32[128,128]，默认精度', dot(), (f32(128, 128), f32(128, 128))),
        ('f32[128,128] @ f32[128,128]，precision=HIGHEST', dot(jax.lax.Precision.HIGHEST), (f32(128, 128), f32(128, 128))),
        ('bf16[128,256] @ bf16[256,128]', dot(), (bf16(128, 256), bf16(256, 128))),
        ('bf16[128,128] @ bf16[128,256]', dot(), (bf16(128, 128), bf16(128, 256))),
    )
    for name, f, arguments in cases:
        tpuasm_tools.compare_in_vmem(clock, name, f, [arguments] * 4)

if __name__ == '__main__':
    main()
