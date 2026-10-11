"""x * 2 在 TC VMEM 边界内的周期数：f32、bf16，以及不整除 8×128 的 shape；原生 XLA 与 Pallas kernel 各测一次。"""
import tpu_init
tpu_init.initialize_one_core()

import ml_dtypes
import numpy as np

import tpuasm_tools

def main() -> None:
    clock = tpuasm_tools.KernelClock(num_cores=1)
    rng = np.random.default_rng(0)
    cases = (
        ('x * 2，f32[64,128]', rng.standard_normal((64, 128), dtype=np.float32)),
        ('x * 2，f32[128,128]', rng.standard_normal((128, 128), dtype=np.float32)),
        ('x * 2，bf16[64,128]', rng.standard_normal((64, 128), dtype=np.float32).astype(ml_dtypes.bfloat16)),
        ('x * 2，bf16[128,128]', rng.standard_normal((128, 128), dtype=np.float32).astype(ml_dtypes.bfloat16)),
        ('x * 2，f32[9,130]', rng.standard_normal((9, 130), dtype=np.float32)),
    )
    for name, x in cases:
        tpuasm_tools.compare_in_vmem(clock, name, lambda x: x * 2, [(x,)] * 4)

if __name__ == '__main__':
    main()
