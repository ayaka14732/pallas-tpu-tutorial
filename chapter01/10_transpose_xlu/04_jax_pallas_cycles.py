"""转置在 TC VMEM 边界内的周期数：一个与两个 128×128 的 f32、打包的 bf16、256×256 和 129×129；原生 XLA 与 Pallas kernel 各测一次。"""
import tpu_init
tpu_init.initialize_one_core()

import ml_dtypes
import numpy as np

import tpuasm_tools

def main() -> None:
    clock = tpuasm_tools.KernelClock(num_cores=1)
    rng = np.random.default_rng(0)
    f32 = lambda *shape: rng.standard_normal(shape, dtype=np.float32)
    cases = (
        ('x.T，f32[128,128]', lambda x: x.T, (f32(128, 128),)),
        ('两个 f32[128,128] 各自转置', lambda x, y: (x.T, y.T), (f32(128, 128), f32(128, 128))),
        ('x.T，bf16[128,128]', lambda x: x.T, (f32(128, 128).astype(ml_dtypes.bfloat16),)),
        ('x.T，f32[256,256]', lambda x: x.T, (f32(256, 256),)),
        ('x.T，f32[129,129]', lambda x: x.T, (f32(129, 129),)),
        ('x.T，f32[8,128]', lambda x: x.T, (f32(8, 128),)),
    )
    for name, f, arguments in cases:
        tpuasm_tools.compare_in_vmem(clock, name, f, [arguments] * 4)

if __name__ == '__main__':
    main()
