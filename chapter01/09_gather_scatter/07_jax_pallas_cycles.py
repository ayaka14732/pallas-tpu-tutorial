"""TC VREG 内的 gather 与循环移位在 TC VMEM 边界内的周期数：lane gather、用循环移位合成的 sublane gather、两个方向的固定位移；原生 XLA 与 Pallas kernel 各测一次。"""
import tpu_init
tpu_init.initialize_one_core()

from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

from gather_common import sublane_gather_by_rotation
import tpuasm_tools

def main() -> None:
    clock = tpuasm_tools.KernelClock(num_cores=1)
    rng = np.random.default_rng(0)
    cases = []
    # 每项是（名称，原生 XLA 的写法，Pallas kernel 中的写法，输入）；Pallas 的写法为 None 时与 XLA 相同。
    for rows in (8, 64):
        x = rng.standard_normal((rows, 128), dtype=np.float32)
        lane = rng.integers(0, 128, (rows, 128)).astype(np.int32)
        cases.append((f'lane gather：take_along_axis(x, i, axis=1)，f32[{rows},128]', lambda x, i: jnp.take_along_axis(x, i, axis=1), None, (x, lane)))
        cases.append((f'lane 循环移位 5 位，f32[{rows},128]', lambda x: jnp.roll(x, 5, axis=1), lambda x: pltpu.roll(x, 5, axis=1), (x,)))
    x = rng.standard_normal((8, 128), dtype=np.float32)
    sublane = rng.integers(0, 8, (8, 128)).astype(np.int32)
    cases.append(('sublane gather：take_along_axis(x, i, axis=0)，f32[8,128]（Pallas 用循环移位合成）', lambda x, i: jnp.take_along_axis(x, i, axis=0), sublane_gather_by_rotation, (x, sublane)))
    cases.append(('sublane 循环移位 3 位，f32[8,128]', lambda x: jnp.roll(x, 3, axis=0), lambda x: pltpu.roll(x, 3, axis=0), (x,)))
    for name, f, body, arguments in cases:
        tpuasm_tools.compare_in_vmem(clock, name, f, [arguments] * 4, pallas=tpuasm_tools.vmem_kernel(body) if body else None)

if __name__ == '__main__':
    main()
