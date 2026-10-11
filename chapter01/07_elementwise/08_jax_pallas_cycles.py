"""逐元素运算在 TC VMEM 边界内的周期数：每种运算测 1 个和 128 个 TC VREG，原生 XLA 与 Pallas kernel 各测一次。两种大小之差除以 127，是连续处理时每个 TC VREG 的周期数。"""
import tpu_init
tpu_init.initialize_one_core()

import jax
import jax.numpy as jnp
import numpy as np

import tpuasm_tools

# 每项是（名称，函数，它用到的输入）：x 取自 [-4, 4)，y 取自 [0.25, 4)；整数运算的 a、b 是随机的 int32。
OPS = (
    ('x + y', lambda x, y: x + y, 'xy'),
    ('x * y', lambda x, y: x * y, 'xy'),
    ('x * y + 1', lambda x, y: x * y + 1, 'xy'),
    ('maximum(x, y)', lambda x, y: jnp.maximum(x, y), 'xy'),
    ('where(x > y, x, y)', lambda x, y: jnp.where(x > y, x, y), 'xy'),
    ('x * 2', lambda x: x * 2, 'x'),
    ('exp(x)', jnp.exp, 'x'),
    ('log(y)', jnp.log, 'y'),
    ('tanh(x)', jnp.tanh, 'x'),
    ('1 / y', lambda y: 1 / y, 'y'),
    ('x / y', lambda x, y: x / y, 'xy'),
    ('sqrt(y)', jnp.sqrt, 'y'),
    ('sigmoid(x)', jax.nn.sigmoid, 'x'),
    ('sin(x)', jnp.sin, 'x'),
    ('int32 a + b', lambda a, b: a + b, 'ab'),
    ('int32 a * b', lambda a, b: a * b, 'ab'),
    ('int32 a // b', lambda a, b: a // (b | 1), 'ab'),
)

def main() -> None:
    clock = tpuasm_tools.KernelClock(num_cores=1)
    rng = np.random.default_rng(0)
    summary = {}
    for rows in (8, 1024):
        inputs = {
            'x': rng.uniform(-4, 4, (rows, 128)).astype(np.float32),
            'y': rng.uniform(0.25, 4, (rows, 128)).astype(np.float32),
            'a': rng.integers(-2**31, 2**31, (rows, 128), dtype=np.int64).astype(np.int32),
            'b': rng.integers(-2**31, 2**31, (rows, 128), dtype=np.int64).astype(np.int32),
        }
        for name, f, used in OPS:
            timings = tpuasm_tools.compare_in_vmem(clock, f'{name}，[{rows},128]', f, [tuple(inputs[key] for key in used)] * 2)
            summary[name, rows] = {kind: timing.cycles[0] for kind, timing in timings.items()}
    print('## 汇总：运算；1 个 TC VREG 的周期数（XLA / Pallas）；128 个 TC VREG 的周期数（XLA / Pallas）；Pallas 每多一个 TC VREG 多几个周期')
    for name, _, _ in OPS:
        small, large = summary[name, 8], summary[name, 1024]
        print(f'  {name}：{small["XLA"]} / {small["Pallas"]}；{large["XLA"]} / {large["Pallas"]}；{(large["Pallas"] - small["Pallas"]) / 127:.2f}')

if __name__ == '__main__':
    main()
