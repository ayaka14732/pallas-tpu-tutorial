"""原生 XLA 的 bf16[2048,2048] @ bf16[2048,2048] → f32[2048,2048]：编译后的 HLO 把输入放在哪里，清单中两个 TensorCore 各用了哪些指令，XProf 中设备上的时间。"""
import tpu_init
tpu_init.initialise_one_chip()

from collections import defaultdict
from pathlib import Path
import statistics

import jax
import jax.numpy as jnp
import numpy as np

import tpuasm_tools
import xprof_tools

SIZE = 2048

def inputs(seed: int) -> tuple[jax.Array, jax.Array]:
    """小整数输入：bf16 精确表示，f32 累加没有舍入，结果可以与 NumPy 逐元素比较。"""
    rng = np.random.default_rng(seed)
    lhs = rng.integers(-2, 3, (SIZE, SIZE)).astype(np.float32)
    rhs = rng.integers(-2, 3, (SIZE, SIZE)).astype(np.float32)
    return jnp.asarray(lhs, jnp.bfloat16), jnp.asarray(rhs, jnp.bfloat16)

def clock_times(clock: tpuasm_tools.KernelClock, compiled, pairs: list[tuple[jax.Array, jax.Array]]) -> list[tuple[str, list[int]]]:
    """整个程序与每条 HLO 指令在两个 TensorCore 上的周期数：设备上的 LCC 读数，每对输入运行一次，取中位数。"""
    inputs = iter(pairs)
    return clock.time_ops(compiled, lambda timed: timed(*next(inputs)).block_until_ready(), samples=len(pairs))

def xprof_times(compiled, pairs: list[tuple[jax.Array, jax.Array]]) -> list[tuple[str, list[int]]]:
    """同样的量改由 XProf 的设备事件给出（按 1.05 GHz 换算成周期），用于无法改写的程序。"""
    events = xprof_tools.device_events(xprof_tools.capture(lambda: [compiled(lhs, rhs).block_until_ready() for lhs, rhs in pairs], Path('/tmp/pallas_tpu_tutorial/xprof')))
    durations = defaultdict(lambda: defaultdict(list))
    for event in events:
        name = 'module' if event['track'] == 'XLA Modules' else event['name']
        durations[name][event['device']].append(xprof_tools.device_cycles(event))
    return [(name, [round(statistics.median(cores[device])) for device in sorted(cores)]) for name, cores in durations.items()]

def main() -> None:
    lhs, rhs = inputs(0)
    matmul = lambda lhs, rhs: jnp.dot(lhs, rhs, preferred_element_type=jnp.float32)
    compiled = tpuasm_tools.compile(matmul, lhs, rhs)
    np.testing.assert_array_equal(np.asarray(compiled(lhs, rhs)), np.asarray(lhs, np.float32) @ np.asarray(rhs, np.float32))
    print('数值检查通过')
    print('## 优化后的 HLO')
    text = compiled.as_text()
    entry = text[text.index('\nENTRY') + 1:].split('\n}')[0]
    for line in entry.splitlines()[1:]:
        print('  ' + line.strip().split(', metadata')[0].split(', backend_config')[0])
    listing = tpuasm_tools.kernel_listing(compiled)
    counts = tpuasm_tools.count_mnemonics(listing)
    print('## 清单（两个 TensorCore 执行同一份程序，按 core 编号分工）')
    print('  ' + '，'.join(f'{name} {counts[name]}' for name in sorted(counts) if name.split('.')[0] in ('vmatmul', 'vmatpush', 'vdwg', 'cld', 'vpop', 'dma', 'vld', 'vst')))
    print(tpuasm_tools.listing_outline(compiled))
    pairs = [inputs(seed) for seed in range(1, 17)]
    print('## 设备上的周期数（LCC），16 对新输入的中位数')
    for name, cycles in clock_times(tpuasm_tools.KernelClock(num_cores=2), compiled, pairs):
        print(f'  {name}：TensorCore 0 {cycles[0]} 个周期，TensorCore 1 {cycles[1]} 个周期')

if __name__ == '__main__':
    main()
