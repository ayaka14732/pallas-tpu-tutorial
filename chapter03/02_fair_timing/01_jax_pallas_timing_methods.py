"""同一个比较用不同的计时方法：f32[32768,128] 的 y = 2x + 1，原生 XLA 对照第二章第 2 节的 Pallas 双缓冲 kernel。分别用固定输入、每次新输入、设备上的 LCC、fori_loop 循环和 AB/BA 轮次计时，并检查编译后的 HLO 中输入放在哪一层内存。"""
import tpu_init
tpu_init.initialize_one_chip()

import importlib.util
from pathlib import Path
import re
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

import tpuasm_tools

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('pipeline', ROOT / 'chapter02/02_dma_pipeline/01_pallas_double_buffer.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)
SAMPLES = 50

def synchronized(function, inputs: list[jax.Array]) -> float:
    """依次用 inputs 中的数组调用，每次等到结果，返回中位数（秒）。"""
    samples = []
    for x in inputs:
        start = time.perf_counter()
        function(x).block_until_ready()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)

def memory_notes(compiled) -> str:
    """编译后的 HLO 中与计时边界有关的三件事：跨程序预取、放进 Megacore Shared CMEM 的数组、ROOT 是否是一次 copy。"""
    text = compiled.as_text()
    root = re.search(r'ROOT \S+ = \S+ ([\w-]+)\(', text[text.index('\nENTRY'):]).group(1)
    return f'cross_program_prefetch {"有" if "cross_program_prefetch" in text else "无"}，S(3) {text.count("S(3)")} 处，ROOT 是 {root}'

def main() -> None:
    host = np.arange(pipeline.ROWS * 128, dtype=np.float32).reshape(pipeline.ROWS, 128) / 1024
    x = jnp.asarray(host)
    mesh, pallas_hbm = pipeline.build('输入输出双缓冲', 1, 2048)
    _, pallas_any = pipeline.build('输入输出双缓冲', 1, 2048, out_hbm=False)
    functions = {
        'XLA': lambda x: x * 2.0 + 1.0,
        'Pallas（out_type=pltpu.HBM）': pallas_hbm,
        'Pallas（out_type=ShapeDtypeStruct）': pallas_any,
    }
    # 原生 XLA 不经过 shard_map，编译时不需要 mesh。
    meshes = {name: None if name == 'XLA' else mesh for name in functions}
    candidates = {name: tpuasm_tools.compile(function, x, mesh=meshes[name]) for name, function in functions.items()}
    for name, compiled in candidates.items():
        np.testing.assert_array_equal(np.asarray(compiled(x)), host * 2.0 + 1.0)
        print(f'{name}：数值检查通过；编译后的 HLO 中 {memory_notes(compiled)}')
    fresh = [jnp.asarray(host + i) for i in range(SAMPLES)]
    print('## 方法 1：每次调用并等到结果，固定同一个输入')
    for name, compiled in candidates.items():
        compiled(x).block_until_ready()
        print(f'  {name}：{synchronized(compiled, [x] * SAMPLES) * 1e6:.1f} µs')
    print('## 方法 2：每次调用并等到结果，每次一个新的输入数组')
    for name, compiled in candidates.items():
        print(f'  {name}：{synchronized(compiled, fresh) * 1e6:.1f} µs')
    print('## 方法 3：设备上的 LCC（第 3 节的 KernelClock），每次一个新的输入数组')
    clock = tpuasm_tools.KernelClock(num_cores=2)
    for name, compiled in candidates.items():
        inputs = iter(fresh)
        for op, cycles in clock.time_ops(compiled, lambda timed: timed(next(inputs)).block_until_ready(), samples=16):
            print(f'  {name}，{op}：TensorCore 0 {cycles[0]} 个周期，TensorCore 1 {cycles[1]} 个周期')
    print('## 方法 4：在一个 jit 中用 fori_loop 重复 N 次，(T(32) − T(16)) / 16')
    for name, function in functions.items():
        times = {}
        for count in (16, 32):
            looped = tpuasm_tools.compile(lambda x: jax.lax.fori_loop(0, count, lambda i, y: function(y), x), x, mesh=meshes[name])
            looped(x).block_until_ready()
            times[count] = synchronized(looped, [x] * 10)
            if count == 32:
                print(f'  {name}：每次 {(times[32] - times[16]) / 16 * 1e6:.1f} µs；循环版本的 HLO 中 {memory_notes(looped)}')
    print('## 方法 5：方法 2 按 AB/BA 交替做 4 轮')
    for round_index in range(4):
        names = list(candidates)
        order = names if round_index % 2 == 0 else names[::-1]
        results = {name: synchronized(candidates[name], fresh) for name in order}
        print(f'  第 {round_index + 1} 轮（{"、".join(order)}）：' + '，'.join(f'{name} {results[name] * 1e6:.1f} µs' for name in names))

if __name__ == '__main__':
    main()
