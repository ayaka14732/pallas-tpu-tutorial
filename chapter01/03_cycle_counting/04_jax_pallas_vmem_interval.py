"""把计时边界放在 TC VMEM：输入和结果都固定在 TC VMEM，只测中间的计算。原生 XLA 与 Pallas kernel 用同一个外壳，对 x * 2 测三种大小，再把终点逐个 bundle 向后移动，看区间内每个 bundle 发射的时刻。"""
import tpu_init
tpu_init.initialize_one_core()

import jax
import numpy as np

import tpuasm_tools

def scale(x: jax.Array) -> jax.Array:
    return x * 2.0

def check(result: jax.Array, x: np.ndarray) -> None:
    np.testing.assert_array_equal(np.asarray(result), x * 2.0)

def main() -> None:
    clock = tpuasm_tools.KernelClock(num_cores=1)
    rng = np.random.default_rng(0)
    candidates = {'XLA': scale, 'Pallas': tpuasm_tools.vmem_kernel(scale, 'scale')}
    for rows in (8, 64, 1024):
        print(f'## x * 2，f32[{rows},128]')
        runs = [(rng.standard_normal((rows, 128), dtype=np.float32),) for _ in range(8)]
        for name, function in candidates.items():
            timing = tpuasm_tools.time_in_vmem(clock, function, runs, check)
            counts = tpuasm_tools.vector_counts(timing.listing)
            print(f'  {name}：{"；".join(timing.layouts)}')
            print(f'    区间是 bundle [{timing.first}, {timing.last})，共 {timing.bundles} 个；其中的 HLO 指令：{"、".join(timing.inside)}；DMA：{"、".join(timing.moves) or "无"}')
            print(f'    区间内的向量指令：{"，".join(f"{mnemonic} {counts[mnemonic]}" for mnemonic in sorted(counts))}')
            print(f'    8 份输入的周期数：{timing.cycles}')

    print('## Pallas，f32[8,128]：起点不动，终点逐个 bundle 向后移动')
    x = rng.standard_normal((8, 128), dtype=np.float32)
    mesh = jax.make_mesh((1,), ('device',))
    compiled = tpuasm_tools.compile(tpuasm_tools.vmem_shell(candidates['Pallas'], mesh), x, mesh=mesh)
    serialized = tpuasm_tools.serialize(compiled)
    first, last, *_ = tpuasm_tools.vmem_interval(compiled)
    names = {start: name for name, start, _ in tpuasm_tools.hlo_ops(compiled)}
    for end in range(first + 1, last + 1):
        cycles, = clock.interval(compiled, first, end, lambda timed: check(timed(x), x))
        text = ' '.join(tpuasm_tools.bundle_text(serialized, end - 1).split())
        print(f'  到 bundle {end - 1} 为止：{cycles:3d}    {text}' + (f'    # {names[end - 1]} 开始' if end - 1 in names else ''))

if __name__ == '__main__':
    main()
