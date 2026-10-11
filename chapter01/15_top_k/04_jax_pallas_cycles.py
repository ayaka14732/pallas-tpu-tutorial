"""top-k 在 TC VMEM 边界内的周期数：原生 XLA、Pallas 的 lax.top_k 与三种手写写法；只改 k、只改宽度、只改行数。每个程序用 4 份输入各运行一次，输出全部核对。"""
import tpu_init
tpu_init.initialize_one_core()

from collections.abc import Callable

import jax
import numpy as np

from top_k_common import top_k_by_hand, top_k_by_rank, top_k_lifted
import tpuasm_tools

def check(k: int) -> Callable[[tuple[jax.Array, jax.Array], np.ndarray], None]:
    def verify(result: tuple[jax.Array, jax.Array], x: np.ndarray) -> None:
        order = np.argsort(-x, axis=1, kind='stable')[:, :k]
        np.testing.assert_array_equal(np.asarray(result[0]), np.take_along_axis(x, order, axis=1))
        np.testing.assert_array_equal(np.asarray(result[1]), order)

    return verify

def main() -> None:
    clock = tpuasm_tools.KernelClock(num_cores=1)
    rng = np.random.default_rng(0)
    # 每行是 0 到 n − 1 的随机排列：没有并列，下标只有一个正确答案。
    inputs = lambda rows, columns: [(np.stack([rng.permutation(columns) for _ in range(rows)]).astype(np.float32),) for _ in range(4)]
    official = lambda k: tpuasm_tools.vmem_kernel(lambda x: jax.lax.top_k(x, k, is_stable=False), 'top_k')
    summary = []
    for k in (1, 8, 16, 32):
        runs = inputs(8, 128)
        candidates = {
            'XLA，is_stable=False': lambda x: jax.lax.top_k(x, k, is_stable=False),
            'XLA，is_stable=True': lambda x: jax.lax.top_k(x, k, is_stable=True),
            'Pallas，lax.top_k': official(k),
            'Pallas，掩码排除已选位置': tpuasm_tools.vmem_kernel(top_k_by_hand(k), 'top_k'),
            'Pallas，抬高': tpuasm_tools.vmem_kernel(top_k_lifted(k), 'top_k'),
            'Pallas，秩计数': tpuasm_tools.vmem_kernel(top_k_by_rank(k), 'top_k'),
        }
        print(f'## f32[8,128]，k = {k}')
        for name, function in candidates.items():
            timing = tpuasm_tools.time_in_vmem(clock, function, runs, check(k))
            counts = tpuasm_tools.vector_counts(timing.listing)
            xlu = sum(count for mnemonic, count in counts.items() if '.xlane' in mnemonic or mnemonic.startswith(('vrot.0', 'vrot.1', 'vrot.2', 'vrot.3', 'vperm')))
            print(f'  {name}：{timing.summary()} 个周期；区间 {timing.bundles} 个 bundle，XLU 提交 {xlu} 次；HLO 指令：{"、".join(timing.inside)}；DMA：{"、".join(sorted(set(timing.moves))) or "无"}')
            summary.append((name, k, timing.summary()))
    print('## 汇总：f32[8,128]，k = 1、8、16、32')
    for name in candidates:
        print(f'  {name}：' + '、'.join(cycles for candidate, _, cycles in summary if candidate == name))

    print('## 只改形状，k = 8')
    for rows, columns in ((8, 256), (8, 1024), (64, 128), (128, 128)):
        runs = inputs(rows, columns)
        for name, function in (('XLA，is_stable=False', lambda x: jax.lax.top_k(x, 8, is_stable=False)), ('Pallas，lax.top_k', official(8))):
            try:
                timing = tpuasm_tools.time_in_vmem(clock, function, runs, check(8))
            except jax.errors.JaxRuntimeError as error:
                print(f'  f32[{rows},{columns}]，{name}：编译失败：{str(error).split(", metadata=")[0].splitlines()[0][:160]}')
                continue
            print(f'  f32[{rows},{columns}]，{name}：{timing.summary()} 个周期；区间 {timing.bundles} 个 bundle；DMA：{"、".join(sorted(set(timing.moves))) or "无"}')

if __name__ == '__main__':
    main()
