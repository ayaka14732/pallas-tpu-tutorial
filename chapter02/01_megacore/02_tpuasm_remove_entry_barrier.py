"""两个 TensorCore 各算一半、互不读取对方数据时，kernel 入口的跨核汇合是多余的：用 tpuasm 把这三条同步指令换成 vnop，检查数值仍然正确。"""
import tpu_init
tpu_init.initialise_one_chip()

import importlib

import jax.numpy as jnp
import numpy as np

import tpuasm_tools

two_cores = importlib.import_module('01_pallas_two_cores')
# 入口汇合的三条指令；kernel 出口还有同样的三条，所以只改 kernel 中第一次出现的那一组。
ENTRY_BARRIER = (
    'vsyncadd.remote.s32',
    'vwait.ge [sflag:45], 1',
    'vsyncadd.s32 [sflag:45], -1',
)

def main() -> None:
    x = jnp.arange(64 * 128, dtype=jnp.float32).reshape(64, 128)
    mesh, scale = two_cores.build(2)
    compiled = tpuasm_tools.compile(scale, x, mesh=mesh)
    serialized = tpuasm_tools.serialize(compiled)
    edits = {}
    for instruction in ENTRY_BARRIER:
        pc = tpuasm_tools.find_bundles(serialized, instruction)[0]
        # 只把这条指令换成 vnop，同一 bundle 中其他槽的指令保持不变。
        text = next(line for line in tpuasm_tools.bundle_text(serialized, pc).splitlines() if instruction in line)
        edits[pc] = (text.strip(' {};'), 'misc: vnop')
        print(f'bundle {pc}：{edits[pc][0]} → vnop')
    patched = tpuasm_tools.edit_bundles(serialized, edits)
    function = tpuasm_tools.load(patched, compiled)
    for trial in range(100):
        np.testing.assert_array_equal(np.asarray(function(x + trial)), (np.asarray(x) + trial) * 2.0)
    listing = tpuasm_tools.kernel_listing(patched, pallas_only=True)
    counts = tpuasm_tools.count_mnemonics(listing)
    print(f'去掉入口汇合后连续运行 100 次，数值检查全部通过；剩余 vsyncadd.remote {counts["vsyncadd.remote.s32"]} 条（kernel 出口）')
    print('\n'.join(line.split('#')[0].rstrip() for line in listing.splitlines() if 'vnop' in line or 'sflag:45' in line or 'remote' in line))

if __name__ == '__main__':
    main()
