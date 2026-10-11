"""经过 TC VMEM 的按索引读写：vsetiar 设置索引寄存器，vld.iar、vst.iar 按“地址 + 子通道号 + 索引”逐元素选行。用 tpuasm 直接执行，确定三种 vsetiar 的含义，再用它们实现 sublane gather 与 sublane scatter，并计时。"""
import tpu_init
tpu_init.initialize_one_chip()

import jax
import numpy as np

import tpuasm_tools
from tpuasm_tools import bundle, read_lcc

GAP = bundle('misc: vnop') * 12
END = read_lcc(21) + bundle('s0: sfence') + read_lcc(22)
# v1 = 128 × 子通道号 + 通道号，v3 = 子通道号；v11 = 第 1 个 tile（索引）。
SETUP = bundle('va0: vlaneseq.8x128.u32 v1') + bundle('va1: vshrl.8x128.s32 v3, v1, 0x7') + bundle('vld: vld.8x128 v11, [vmem:0x8]') + GAP
# 表：从 0x80 起 16 个 tile，第 r 行第 l 列的值是 128r + l（r 从表头算起）。
TABLE = ''.join(bundle(f'va0: vadd.8x128.s32 v4, {1024 * tile}, v1') + bundle(f'vst: vst.8x128 [vmem:0x{0x80 + 8 * tile:x}], v4') for tile in range(16)) + GAP
# v13 = (索引 − 子通道号) mod 8：从第 s 个子通道出发，再走这么多行就到第 i 行（或第 i + 8 行）。
OFFSET = bundle('va0: vsub.8x128.s32 v12, v11, v3') + bundle('va0: vand.8x128.u32 v13, 0x7, v12')

def main() -> None:
    probe = tpuasm_tools.LccProbe()
    rng = np.random.default_rng(0)
    x = rng.integers(0, 1 << 20, (8, 128)).astype(np.uint32)
    s, l = np.arange(8)[:, None], np.arange(128)[None, :]

    def run(index: np.ndarray, body: str, tiles: int = 1) -> np.ndarray:
        """第 0 个 tile 放 x，第 1 个 tile 放 index；返回片段写进第 1 个 tile 起的 tiles 个 tile。"""
        host = probe.host.copy()
        host[:8], host[8:16] = x, index
        probe.x = jax.device_put(host, jax.local_devices()[0])
        return probe.run_tiles(SETUP + body)[0, 0, :tiles].astype(np.int64)

    print('## vld.iar0 读的是表的哪一行（表从 0x80 起，第 r 行第 l 列的值是 128r + l）')
    index = rng.integers(0, 8, (8, 128)).astype(np.uint32)
    for mode, rule, expected in (('raw', 's + i[s,l]', s + index), ('lane', 's + i[0,l]', s + index[:1]), ('sublane', 's + 16 × i[s,l]', s + 16 * index)):
        out = run(index, TABLE + bundle(f'vst: vsetiar.{mode} iar0, v11') + GAP + bundle('vld: vld.iar0.8x128 v14, [vmem:0x80]') + GAP + bundle('vst: vst.8x128 [vmem:0x8], v14') + GAP)[0]
        print(f'  vsetiar.{mode}：通道号不变：{bool(np.all(out % 128 == l))}；行号等于 {rule}：{bool(np.array_equal(out // 128, np.broadcast_to(expected, out.shape)))}')

    print('## sublane gather：y[s,l] = x[i[s,l], l]')
    gather = bundle('vst: vst.8x128 [vmem:0x80], v10') + bundle('vst: vst.8x128 [vmem:0x88], v10') + OFFSET + bundle('vst: vsetiar.raw iar0, v13') + bundle('vld: vld.iar0.8x128 v14, [vmem:0x80]')
    out = run(index, gather + GAP + bundle('vst: vst.8x128 [vmem:0x8], v14') + GAP)[0]
    print(f'  与 np.take_along_axis(x, i, axis=0) 相同：{bool(np.array_equal(out, np.take_along_axis(x, index.astype(np.int64), axis=0)))}')
    (_, r2), = {tuple(row) for row in probe.run(read_lcc(20) + gather + bundle('va0: vadd.8x128.s32 v15, 1, v14') + END, setup=SETUP).tolist()}
    print(f'  7 条指令（2 条 vst、vsub、vand、vsetiar、vld.iar，加一条使用结果的 vadd）：R2 − R0 = {r2}')

    print('## sublane scatter：y[i[s,l], l] = x[s,l]，每列的 i 是 0–7 的一个排列')
    permutation = np.stack([rng.permutation(8) for _ in range(128)], axis=1).astype(np.uint32)
    # 先把目标的两个 tile 填成全 1；写入的行是 s + (i − s) mod 8，即第 i 行或第 i + 8 行，最后把两个 tile 合起来。
    scatter = bundle('va0: vimm.8x128.s32 v7, -1') + bundle('vst: vst.8x128 [vmem:0x80], v7') + bundle('vst: vst.8x128 [vmem:0x88], v7') + OFFSET + bundle('vst: vsetiar.raw iar0, v13') + bundle('vst: vst.iar.8x128 [vmem:0x80], iar0, v10') + bundle('vld: vld.8x128 v14, [vmem:0x80]') + bundle('vld: vld.8x128 v15, [vmem:0x88]') + bundle('va0: veq.8x128.s32 vm0, v14, -1') + bundle('va0: vsel.8x128 v16, vm0, v15, v14')
    out = run(permutation, scatter + GAP + bundle('vst: vst.8x128 [vmem:0x8], v16') + GAP)[0]
    expected = np.zeros_like(x)
    np.put_along_axis(expected, permutation.astype(np.int64), x, axis=0)
    print(f'  与 np.put_along_axis(y, i, x, axis=0) 相同：{bool(np.array_equal(out, expected))}')

    print('## 计时：R2 − R0（其中 13 个周期是读数与排空的固定开销）')
    load = lambda instruction: ''.join(bundle(f'vld: {instruction} v{14 + n % 4}, [vmem:0x{0x80 + 8 * (n % 8):x}]') for n in range(16))
    store = lambda instruction, operands: ''.join(bundle(f'vst: {instruction} [vmem:0x{0x80 + 8 * (n % 8):x}], {operands}') for n in range(16))
    sequences = (
        ('vld，紧接着使用', bundle('vld: vld.8x128 v14, [vmem:0x80]') + bundle('va0: vadd.8x128.s32 v15, 1, v14')),
        ('vld.iar0，紧接着使用', bundle('vld: vld.iar0.8x128 v14, [vmem:0x80]') + bundle('va0: vadd.8x128.s32 v15, 1, v14')),
        ('vsetiar.raw、vld.iar0，紧接着使用', bundle('vst: vsetiar.raw iar0, v11') + bundle('vld: vld.iar0.8x128 v14, [vmem:0x80]') + bundle('va0: vadd.8x128.s32 v15, 1, v14')),
        ('vst 之后从同一个 tile vld.iar0，紧接着使用', bundle('vst: vst.8x128 [vmem:0x80], v10') + bundle('vld: vld.iar0.8x128 v14, [vmem:0x80]') + bundle('va0: vadd.8x128.s32 v15, 1, v14')),
        ('16 条 vld', load('vld.8x128')),
        ('16 条 vld.iar0', load('vld.iar0.8x128')),
        ('16 条 vst', store('vst.8x128', 'v10')),
        ('16 条 vst.iar', store('vst.iar.8x128', 'iar0, v10')),
    )
    # 计时用按通道的索引（vsetiar.lane）：同一列的 8 个子通道偏移相同，vst.iar 不会有两个元素写到同一个位置。
    for name, sequence in sequences:
        (_, r2), = {tuple(row) for row in probe.run(read_lcc(20) + sequence + END, setup=SETUP + bundle('vst: vsetiar.lane iar0, v11') + GAP).tolist()}
        print(f'  {name}：{r2}')

if __name__ == '__main__':
    main()
