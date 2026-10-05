"""vpush 把 TC VREG 的哪个元素交给标量单元：用 tpuasm 对一个随机的 TC VREG 执行 vpush、spop，再把取出的标量广播写回，与输入逐元素比较。"""
import tpu_init
tpu_init.initialize_one_chip()

import numpy as np

import tpuasm_tools
from tpuasm_tools import bundle

def main() -> None:
    probe = tpuasm_tools.LccProbe()
    # v10 是载体读入的随机 u32 tile；标量经 vmov 广播后写进第 1 个 tile。
    body = bundle('vst: vpush v2sf, v10') + bundle('s0: spop s24, v2sf') + bundle('va0: vmov.8x128 v11, s24') + bundle('misc: vnop') * 8 + bundle('vst: vst.8x128 [vmem:0x8], v11')
    tile = probe.run_tiles(body)[0, 0, 0]
    assert np.all(tile == tile[0, 0])
    positions = np.argwhere(probe.host[:8] == tile[0, 0])
    print(f'spop 取出的值 0x{int(tile[0, 0]):08x}；输入中等于它的位置（sublane, lane）：{positions.tolist()}')

if __name__ == '__main__':
    main()
