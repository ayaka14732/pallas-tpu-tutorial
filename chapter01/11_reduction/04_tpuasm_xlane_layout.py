"""vadd.xlane 与 vmax.xlane 的结果放在哪里：用 tpuasm 直接执行一条跨 lane 归约，把取回的整个 TC VREG 写回主机检查。"""
import tpu_init
tpu_init.initialise_one_chip()

import jax
import numpy as np

import tpuasm_tools
from tpuasm_tools import bundle

GAP = bundle('misc: vnop') * 16

def main() -> None:
    probe = tpuasm_tools.LccProbe()
    # 第 0 个 tile 是 f32[8,128] 的输入：x[s,l] = (128s + l) mod 7，各行的和与最大值互不相同。
    x = (np.arange(8 * 128).reshape(8, 128) % 7).astype(np.float32)
    x[:, 37] += np.arange(8) * 10
    host = np.zeros_like(probe.host)
    host[:8] = x.view(np.uint32)
    probe.x = jax.device_put(host, jax.local_devices()[0])
    for name, instruction, expected in (('vadd.xlane', 'vadd.xlane.0.8x128.f32', x.sum(axis=1)), ('vmax.xlane', 'vmax.xlane.0.8x128.f32', x.max(axis=1))):
        body = bundle(f'vx0: {instruction} trf0, v10') + GAP + bundle('vr0: vpop.8x128 v11, trf0') + GAP + bundle('vst: vst.8x128 [vmem:0x8], v11') + GAP
        result = probe.run_tiles(body)[0, 0, 0].view(np.float32)
        print(f'## {name}')
        print(f'  每行的期望值：{expected.tolist()}')
        print(f'  取回的 TC VREG 第 0、1、127 列：{result[:, 0].tolist()}、{result[:, 1].tolist()}、{result[:, 127].tolist()}')
        print(f'  每行 128 个 lane 都等于该行的结果：{bool(np.all(result == expected[:, None]))}')

if __name__ == '__main__':
    main()
