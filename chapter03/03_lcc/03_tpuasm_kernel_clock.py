"""给整个 kernel 计时：SMEM 的内容在程序之间保留，所以可以在任意程序的指定位置插入 LCC 读数、存进 SMEM，程序运行之后再由另一个程序取回。用第 1 节那个设备时间已知的 kernel 验证。"""
import tpu_init
tpu_init.initialize_one_chip()

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import tpuasm_tools
from tpuasm_tools import bundle

spec = importlib.util.spec_from_file_location('five_times', Path(__file__).resolve().parents[1] / '01_five_times' / '01_pallas_host_and_device_time.py')
five_times = importlib.util.module_from_spec(spec)
spec.loader.exec_module(five_times)

def main() -> None:
    clock = tpuasm_tools.KernelClock(num_cores=2)

    print('## SMEM 的内容在程序之间保留')
    probe = clock.probe
    address = tpuasm_tools.CLOCK_BASE + 0x80
    read = bundle(f's1: sld s24, [smem:0x{address:x}]') + bundle('va0: vmov.8x128 v12, s24') + bundle('misc: vnop') * 8 + bundle('vst: vst.8x128 [vmem:0x8], v12')
    probe.run_tiles(bundle('s0: simm.s32 s24, 0') + bundle(f's1: sst [smem:0x{address:x}], s24'))
    print(f'  程序 A 写入 0 之后，程序 B 读到：{probe.run_tiles(read)[0, :, 0, 0, 0].tolist()}（两个 TensorCore）')
    probe.run_tiles(bundle('s0: simm.s32 s24, 20261004') + bundle(f's1: sst [smem:0x{address:x}], s24'))
    print(f'  程序 A 写入 20261004 之后，程序 B 读到：{probe.run_tiles(read)[0, :, 0, 0, 0].tolist()}')

    x = jnp.zeros((8, 128), jnp.float32)
    for delay_ns in five_times.DELAYS_NS:
        mesh, wait = five_times.build(delay_ns)
        compiled = tpuasm_tools.compile(wait, x, mesh=mesh)
        ops = tpuasm_tools.hlo_ops(compiled)
        (_, module_start, module_end), (name, start, end) = ops
        print(f'## pl.delay({delay_ns})：vtrace 标记所在的 bundle：{ops}')
        # 六次读数：程序开始、kernel 开始（连读两次）、kernel 结束（连读两次）、程序结束。结束标记之后的位置是它的下一个 bundle。
        timed = clock.instrument(compiled, [module_start, start, start, end + 1, end + 1, module_end + 1])
        for run in range(3):
            np.testing.assert_array_equal(np.asarray(timed(x)), 1.0)
            readings = clock.read(6).astype(np.int64)
            for core, row in enumerate(readings):
                deltas = (row[1:] - row[:-1]).tolist()
                print(f'  第 {run + 1} 次运行，TensorCore {core}：程序开始 → kernel 开始 {deltas[0]}，相邻两次读数 {deltas[1]}，kernel {deltas[2]}，相邻两次读数 {deltas[3]}，kernel 结束 → 程序结束 {deltas[4]}')

if __name__ == '__main__':
    main()
