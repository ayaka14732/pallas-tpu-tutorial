"""原生 XLA 的 all-reduce：shard_map 中的 jax.lax.psum，与 Pallas 版本相同的三种大小；统计 XLA 生成的 remote DMA，并用 LCC 给 psum 计时。"""
import tpu_init
tpu_init.initialize_local_chips()

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

import tpuasm_tools

def main() -> None:
    devices = {device.id: device for device in jax.devices()}
    mesh = jax.sharding.Mesh(np.array([devices[i] for i in (0, 1, 3, 2)]), ('device',))

    @jax.shard_map(
        mesh=mesh,
        in_specs=P('device'),
        out_specs=P('device'),
        check_vma=False,
    )
    def all_reduce(x: jax.Array) -> jax.Array:
        return jax.lax.psum(x, 'device')

    clock = tpuasm_tools.KernelClock(num_cores=1)
    for rows in (32, 256, 2048):
        host = np.random.default_rng(rows).integers(-100, 100, (4, rows, 128)).astype(np.float32)
        x = jax.device_put(jnp.asarray(host), jax.NamedSharding(mesh, P('device')))
        compiled = tpuasm_tools.compile(all_reduce, x, mesh=mesh)
        np.testing.assert_array_equal(np.asarray(compiled(x)), np.broadcast_to(host.sum(axis=0), host.shape))
        listing = tpuasm_tools.kernel_listing(compiled)
        counts = tpuasm_tools.count_mnemonics(listing)
        entries = sorted({line[len('# entry bundle: '):].split(' = ')[0] for line in listing.splitlines() if line.startswith('# entry bundle')})
        # 与 Pallas 版本一样取差值：fori_loop 循环 64 次与 32 次的整个程序在 device 0 的 TensorCore 0 上的周期数（LCC），相减除以 32；循环中每次先除以 4 再求和，数值保持不变。
        modules = {}
        for repeats in (32, 64):
            program = tpuasm_tools.compile(lambda x, n=repeats: jax.lax.fori_loop(0, n, lambda _, y: all_reduce(y * 0.25), x), x, mesh=mesh)
            modules[repeats] = dict(clock.time_ops(program, lambda timed: jax.block_until_ready(timed(x))))['module'][0]
        cycles = (modules[64] - modules[32]) / 32
        print(f'## 每颗芯片 f32[{rows},128]（{rows * 512 // 1024} KiB）：数值检查通过；每次 {cycles:.0f} 个周期')
        print(f'  HLO 段：{"、".join(entries)}；dma.general {counts["dma.general"]} 条，vsyncadd.remote {counts["vsyncadd.remote.s32"]} 条，cld {counts["cld.8x128"]} 条')

if __name__ == '__main__':
    main()
