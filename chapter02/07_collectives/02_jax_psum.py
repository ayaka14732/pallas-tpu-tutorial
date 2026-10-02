"""原生 XLA 的 all-reduce：shard_map 中的 jax.lax.psum，与 Pallas 版本相同的三种大小；统计 XLA 生成的 remote DMA，并用循环计时。"""
import tpu_init
tpu_init.initialise_local_chips()

import statistics
import time

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

    for rows in (32, 256, 2048):
        host = np.random.default_rng(rows).integers(-100, 100, (4, rows, 128)).astype(np.float32)
        x = jax.device_put(jnp.asarray(host), jax.NamedSharding(mesh, P('device')))
        compiled = tpuasm_tools.compile(all_reduce, x, mesh=mesh)
        np.testing.assert_array_equal(np.asarray(compiled(x)), np.broadcast_to(host.sum(axis=0), host.shape))
        listing = tpuasm_tools.kernel_listing(compiled)
        counts = tpuasm_tools.count_mnemonics(listing)
        entries = sorted({line[len('# entry bundle: '):].split(' = ')[0] for line in listing.splitlines() if line.startswith('# entry bundle')})
        # 循环中每次先除以 4 再求和，数值保持不变；计时取循环 64 次与 32 次之差除以 32。
        times = {}
        for repeats in (32, 64):
            program = jax.jit(lambda x, n=repeats: jax.lax.fori_loop(0, n, lambda _, y: all_reduce(y * 0.25), x))
            jax.block_until_ready(program(x))
            samples = []
            for _ in range(20):
                start = time.perf_counter()
                jax.block_until_ready(program(x))
                samples.append(time.perf_counter() - start)
            times[repeats] = statistics.median(samples)
        per_call = (times[64] - times[32]) / 32 * 1e6
        print(f'## 每颗芯片 f32[{rows},128]（{rows * 512 // 1024} KiB）：数值检查通过；每次约 {per_call:.1f} µs')
        print(f'  HLO 段：{"、".join(entries)}；dma.general {counts["dma.general"]} 条，vsyncadd.remote {counts["vsyncadd.remote.s32"]} 条，cld {counts["cld.8x128"]} 条')

if __name__ == '__main__':
    main()
