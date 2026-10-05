"""在 kernel 中用 jax.named_scope 标出四个区域：发起输入 DMA、等待输入 DMA、计算、输出 DMA（发起并等待）。比较打开与关闭 region trace 时的清单，再从 XProf 读出各区域的时间。"""
import tpu_init
tpu_init.initialize_one_chip()

from collections import defaultdict
from pathlib import Path
import re
import statistics

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

import tpuasm_tools
import xprof_tools

def build(nested: bool = False):
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def affine(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.SemaphoreType.DMA),
            name='affine',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, sem: Ref) -> None:
            load = pltpu.make_async_copy(x_hbm, x_vmem, sem)
            with jax.named_scope('load_start'):
                load.start()
            with jax.named_scope('load_wait'):
                load.wait()
            if nested:
                # 外层区域包住两个内层区域。
                with jax.named_scope('compute_and_store'):
                    with jax.named_scope('compute'):
                        x_vmem[...] = x_vmem[...] * 2.0 + 1.0
                    with jax.named_scope('store'):
                        pltpu.async_copy(x_vmem, o_hbm, sem).wait()
            else:
                with jax.named_scope('compute'):
                    x_vmem[...] = x_vmem[...] * 2.0 + 1.0
                with jax.named_scope('store'):
                    pltpu.async_copy(x_vmem, o_hbm, sem).wait()

        return kernel(x)

    return mesh, affine

def joined_bundles(listing: str) -> list[str]:
    """清单中的 bundle，每个拼成一行，去掉源码注释。"""
    bundles, current = [], []
    for line in listing.splitlines():
        text = line.split('#')[0].strip()
        if text.startswith('{'):
            current = [text]
        elif current:
            current.append(text)
        if current and text.endswith('}'):
            bundles.append(re.sub(r'\s*;\s*\.encoding \{[^}]*\}', '', ' '.join(current)))
            current = []
    return bundles

def device_medians(compiled, x: jax.Array) -> dict[tuple[str, str, str], int]:
    """每个事件 16 次调用的中位数：GTC 高 60 位之差。"""
    events = xprof_tools.device_events(xprof_tools.capture(lambda: [compiled(x).block_until_ready() for _ in range(16)], Path('/tmp/pallas_tpu_tutorial/xprof')))
    durations = defaultdict(list)
    for event in events:
        name = 'module' if event['track'] == 'XLA Modules' else event['name']
        durations[(event['device'], event['track'], name)].append(xprof_tools.gtc_ticks(event))
    return {key: round(statistics.median(values)) for key, values in durations.items()}

def main() -> None:
    x = jnp.arange(1024 * 128, dtype=jnp.float32).reshape(1024, 128) / 1024
    for nested in (False, True):
        mesh, affine = build(nested)
        for flag in (('false', 'true') if not nested else ('true',)):
            compiled = tpuasm_tools.compile(affine, x, mesh=mesh, compiler_options={'xla_enable_custom_call_region_trace': flag})
            np.testing.assert_array_equal(np.asarray(compiled(x)), np.asarray(x) * 2.0 + 1.0)
            bundles = joined_bundles(tpuasm_tools.kernel_listing(compiled, pallas_only=True))
            traces = [index for index, text in enumerate(bundles) if 'vtrace' in text]
            print(f'## {"嵌套区域，" if nested else ""}xla_enable_custom_call_region_trace={flag}：数值检查通过；kernel 段 {len(bundles)} 个 bundle，其中含 vtrace 的 {len(traces)} 个')
            if flag == 'true' and not nested:
                # 打印整个 kernel 段，连续 6 个以上不含 vtrace 的 bundle 折叠成一行。
                print('  清单（[编号] bundle）：')
                index = 0
                while index < len(bundles):
                    run = index
                    while run < len(bundles) and 'vtrace' not in bundles[run]:
                        run += 1
                    if run - index > 6:
                        for k in (index, index + 1):
                            print(f'    [{k}] {bundles[k]}')
                        print(f'    … {run - index - 4} 个 bundle')
                        for k in (run - 2, run - 1):
                            print(f'    [{k}] {bundles[k]}')
                    else:
                        for k in range(index, run):
                            print(f'    [{k}] {bundles[k]}')
                    if run < len(bundles):
                        print(f'    [{run}] {bundles[run]}')
                    index = run + 1
            else:
                print('  ' + '；'.join(bundles[index].strip('{} ') for index in traces))
            print('  XProf，16 次调用的中位数（GTC 高 60 位之差 ΔT，每个计数 1/0.7 ns）：')
            for (device, track, name), value in sorted(device_medians(compiled, x).items()):
                if device == '/device:TPU:0' or track == 'XLA Modules':
                    print(f'    {device} {track} {name}：ΔT = {value}，即 {value / 0.7:.1f} ns')

if __name__ == '__main__':
    main()
