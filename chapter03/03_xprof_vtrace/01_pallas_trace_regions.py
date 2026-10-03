"""在 kernel 中用 jax.named_scope 标出四个区域：发起输入 DMA、等待输入 DMA、计算、输出 DMA（发起并等待）。比较打开与关闭 region trace 时的清单，再从 XProf 读出各区域的时间。"""
import tpu_init
tpu_init.initialise_one_chip()

from collections import defaultdict
from pathlib import Path
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

def build():
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
            with jax.named_scope('compute'):
                x_vmem[...] = x_vmem[...] * 2.0 + 1.0
            with jax.named_scope('store'):
                pltpu.async_copy(x_vmem, o_hbm, sem).wait()

        return kernel(x)

    return mesh, affine

def main() -> None:
    x = jnp.arange(1024 * 128, dtype=jnp.float32).reshape(1024, 128) / 1024
    mesh, affine = build()
    for flag in ('false', 'true'):
        compiled = tpuasm_tools.compile(affine, x, mesh=mesh, compiler_options={'xla_enable_custom_call_region_trace': flag})
        np.testing.assert_array_equal(np.asarray(compiled(x)), np.asarray(x) * 2.0 + 1.0)
        listing = tpuasm_tools.kernel_listing(compiled, pallas_only=True)
        bundles = [line for line in listing.splitlines() if line.startswith('{')]
        traces = [line.split('#')[0].strip(' {};') for line in listing.splitlines() if 'vtrace' in line.split('#')[0]]
        print(f'## xla_enable_custom_call_region_trace={flag}：数值检查通过；kernel 段 {len(bundles)} 个 bundle，其中 vtrace {len(traces)} 条')
        print('  ' + '；'.join(traces))
    events = xprof_tools.device_events(xprof_tools.capture(lambda: [compiled(x).block_until_ready() for _ in range(16)], Path('/tmp/pallas_tpu_tutorial/xprof')))
    durations = defaultdict(list)
    for event in events:
        durations[(event['device'], event['track'], event['name'])].append(xprof_tools.duration_us(event))
    print('## XProf，16 次调用的中位数')
    for (device, track, name), values in sorted(durations.items()):
        print(f'  {device} {track} {name}：{statistics.median(values) * 1000:.0f} ns')

if __name__ == '__main__':
    main()
