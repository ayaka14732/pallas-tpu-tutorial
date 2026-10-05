"""split-chip 模式：runtime 把一颗芯片的两个 TensorCore 当作两个 device；kernel 用 shard_map 在两个 device 上各算一半，与 Megacore 的写法对照。"""
import os

import tpu_init
tpu_init.initialize_one_chip()
# 这是 libtpu 的启动参数，必须在 TPU 初始化之前设置；它改变 runtime 的 device 划分，不是某次编译的选项。
os.environ['LIBTPU_INIT_ARGS'] = f"{os.environ.get('LIBTPU_INIT_ARGS', '')} --deepsea_chip_config_name=legacy".strip()

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

import tpuasm_tools

def main() -> None:
    devices = jax.devices()
    print('devices：', [(device.id, device.coords, device.core_on_chip, device.num_cores) for device in devices])
    # jax.make_mesh 只接受每颗芯片一个 device 的 Megacore 模式，这里直接用 device 列表构造 Mesh。
    mesh = jax.sharding.Mesh(np.array(devices), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P('device'),
        out_specs=P('device'),
        check_vma=False,
    )
    def scale(x: jax.Array) -> jax.Array:
        # shard_map 已经把 x 沿行切成两半：每个 device（即每个 TensorCore）拿到 f32[32,128]。
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.SemaphoreType.DMA),
            name='scale',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            x_vmem[...] = x_vmem[...] * 2.0
            pltpu.async_copy(x_vmem, o_hbm, sem).wait()

        return kernel(x)

    x = jax.device_put(jnp.arange(64 * 128, dtype=jnp.float32).reshape(64, 128), jax.NamedSharding(mesh, P('device')))
    compiled = tpuasm_tools.compile(scale, x, mesh=mesh)
    np.testing.assert_array_equal(np.asarray(compiled(x)), np.asarray(x) * 2.0)
    listing = tpuasm_tools.kernel_listing(compiled, pallas_only=True)
    counts = tpuasm_tools.count_mnemonics(listing)
    bundles = sum(line.startswith('{') for line in listing.splitlines())
    print(f'数值检查通过；kernel 段 {bundles} 个 bundle；vld {counts["vld.8x128"]}、vmul {counts["vmul.8x128.f32"]}、vst {counts["vst.8x128"]}；vsyncadd.remote {counts["vsyncadd.remote.s32"]}')
    print(listing)

if __name__ == '__main__':
    main()
