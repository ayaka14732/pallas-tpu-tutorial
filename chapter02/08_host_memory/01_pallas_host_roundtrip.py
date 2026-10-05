"""kernel 直接读写主机内存：输入与输出都在 pinned host memory，数据经 HOST → HBM → TC VMEM 计算 2x + 1，再经 HBM → HOST 写回；另试 HOST 与 TC VMEM 之间直接 DMA。"""
import tpu_init
tpu_init.initialize_one_chip()

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

import tpuasm_tools

def build(mesh: jax.sharding.Mesh, direct: bool):
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def roundtrip(x: jax.Array) -> jax.Array:
        @pl.kernel(
            # 第一个输出在主机内存；第二个输出只作为 HBM 中转 buffer（Mosaic 不允许在 scratch 中分配 HBM）。
            out_type=(pl.MemoryRef(jax.core.ShapedArray(x.shape, x.dtype), pl.HOST), pltpu.HBM(x.shape, x.dtype)),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.SemaphoreType.DMA),
            name='roundtrip',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_host: Ref, o_host: Ref, staging_hbm: Ref, vmem: Ref, sem: Ref) -> None:
            if direct:
                pltpu.async_copy(x_host, vmem, sem).wait()
                vmem[...] = vmem[...] * 2.0 + 1.0
                pltpu.async_copy(vmem, o_host, sem).wait()
                return
            pltpu.async_copy(x_host, staging_hbm, sem).wait()
            pltpu.async_copy(staging_hbm, vmem, sem).wait()
            vmem[...] = vmem[...] * 2.0 + 1.0
            pltpu.async_copy(vmem, staging_hbm, sem).wait()
            pltpu.async_copy(staging_hbm, o_host, sem).wait()

        # 输入与输出都要求放在主机内存。
        result, _ = kernel(pltpu.with_memory_space_constraint(x, pl.HOST))
        return pltpu.with_memory_space_constraint(result, pl.HOST)

    return roundtrip

def main() -> None:
    mesh = jax.make_mesh((1,), ('device',))
    host = jax.NamedSharding(mesh, P(), memory_kind='pinned_host')
    x = jax.device_put(jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128), host)
    print(f'输入 memory_kind：{x.sharding.memory_kind}')
    for direct in (False, True):
        name = 'HOST ↔ TC VMEM 直接 DMA' if direct else 'HOST → HBM → TC VMEM → HBM → HOST'
        try:
            compiled = tpuasm_tools.compile(build(mesh, direct), x, mesh=mesh, out_shardings=host)
        except Exception as error:
            print(f'## {name}：编译失败：{str(error).splitlines()[0]}')
            continue
        result = compiled(x)
        np.testing.assert_array_equal(np.asarray(result), np.asarray(x) * 2.0 + 1.0)
        print(f'## {name}：数值检查通过；输出 memory_kind：{result.sharding.memory_kind}')
        print(tpuasm_tools.kernel_listing(compiled, pallas_only=True))

if __name__ == '__main__':
    main()
