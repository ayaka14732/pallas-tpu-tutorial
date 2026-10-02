"""bf16 的两种打包格式：f32[16,128] 转为 bf16 后按位读出每个 32 bit 位置装的是哪两行；再用 tpuasm 把 vpackc 换成 vpack，比较两种格式。"""
import tpu_init
tpu_init.initialise_one_chip()

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

import tpuasm_tools

def rows_in_words(words: np.ndarray) -> list[tuple[int, int]]:
    """每个 sublane 的 32 bit 位置中，低 16 bit 与高 16 bit 分别是第几行（输入第 r 行的值就是 r）。"""
    low = (words & 0xffff).astype(np.uint16).view(jnp.bfloat16).astype(np.float32)
    high = (words >> 16).astype(np.uint16).view(jnp.bfloat16).astype(np.float32)
    assert np.all(low == low[:, :1]) and np.all(high == high[:, :1])
    return [(int(low[s, 0]), int(high[s, 0])) for s in range(8)]

def main() -> None:
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def pack(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct((8, 128), jnp.uint32),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM((16, 128), jnp.float32), pltpu.VMEM((8, 128), jnp.uint32), pltpu.SemaphoreType.DMA),
            name='pack',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, o_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            o_vmem[...] = pltpu.bitcast(x_vmem[...].astype(jnp.bfloat16), jnp.uint32)
            pltpu.async_copy(o_vmem, o_hbm, sem).wait()

        return kernel(x)

    # 第 r 行的每个元素都等于 r，打包后从位型就能读出行号。
    x = jnp.asarray(np.repeat(np.arange(16, dtype=np.float32)[:, None], 128, axis=1))
    compiled = tpuasm_tools.compile(pack, x, mesh=mesh)
    listing = tpuasm_tools.kernel_listing(compiled)
    print('Mosaic 生成的打包指令：')
    print('\n'.join(line for line in listing.splitlines() if 'vpack' in line or 'vld:' in line))
    print('vpackc：sublane s 的 (低 16 bit, 高 16 bit) =', rows_in_words(np.asarray(compiled(x))))

    def use_vpack(source: str) -> str:
        assert source.count('vpackc.8x128.f32.f16') == 1
        return source.replace('vpackc.8x128.f32.f16', 'vpack.8x128.f32.f16')

    patched = tpuasm_tools.load(tpuasm_tools.replace_listing(tpuasm_tools.serialize(compiled), use_vpack), compiled)
    print('vpack ：sublane s 的 (低 16 bit, 高 16 bit) =', rows_in_words(np.asarray(patched(x))))

if __name__ == '__main__':
    main()
