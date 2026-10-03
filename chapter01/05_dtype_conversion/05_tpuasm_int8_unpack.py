"""int8 → int32 的三条指令逐条验证：先用 pltpu.bitcast 读出 int8 的打包格式，再用 tpuasm 单独执行 vld.sshfl 与编译器算移位量的几条指令，看它们各自做了什么。"""
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
from tpuasm_tools import bundle

SHUFFLES = ((0x76543210, 0x0), (0x01234567, 0x0), (0x11110000, 0x0), (0x11110000, 0x2), (0x33221100, 0x0))

def int8_words() -> np.ndarray:
    """int8[32,128] 的第 r 行全为 r，按位重新解释成 u32[8,128]。"""
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def words(x: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct((8, 128), jnp.uint32),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM((32, 128), jnp.int8), pltpu.VMEM((8, 128), jnp.uint32), pltpu.SemaphoreType.DMA),
            name='words',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, o_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            o_vmem[...] = pltpu.bitcast(x_vmem[...], jnp.uint32)
            pltpu.async_copy(o_vmem, o_hbm, sem).wait()

        return kernel(x)

    x = jnp.asarray(np.repeat(np.arange(32, dtype=np.int8)[:, None], 128, axis=1))
    return np.asarray(tpuasm_tools.compile(words, x, mesh=mesh)(x))

def source_sublanes(tile: np.ndarray, source: np.ndarray) -> list[int]:
    """tile 的每个 sublane 等于 source（输入的前 16 行）的第几行。"""
    return [next(row for row in range(len(source)) if np.array_equal(tile[sublane], source[row])) for sublane in range(8)]

def main() -> None:
    words = int8_words()
    assert np.all(words == words[:, :1])
    print('## int8[32,128] 按位读成 u32[8,128]：第 s 个 sublane 的字节 0–3 来自第几行')
    for sublane in range(8):
        print(f'  sublane {sublane}：{[int(words[sublane, 0] >> (8 * byte) & 0xFF) for byte in range(4)]}')

    probe = tpuasm_tools.LccProbe()
    print('## vld.sshfl：输出的第 s 个 sublane 来自 TC VMEM 中的第几行')
    body = ''.join(bundle(f'vld: vld.sshfl.8x128 v13, [vmem:0x{address:x}], 0x{pattern:08x}') + bundle(f'vst: vst.8x128 [vmem:0x{tile * 8:x}], v13') for tile, (pattern, address) in enumerate(SHUFFLES, 1))
    tiles = probe.run_tiles(body)[0, 0]
    for (pattern, address), tile in zip(SHUFFLES, tiles):
        print(f'  地址 0x{address:x}，模式 0x{pattern:08x}：{source_sublanes(tile, probe.host[:16])}')

    print('## 编译器算出的移位量：v4 = 24 − ((vlaneseq >> 4) & 0x18)')
    body = bundle('va0: vlaneseq.8x128.u32 v1') + bundle('va1: vshrl.8x128.s32 v2, v1, 0x4') + bundle('va0: vand.8x128.u32 v3, 0x18, v2') + bundle('va0: vsub.8x128.s32 v4, 24, v3') + bundle('vst: vst.8x128 [vmem:0x8], v1') + bundle('vst: vst.8x128 [vmem:0x10], v4')
    lanes, shifts = probe.run_tiles(body)[0, 0, :2]
    print(f'  vlaneseq 第 s 个 sublane 第 0、1、127 个 lane：{[[int(lanes[s, c]) for c in (0, 1, 127)] for s in range(8)]}')
    assert np.all(shifts == shifts[:, :1])
    print(f'  第 s 个 sublane 的左移量：{[int(shifts[s, 0]) for s in range(8)]}')

if __name__ == '__main__':
    main()
