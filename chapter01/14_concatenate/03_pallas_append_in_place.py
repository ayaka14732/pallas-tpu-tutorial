"""在已有数组的运行时位置追加若干行：数组作为 jax Ref 传入 kernel，DMA 直接写进它的窗口，不生成新数组。连续追加三次检查结果，并检查编译后的 HLO 与清单。"""
import tpu_init
tpu_init.initialize_one_chip()

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

import tpuasm_tools

ROWS, NEW = 64, 3

def build():
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @pl.kernel(
        out_type=(),
        mesh=tc_mesh,
        scratch_types=(pltpu.SMEM((1,), jnp.int32), pltpu.SemaphoreType.DMA),
        name='append',
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            disable_semaphore_checks=True,
        ),
    )
    def kernel(rows_hbm: Ref, position_hbm: Ref, buffer_hbm: Ref, position_smem: Ref, sem: Ref) -> None:
        pltpu.async_copy(position_hbm, position_smem, sem).wait()
        # 写入的起始行在运行时才知道；数组只有一列 tile，任意行窗口都可以由一次 DMA 描述。
        pltpu.async_copy(rows_hbm, buffer_hbm.at[pl.ds(position_smem[0], NEW)], sem).wait()

    def append(buffer: Ref, rows: jax.Array, position: jax.Array) -> None:
        # kernel 没有输出：它通过 Ref 修改 buffer。
        kernel(rows, position, buffer)

    return append

def main() -> None:
    append = build()
    buffer = jax.new_ref(jnp.zeros((ROWS, 128), jnp.float32))
    rows = jnp.zeros((NEW, 128), jnp.float32)
    position = jnp.zeros((1,), jnp.int32)
    compiled = tpuasm_tools.compile(append, buffer, rows, position)
    expected = np.zeros((ROWS, 128), np.float32)
    for step in range(3):
        start = 5 + NEW * step
        compiled(buffer, jnp.full((NEW, 128), step + 1.0), jnp.array([start], jnp.int32))
        expected[start:start + NEW] = step + 1.0
    print(f'连续追加 3 次（起始行 5、8、11）后数值检查：{bool(np.array_equal(np.asarray(buffer[...]), expected))}；前 16 行的第 0 列：{np.asarray(buffer[...])[:16, 0].tolist()}')
    text = compiled.as_text()
    entry = text[text.index('\nENTRY') + 1:].split('\n}')[0]
    print('## 优化后的 HLO')
    alias = text.split('input_output_alias=')[1].split(', entry_computation_layout')[0] if 'input_output_alias=' in text else '无'
    print(f'  input_output_alias：{alias}')
    for line in entry.splitlines()[1:]:
        print('  ' + line.strip().split(', metadata')[0].split(', custom_call_target')[0][:160])
    listing = tpuasm_tools.kernel_listing(compiled, pallas_only=True)
    print('## 清单中的 DMA')
    print('\n'.join(line.split('#')[0].strip() for line in listing.splitlines() if 'dma.' in line or 'sld' in line))

if __name__ == '__main__':
    main()
