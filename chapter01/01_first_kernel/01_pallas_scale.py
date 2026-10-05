"""第一个 Pallas TPU kernel：单 TC 把 f32[8,128] 从 HBM 搬到 TC VMEM、乘以 2、再搬回 HBM。"""
# 第一步：在 import jax 之前决定 runtime 打开哪颗芯片。
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

def main() -> None:
    # 第二步：SPMD 层面的两级 mesh。device mesh 只含这一颗芯片；TensorCore mesh 声明 kernel 只用其中一个 TensorCore。
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    # 第三步：shard_map 把普通 JAX 函数变成“每颗芯片各执行一份”的程序；这里只有一颗芯片，输入输出都不切分。
    @jax.shard_map(
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )
    def scale(x: jax.Array) -> jax.Array:
        # 第四步：pl.kernel 声明 kernel 的输出类型、执行在哪个 mesh 上、需要哪些临时 buffer。
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
            # 参数依次是：输入 Ref、输出 Ref、scratch_types 中的各个 Ref。它们都是内存的引用，不是数组的值。
            # 输入 Ref 位于 HBM，向量单元不能直接读写，先用 DMA 搬到 TC VMEM，并等待完成。
            pltpu.async_copy(x_hbm, x_vmem, sem).wait()
            # 读 Ref 得到值（位于 TC VREG），对值做运算，再把值写回 Ref。
            x_vmem[...] = x_vmem[...] * 2.0
            # 结果搬回输出 Ref。kernel 没有返回值：输出就是写进 o_hbm 的内容。
            pltpu.async_copy(x_vmem, o_hbm, sem).wait()

        return kernel(x)

    # 第五步：像普通 JAX 函数一样编译、调用。
    x = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
    compiled = tpuasm_tools.compile(scale, x, mesh=mesh)
    np.testing.assert_array_equal(np.asarray(compiled(x)), np.asarray(x) * 2.0)
    print('数值检查通过')

    # kernel 在 XLA 程序中是一条 custom-call；打印编译后的 HLO 中与它相关的行。
    print('\n# 编译后的 HLO')
    print('\n'.join(line.strip()[:160] for line in compiled.as_text().splitlines() if 'custom-call' in line or 'ENTRY' in line or 'ROOT' in line))

if __name__ == '__main__':
    main()
