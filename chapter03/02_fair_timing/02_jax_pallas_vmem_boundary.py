"""第一章第 3 节的 TC VMEM 计时边界由外壳的三件事保证：内存空间、布局、多个结果共同的终点。这里依次去掉布局约束、去掉 optimization_barrier、改为给每条 HLO 指令各插一对读数，看边界怎样被破坏。外壳在 tpuasm_tools 中（pinned、vmem_shell、vmem_interval）。"""
import tpu_init
tpu_init.initialize_one_core()

from collections.abc import Callable
from typing import Any

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

import tpuasm_tools

MESH = jax.make_mesh((1,), ('device',))
TC = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)
PARAMS = pltpu.CompilerParams(
    disable_bounds_checks=True,
    disable_semaphore_checks=True,
)
K = 8

def xla_affine(x: jax.Array) -> jax.Array:
    return x * 2.0 + 1.0

def pallas_affine(x: jax.Array) -> jax.Array:
    """操作数和结果都声明在 TC VMEM 的 kernel：没有 DMA，也没有信号量。"""
    @pl.kernel(
        out_type=pltpu.VMEM(x.shape, x.dtype),
        mesh=TC,
        name='affine',
        compiler_params=PARAMS,
    )
    def kernel(x_vmem: Ref, o_vmem: Ref) -> None:
        o_vmem[...] = x_vmem[...] * 2.0 + 1.0

    return kernel(pltpu.with_memory_space_constraint(x, pltpu.VMEM))

def xla_top_k(stable: bool) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
    return lambda x: jax.lax.top_k(x, K, is_stable=stable)

def pallas_top_k(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    @pl.kernel(
        out_type=(pltpu.VMEM((x.shape[0], K), jnp.float32), pltpu.VMEM((x.shape[0], K), jnp.int32)),
        mesh=TC,
        name='top_k',
        compiler_params=PARAMS,
    )
    def kernel(x_vmem: Ref, v_vmem: Ref, i_vmem: Ref) -> None:
        v_vmem[...], i_vmem[...] = jax.lax.top_k(x_vmem[...], K, is_stable=False)

    return kernel(pltpu.with_memory_space_constraint(x, pltpu.VMEM))

def measure(clock: tpuasm_tools.KernelClock, name: str, function: Callable[[jax.Array], Any], inputs: list[np.ndarray], check: Callable[[Any, np.ndarray], None], *, show_instructions: bool = False, **options: bool) -> None:
    """编译、列出边界的证据，再在区间两端各读一次 LCC：每份输入运行一次，输出全部核对。show_instructions 为 True 时另外列出区间内向量指令的条数。"""
    try:
        compiled = tpuasm_tools.compile(tpuasm_tools.vmem_shell(function, MESH, **options), inputs[0], mesh=MESH)
    except jax.errors.JaxRuntimeError as error:
        print(f'  {name}：编译失败：{str(error).split(", metadata=")[0]}')
        return
    first, last, layouts, inside, moves, listing = tpuasm_tools.vmem_interval(compiled)
    counts = tpuasm_tools.vector_counts(listing)
    timed = clock.instrument(compiled, [first, last])
    samples = []
    for x in inputs:
        result = timed(x)
        jax.block_until_ready(result)
        readings = clock.read(2)[0].astype(np.int64)
        samples.append(int(readings[1] - readings[0]) - tpuasm_tools.CLOCK_OVERHEAD)
        check(result, x)
    print(f'  {name}：{"；".join(layouts)}')
    print(f'    区间是 bundle [{first}, {last})，其中的 HLO 指令：{"、".join(inside)}；DMA：{"、".join(moves) or "无"}')
    if show_instructions:
        print(f'    区间内的向量指令：{"，".join(f"{mnemonic} {counts[mnemonic]}" for mnemonic in sorted(counts))}')
    print(f'    {len(samples)} 份输入的周期数：{samples}')

def check_affine(result: jax.Array, x: np.ndarray) -> None:
    np.testing.assert_array_equal(np.asarray(result), x * 2.0 + 1.0)

def check_top_k(result: tuple[jax.Array, jax.Array], x: np.ndarray) -> None:
    order = np.argsort(-x, axis=1, kind='stable')[:, :K]
    np.testing.assert_array_equal(np.asarray(result[0]), np.take_along_axis(x, order, axis=1))
    np.testing.assert_array_equal(np.asarray(result[1]), order)

def main() -> None:
    clock = tpuasm_tools.KernelClock(num_cores=1)
    rng = np.random.default_rng(0)
    wide = [rng.standard_normal((1024, 128), dtype=np.float32) for _ in range(8)]
    narrow = [rng.standard_normal((16, 8), dtype=np.float32) for _ in range(8)]
    # 每行是 0 到 127 的随机排列：没有并列，top-k 的下标只有一个正确答案。
    rows = [np.stack([rng.permutation(128) for _ in range(8)]).astype(np.float32) for _ in range(8)]
    print('## y = 2x + 1，f32[1024,128]')
    measure(clock, 'XLA', xla_affine, wide, check_affine, show_instructions=True)
    measure(clock, 'Pallas', pallas_affine, wide, check_affine, show_instructions=True)
    print('## y = 2x + 1，f32[16,8]')
    measure(clock, 'XLA', xla_affine, narrow, check_affine)
    measure(clock, 'Pallas', pallas_affine, narrow, check_affine)
    print('## y = 2x + 1，f32[16,8]，只去掉布局约束')
    measure(clock, 'XLA', xla_affine, narrow, check_affine, constrain=False)
    measure(clock, 'Pallas', pallas_affine, narrow, check_affine, constrain=False)
    print('## 两个结果：top-8，f32[8,128]')
    measure(clock, 'XLA（is_stable=True）', xla_top_k(True), rows, check_top_k)
    measure(clock, 'XLA（is_stable=False）', xla_top_k(False), rows, check_top_k)
    measure(clock, 'Pallas', pallas_top_k, rows, check_top_k)
    print('## top-8，f32[8,128]，只去掉 optimization_barrier')
    measure(clock, 'XLA（is_stable=True）', xla_top_k(True), rows, check_top_k, barrier=False)
    measure(clock, 'Pallas', pallas_top_k, rows, check_top_k, barrier=False)
    print('## top-8，f32[8,128]：改为给每条 HLO 指令各插一对读数')
    for name, function in (('XLA（is_stable=True）', xla_top_k(True)), ('Pallas', pallas_top_k)):
        compiled = tpuasm_tools.compile(tpuasm_tools.vmem_shell(function, MESH), rows[0], mesh=MESH)
        _, _, _, inside, _, _ = tpuasm_tools.vmem_interval(compiled)
        inputs = iter(rows)
        cycles = {op: value[0] for op, value in clock.time_ops(compiled, lambda timed: jax.block_until_ready(timed(next(inputs))), samples=len(rows))}
        print(f'  {name}：' + '，'.join(f'{op} {cycles[op]}' for op in inside) + f'；合计 {sum(cycles[op] for op in inside)}')

if __name__ == '__main__':
    main()
