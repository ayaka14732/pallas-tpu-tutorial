"""本节实验共用：两输入单 TC kernel 的构造，以及按 MXU 状态统计清单中的矩阵指令。"""
from collections.abc import Callable
import re

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

import tpuasm_tools

def compile_kernel(body: Callable[[Ref, Ref, Ref], None], lhs: jax.Array, rhs: jax.Array, out: jax.ShapeDtypeStruct) -> jax.stages.Compiled:
    """lhs、rhs 整块进入 TC VMEM，body(lhs_vmem, rhs_vmem, o_vmem) 写出结果，再整块写回 HBM。"""
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

    @jax.shard_map(
        mesh=mesh,
        in_specs=(P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    def function(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=out,
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM(lhs.shape, lhs.dtype), pltpu.VMEM(rhs.shape, rhs.dtype), pltpu.VMEM(out.shape, out.dtype), pltpu.SemaphoreType.DMA),
            name='matmul',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(lhs_hbm: Ref, rhs_hbm: Ref, o_hbm: Ref, lhs_vmem: Ref, rhs_vmem: Ref, o_vmem: Ref, sem: Ref) -> None:
            pltpu.async_copy(lhs_hbm, lhs_vmem, sem).wait()
            pltpu.async_copy(rhs_hbm, rhs_vmem, sem).wait()
            body(lhs_vmem, rhs_vmem, o_vmem)
            pltpu.async_copy(o_vmem, o_hbm, sem).wait()

        return kernel(lhs, rhs)

    return tpuasm_tools.compile(function, lhs, rhs, mesh=mesh)

def summary(listing: str) -> str:
    """矩阵相关指令按“助记符 目的”计数：push 进入哪个暂存区，vdwg 装载哪个 gains，matmul 与 pop 用哪个结果队列。"""
    counts: dict[str, int] = {}
    for match in re.finditer(r'(vx0|vx1|vr0|vr1|va0|va1): (?:@!?p[0-9]+ )?([a-z][\w.]*) ?([^;}#]*)', listing):
        mnemonic, text = match[2], match[3].strip()
        # 多目的指令的目的操作数是一个元组，例如 (gmr0, gsfn0, mrf0)。
        head, tail = (text[:text.index(')') + 1], text[text.index(')') + 1:]) if text.startswith('(') else ('', text)
        operands = ([head] if head else []) + [item.strip() for item in tail.split(',') if item.strip()]
        if mnemonic.startswith('vmatpush'):
            key = f'{mnemonic} → {operands[0]}'
        elif mnemonic.startswith('vdwg'):
            key = f'{mnemonic} {operands[0]} ← {operands[1]}'
        elif mnemonic.startswith('vmatmul'):
            key = f'{mnemonic} → {operands[0]}'
        elif mnemonic == 'vpop.8x128' and operands[1].startswith('mrf'):
            key = f'vpop ← {operands[1]}'
        elif mnemonic.startswith(('vpack', 'vunpack', 'vxpose', 'vadd', 'vmul')):
            key = mnemonic
        else:
            continue
        counts[key] = counts.get(key, 0) + 1
    return '\n'.join(f'  {key}：{value}' for key, value in sorted(counts.items()))
