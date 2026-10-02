"""显式控制 MXU 的公开接口：push 一次 RHS 后用两块 LHS 复用它；以及 RHS 以 (N,K) 存放、push 时转置。"""
import tpu_init
tpu_init.initialise_one_chip()

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

import mxu_common
import tpuasm_tools

def reuse_rhs(lhs_vmem: Ref, rhs_vmem: Ref, o_vmem: Ref) -> None:
    """RHS 只 push 一次；第一块 LHS 要求换入刚 push 的 RHS，第二块继续用同一份。"""
    pltpu.matmul_push_rhs(rhs_vmem[...], staging_register=0, mxu_index=0)
    for block in range(2):
        rows = pl.ds(block * 16, 16)
        pltpu.matmul_lhs_fifo(lhs_vmem[rows, :], mxu_index=0, load_staged_rhs=0 if block == 0 else None)
        o_vmem[rows, :] = pltpu.matmul_pop_fifo(shape=(16, 128), dtype=jnp.float32, mxu_index=0)

def transposed_rhs(lhs_vmem: Ref, rhs_vmem: Ref, o_vmem: Ref) -> None:
    """rhs_vmem 以 (N,K) 存放，push 时由硬件转置。"""
    pltpu.matmul_push_rhs(rhs_vmem[...], staging_register=0, mxu_index=0, transpose=True)
    for block in range(2):
        rows = pl.ds(block * 16, 16)
        pltpu.matmul_lhs_fifo(lhs_vmem[rows, :], mxu_index=0, load_staged_rhs=0 if block == 0 else None)
        o_vmem[rows, :] = pltpu.matmul_pop_fifo(shape=(16, 128), dtype=jnp.float32, mxu_index=0)

def main() -> None:
    rng = np.random.default_rng(0)
    lhs = jnp.asarray(rng.integers(-8, 8, (32, 128))).astype(jnp.bfloat16)
    rhs = jnp.asarray(rng.integers(-8, 8, (128, 128))).astype(jnp.bfloat16)
    expected = np.asarray(lhs, np.float32) @ np.asarray(rhs, np.float32)
    for name, body, operand in (('RHS 复用', reuse_rhs, rhs), ('RHS 以 (N,K) 存放', transposed_rhs, rhs.T)):
        try:
            compiled = mxu_common.compile_kernel(body, lhs, operand, jax.ShapeDtypeStruct((32, 128), jnp.float32))
        except Exception as error:
            print(f'## {name}：编译失败')
            print(str(error).splitlines()[0][:300])
            print()
            continue
        result = np.asarray(compiled(lhs, operand))
        print(f'## {name}')
        for block in range(2):
            rows = slice(block * 16, block * 16 + 16)
            print(f'  第 {block} 块 LHS：与精确结果一致 {bool(np.array_equal(result[rows], expected[rows]))}，非零元素 {np.count_nonzero(result[rows])}/{np.count_nonzero(expected[rows])}')
        print(mxu_common.summary(tpuasm_tools.kernel_listing(compiled)))
        print()

if __name__ == '__main__':
    main()
