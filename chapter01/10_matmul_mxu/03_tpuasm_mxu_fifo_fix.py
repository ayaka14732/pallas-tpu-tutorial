"""用 tpuasm 查明并修正上一个实验的错误：先运行一个装入另一份 RHS 的矩阵乘法，再运行被测 kernel，判断每块结果用的是哪份 RHS。"""
import tpu_init
tpu_init.initialise_one_chip()

import importlib

import jax
from jax import Ref
import jax.numpy as jnp
import numpy as np
from tpuasm import executable_programs, format_assembly, parse_assembly

import mxu_common
import tpuasm_tools

fifo = importlib.import_module('02_pallas_mxu_fifo')
DWG_MATMUL = 'vmatmul.packed.dwg.8x128.f16 (gmr0, gsfn0, mrf0)'

def first_pc(serialized: bytes, prefix: str) -> int:
    (_, _, image), = executable_programs(serialized)
    program = parse_assembly(format_assembly(image, target=tpuasm_tools.TARGET))
    return next(pc for pc, bundle in enumerate(program.bundles) if any(instruction.mnemonic.startswith(prefix) for instruction in bundle.instructions))

def main() -> None:
    rng = np.random.default_rng(0)
    lhs = jnp.asarray(rng.integers(-8, 8, (32, 128))).astype(jnp.bfloat16)
    rhs = jnp.asarray(rng.integers(-8, 8, (128, 128))).astype(jnp.bfloat16)
    other = jnp.asarray(rng.integers(-8, 8, (128, 128))).astype(jnp.bfloat16)
    expected = np.asarray(lhs, np.float32) @ np.asarray(rhs, np.float32)
    stale = np.asarray(lhs, np.float32) @ np.asarray(other, np.float32)

    def dot(lhs_vmem: Ref, rhs_vmem: Ref, o_vmem: Ref) -> None:
        o_vmem[...] = jnp.dot(lhs_vmem[...], rhs_vmem[...], preferred_element_type=jnp.float32)

    # 这个正确的 kernel 把 other 装入 MXU0 的 gains，作为“上一个 kernel 留下的状态”。
    previous = mxu_common.compile_kernel(dot, lhs, other, jax.ShapeDtypeStruct((32, 128), jnp.float32))

    def classify(result: np.ndarray) -> list[str]:
        labels = []
        for block in range(2):
            rows = slice(block * 16, block * 16 + 16)
            if np.array_equal(result[rows], expected[rows]):
                labels.append('正确')
            elif np.array_equal(result[rows], stale[rows]):
                labels.append('用了上一个 kernel 的 RHS')
            elif not np.any(result[rows]):
                labels.append('全零')
            else:
                labels.append('其他')
        return labels

    versions = []
    for name, body, operand, staging in (('RHS 复用', fifo.reuse_rhs, rhs, 'gsfn0'), ('RHS 以 (N,K) 存放', fifo.transposed_rhs, rhs.T, 'gsft0')):
        compiled = mxu_common.compile_kernel(body, lhs, operand, jax.ShapeDtypeStruct((32, 128), jnp.float32))
        serialized = tpuasm_tools.serialize(compiled)
        pc = first_pc(serialized, 'vmatmul.packed.dwg')
        # 修正：在第一次 vmatmul 之前单独发出 vdwg，从正确的暂存区装入 gains；两次 vmatmul 都不再带 .dwg。
        fixed = tpuasm_tools.insert_bundles(
            tpuasm_tools.replace_listing(serialized, lambda source: source.replace(DWG_MATMUL, 'vmatmul.packed.8x128.f16 mrf0')),
            {pc: f'{{ vx0: vdwg.128x128.f16 gmr0, {staging} }}'},
        )
        versions.append((name, operand, compiled, tpuasm_tools.load(fixed, compiled)))

    # 先运行修正后的版本，此前进程中没有运行过任何装入其他 RHS 的 kernel。
    for name, operand, _, fixed in versions:
        print(f'## {name}：修正后，此前未运行装入其他 RHS 的 kernel：第 0、1 块分别为 {classify(np.asarray(fixed(lhs, operand)))}')
    print()
    for name, operand, compiled, fixed in versions:
        print(f'## {name}：每次先运行装入 other 的 kernel')
        for label, function in (('原样', compiled), ('修正后', fixed)):
            np.asarray(previous(lhs, other))
            print(f'  {label}：第 0、1 块分别为 {classify(np.asarray(function(lhs, operand)))}')
        print()

if __name__ == '__main__':
    main()
