"""两个 TensorCore 的 bf16[2048,2048] @ bf16[2048,2048] → f32[2048,2048]：每个 TensorCore 算 1024 行，LHS 的一半常驻 TC VMEM，RHS 按 256 列的块双缓冲流过，输出块双缓冲写回。各版本只改流水线的头和尾，以及每次 jnp.dot 的行数。"""
import tpu_init
tpu_init.initialise_one_chip()

import importlib

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

import tpuasm_tools

baseline = importlib.import_module('01_jax_matmul')
SIZE = baseline.SIZE
CORES = 2
ROWS = SIZE // CORES  # 每个 TensorCore 的输出行数
BLOCK = 256  # RHS 与输出每块的列数
BLOCKS = SIZE // BLOCK
VARIANTS = {
    'LHS 一次读入': (1, False, False, 256),
    'LHS 分 4 段，同时发出': (4, False, False, 256),
    'LHS 分 4 段，逐段发出': (4, True, False, 256),
    'LHS 逐段发出，最后一块分段写回': (4, True, True, 256),
    'LHS 逐段发出，最后一块分段写回，每次 jnp.dot 512 行': (4, True, True, 512),
}

def build(panels: int, staged: bool, split_last: bool, chunk: int):
    """panels：LHS 分几段读入；staged：各段是否等前一段到达后才发出；split_last：最后一块是否分段写回；chunk：每次 jnp.dot 的行数。"""
    panel_rows = ROWS // panels
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=CORES)

    @jax.shard_map(
        mesh=mesh,
        in_specs=(P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    def matmul(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct((SIZE, SIZE), jnp.float32),
            mesh=tc_mesh,
            scratch_types=(
                pltpu.VMEM((ROWS, SIZE), jnp.bfloat16),          # 本核的 LHS：4 MiB
                pltpu.VMEM((2, SIZE, BLOCK), jnp.bfloat16),      # RHS 块，双缓冲：2 × 1 MiB
                pltpu.VMEM((2, ROWS, BLOCK), jnp.float32),       # 输出块，双缓冲：2 × 1 MiB
                pltpu.SemaphoreType.DMA((panels,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((panels,)),
            ),
            name='matmul',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
                vmem_limit_bytes=15 * 1024 * 1024,
            ),
        )
        def kernel(lhs_hbm: Ref, rhs_hbm: Ref, o_hbm: Ref, lhs_vmem: Ref, rhs_vmem: Ref, o_vmem: Ref, lhs_sems: Ref, rhs_sems: Ref, o_sems: Ref, last_sems: Ref) -> None:
            core = jax.lax.axis_index('tc')
            first_row = pl.multiple_of(core * ROWS, ROWS)

            def load_lhs(panel: int):
                return pltpu.make_async_copy(lhs_hbm.at[pl.ds(first_row + panel * panel_rows, panel_rows)], lhs_vmem.at[pl.ds(panel * panel_rows, panel_rows)], lhs_sems.at[panel])

            def load_rhs(block: int | jax.Array, slot: int | jax.Array):
                return pltpu.make_async_copy(rhs_hbm.at[:, pl.ds(pl.multiple_of(block * BLOCK, BLOCK), BLOCK)], rhs_vmem.at[slot], rhs_sems.at[slot])

            def store(block: int | jax.Array, slot: int | jax.Array):
                return pltpu.make_async_copy(o_vmem.at[slot], o_hbm.at[pl.ds(first_row, ROWS), pl.ds(pl.multiple_of(block * BLOCK, BLOCK), BLOCK)], o_sems.at[slot])

            def store_panel(block: int, panel: int):
                rows = pl.ds(panel * panel_rows, panel_rows)
                return pltpu.make_async_copy(o_vmem.at[block % 2, rows], o_hbm.at[pl.ds(first_row + panel * panel_rows, panel_rows), pl.ds(block * BLOCK, BLOCK)], last_sems.at[panel])

            def compute(slot: int | jax.Array, panel: int | None = None) -> None:
                # 每次只算 chunk 行：一次 jnp.dot 的操作数要放进 TC VREG，放不下的部分溢出到 TC VMEM，块太大时溢出区会超过 TC VMEM 的容量。
                start, count = (0, ROWS) if panel is None else (panel * panel_rows, panel_rows)
                step = min(chunk, count)

                @pl.loop(start, start + count, step=step)
                def _(row: jax.Array) -> None:
                    rows = pl.ds(pl.multiple_of(row, step), step)
                    o_vmem[slot, rows] = jnp.dot(lhs_vmem[rows], rhs_vmem[slot], preferred_element_type=jnp.float32)

            # prologue：发出 LHS 段和第 0 块 RHS。同时发出的 DMA 分享 HBM 带宽，staged 时只先发出第 0 段。
            for panel in range(1 if staged else panels):
                load_lhs(panel).start()
            load_rhs(0, 0).start()
            # 第 0 块：每到一段 LHS 就算这一段的行。
            load_rhs(0, 0).wait()
            load_rhs(1, 1).start()
            for panel in range(panels):
                load_lhs(panel).wait()
                if staged and panel + 1 < panels:
                    load_lhs(panel + 1).start()
                compute(0, panel)
            store(0, 0).start()
            last = BLOCKS - 1 if split_last else BLOCKS

            @pl.loop(1, last)
            def _(block: jax.Array) -> None:
                slot = block % 2
                load_rhs(block, slot).wait()

                @pl.when(block + 1 < BLOCKS)
                def _() -> None:
                    load_rhs(block + 1, 1 - slot).start()

                # 输出 buffer slot 上一次用于 block - 2。
                @pl.when(block >= 2)
                def _() -> None:
                    store(block - 2, slot).wait()

                compute(slot)
                store(block, slot).start()

            if split_last:
                # 最后一块：每算完一段行就立即写回这一段，与后面几段的计算重叠。
                block = BLOCKS - 1
                load_rhs(block, block % 2).wait()
                store(block - 2, block % 2).wait()
                for panel in range(panels):
                    compute(block % 2, panel)
                    store_panel(block, panel).start()
                store(block - 1, (block - 1) % 2).wait()
                for panel in range(panels):
                    store_panel(block, panel).wait()
            else:
                store(BLOCKS - 2, BLOCKS % 2).wait()
                store(BLOCKS - 1, (BLOCKS - 1) % 2).wait()

        return kernel(lhs, rhs)

    return mesh, matmul

def main() -> None:
    lhs, rhs = baseline.inputs(0)
    reference = np.asarray(lhs, np.float32) @ np.asarray(rhs, np.float32)
    pairs = [baseline.inputs(seed) for seed in range(1, 17)]
    clock = tpuasm_tools.KernelClock(num_cores=2)
    for name, (panels, staged, split_last, chunk) in VARIANTS.items():
        mesh, matmul = build(panels, staged, split_last, chunk)
        compiled = tpuasm_tools.compile(matmul, lhs, rhs, mesh=mesh)
        np.testing.assert_array_equal(np.asarray(compiled(lhs, rhs)), reference)
        try:
            listing = tpuasm_tools.kernel_listing(compiled, pallas_only=True)
        except jax.errors.JaxRuntimeError as error:
            # 程序太大时，executable 连同编译器元数据无法序列化：读不到清单，也无法插入 LCC 读数，只能用 XProf。
            print(f'## {name}：数值检查通过；无法序列化 executable：{str(error).splitlines()[0]}')
            times, method = baseline.xprof_times(compiled, pairs), 'XProf'
        else:
            counts = tpuasm_tools.count_mnemonics(listing)
            bundles = sum(line.startswith('{') for line in listing.splitlines())
            print(f'## {name}：数值检查通过；kernel 段 {bundles} 个 bundle；' + '，'.join(f'{mnemonic} {counts[mnemonic]}' for mnemonic in sorted(counts) if mnemonic.split('.')[0] in ('vmatmul', 'vmatpush', 'vdwg', 'dma')))
            times, method = baseline.clock_times(clock, compiled, pairs), 'LCC'
        for op, cycles in times:
            print(f'  {method}，{op}：TensorCore 0 {cycles[0]} 个周期，TensorCore 1 {cycles[1]} 个周期')

if __name__ == '__main__':
    main()
