"""Pallas 的硬件随机数接口：pltpu.prng_seed 与 pltpu.prng_random_bits 在清单中变成什么；输出是否就是第 1 节的 xorshift128+；两个 TensorCore 用同一个种子时的结果。"""
import tpu_init
tpu_init.initialise_one_chip()

import jax
from jax import Ref
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

import rng_oracle
import tpuasm_tools

def build(rows: int, num_cores: int = 1, seeds=lambda core: (7,)):
    """每个 TensorCore 设定种子 seeds(core)，生成 u32[rows,128] 写进输出的第 core 块。"""
    mesh = jax.make_mesh((1,), ('device',))
    tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=num_cores)

    @jax.shard_map(
        mesh=mesh,
        in_specs=(),
        out_specs=P(),
        check_vma=False,
    )
    def draw() -> jax.Array:
        @pl.kernel(
            out_type=jax.ShapeDtypeStruct((num_cores * rows, 128), jnp.uint32),
            mesh=tc_mesh,
            scratch_types=(pltpu.VMEM((rows, 128), jnp.uint32), pltpu.SemaphoreType.DMA),
            name='draw',
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
        )
        def kernel(o_hbm: Ref, bits_vmem: Ref, sem: Ref) -> None:
            core = jax.lax.axis_index('tc')
            pltpu.prng_seed(*seeds(core))
            bits_vmem[...] = pltpu.prng_random_bits((rows, 128)).astype(jnp.uint32)
            pltpu.async_copy(bits_vmem, o_hbm.at[pl.ds(core * rows, rows)], sem).wait()

        return kernel()

    return mesh, draw

def expanded_state(compiled) -> np.ndarray:
    """把最后一条 vrng 换成 getrngseed：kernel 写出的就是种子展开得到的硬件状态。"""
    serialized = tpuasm_tools.serialize(compiled)
    pc = tpuasm_tools.find_bundles(serialized, 'vrng')[-1]
    instruction, = [item.split('#')[0].strip(' {};\n') for item in tpuasm_tools.bundle_text(serialized, pc).split(';') if 'vrng' in item]
    return np.asarray(tpuasm_tools.load(tpuasm_tools.edit_bundles(serialized, {pc: (instruction, instruction.replace('vrng.8x128.u32', 'getrngseed'))}), compiled)())

def threefry2x32(key: tuple[int, int], x0: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """主机上的 threefry2x32（20 轮），与 jax.random 的默认实现相同。"""
    schedule = [np.uint32(key[0]), np.uint32(key[1]), np.uint32(key[0]) ^ np.uint32(key[1]) ^ np.uint32(0x1BD11BDA)]
    with np.errstate(over='ignore'):
        x0, x1 = x0 + schedule[0], x1 + schedule[1]
        for group in range(5):
            for rotation in ((13, 15, 26, 6), (17, 29, 16, 24))[group % 2]:
                x0 = x0 + x1
                x1 = ((x1 << np.uint32(rotation)) | (x1 >> np.uint32(32 - rotation))) ^ x0
            x0, x1 = x0 + schedule[(group + 1) % 3], x1 + schedule[(group + 2) % 3] + np.uint32(group + 1)
    return x0, x1

def main() -> None:
    mesh, draw = build(8)
    compiled = tpuasm_tools.compile(draw, mesh=mesh)
    bits = np.asarray(compiled())
    listing = tpuasm_tools.kernel_listing(compiled, pallas_only=True)
    lines = listing.splitlines()
    seed_line = next(index for index, line in enumerate(lines) if 'setrngseed' in line.split('#')[0])
    before = '\n'.join(lines[:seed_line])
    print(f'## prng_seed(7) + prng_random_bits((8, 128))：kernel 段 {sum(line.startswith("{") for line in lines)} 个 bundle，setrngseed 之前 {sum(line.startswith("{") for line in lines[:seed_line])} 个')
    print('  setrngseed 之前的指令：' + '，'.join(f'{name} {count}' for name, count in tpuasm_tools.count_mnemonics(before).most_common()))
    print('  ' + '；'.join(line.split('#')[0].strip() for line in lines if 'rng' in line.split('#')[0]))
    state = expanded_state(compiled)
    expected, _, _ = rng_oracle.vrng(*rng_oracle.state_from_tile(state))
    print(f'  读回的状态经 xorshift128+ 模型得到的 tile 与 kernel 的输出一致：{bool(np.array_equal(expected, bits))}')
    print('## 只改种子的个数：setrngseed 之前的向量运算')
    for seeds in ((7,), (7, 1), (7, 1, 2)):
        mesh, draw = build(8, seeds=lambda core: seeds)
        try:
            compiled = tpuasm_tools.compile(draw, mesh=mesh)
        except Exception as error:
            print(f'  prng_seed{seeds}：编译失败：{str(error).splitlines()[0].split(": ")[-1]}')
            continue
        lines = tpuasm_tools.kernel_listing(compiled, pallas_only=True).splitlines()
        seed_line = next(index for index, line in enumerate(lines) if 'setrngseed' in line.split('#')[0])
        counts = tpuasm_tools.count_mnemonics('\n'.join(lines[:seed_line]))
        vector = sum(count for name, count in counts.items() if name.startswith('v') and not name.startswith(('vtrace', 'vld', 'vst')))
        print(f'  prng_seed{seeds}：{vector} 条，其中 vshll {counts.get("vshll.8x128.s32", 0)}')
        # 种子展开：计数器的两个字都是元素的序号 i，key 是倒序的种子（只有一个种子时两个字相同），两个输出字异或后交给 setrngseed。
        index = np.arange(1024, dtype=np.uint32).reshape(8, 128)
        x0, x1 = threefry2x32((seeds[-1], seeds[0]), index, index)
        same = np.array_equal(rng_oracle.state_view(*rng_oracle.state_from_tile(x0 ^ x1)), expanded_state(compiled))
        print(f'    状态与主机上的 threefry2x32(key={(seeds[-1], seeds[0])}, 计数器=(i, i)) 两个输出字的异或一致：{bool(same)}')
    for seed in (0, 1):
        mesh, draw = build(8, seeds=lambda core: (seed,))
        print(f'## prng_seed({seed})：输出中 0 的个数 {int(np.sum(np.asarray(tpuasm_tools.compile(draw, mesh=mesh)()) == 0))} / 1024')
    mesh, draw = build(64)
    compiled = tpuasm_tools.compile(draw, mesh=mesh)
    counts = tpuasm_tools.count_mnemonics(tpuasm_tools.kernel_listing(compiled, pallas_only=True))
    print(f'## prng_random_bits((64, 128))：vrng {counts["vrng.8x128.u32"]} 条，setrngseed {counts["setrngseed"]} 条')
    for name, seeds in (('两个 TensorCore 都 prng_seed(7)', lambda core: (7,)), ('prng_seed(7, core)', lambda core: (7, core))):
        mesh, draw = build(8, num_cores=2, seeds=seeds)
        tiles = np.asarray(tpuasm_tools.compile(draw, mesh=mesh)()).reshape(2, 8, 128)
        print(f'## {name}：两个 TensorCore 的输出相同 {bool(np.array_equal(tiles[0], tiles[1]))}')

if __name__ == '__main__':
    main()
