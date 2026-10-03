"""TensorCore 的三条随机数指令：setrngseed 装入状态，getrngseed 读出状态，vrng.8x128.u32 产生一个 u32[8,128] 并推进状态。用 tpuasm 直接执行它们，与主机端 xorshift128+ 模型逐位比较。"""
import tpu_init
tpu_init.initialise_one_chip()

import jax
import numpy as np

import rng_oracle
import tpuasm_tools
from tpuasm_tools import bundle

GAP = bundle('misc: vnop') * 16  # 相邻两条随机数指令之间留足间隔：本节只看语义，时序在第 6 节测量

def store(register: str, tile: int) -> str:
    return bundle(f'vst: vst.8x128 [vmem:0x{tile * 8:x}], {register}')

def body(slot: str, seed: str) -> str:
    """装入 seed 中的状态，然后依次：读状态 → tile 1，vrng → tile 2，读状态 → tile 3，vrng → tile 4，读状态 → tile 5。"""
    text = bundle(f'{slot}: setrngseed {seed}') + GAP
    for tile, instruction in enumerate(('getrngseed v11', 'vrng.8x128.u32 v11', 'getrngseed v11', 'vrng.8x128.u32 v11', 'getrngseed v11'), 1):
        text += bundle(f'{slot}: {instruction}') + GAP + store('v11', tile) + GAP
    return text

def expected(seed_tile: np.ndarray) -> list[np.ndarray]:
    s0, s1 = rng_oracle.state_from_tile(seed_tile)
    tiles = [rng_oracle.state_view(s0, s1)]
    for _ in range(2):
        out, s0, s1 = rng_oracle.vrng(s0, s1)
        tiles += [out, rng_oracle.state_view(s0, s1)]
    return tiles

def main() -> None:
    probe = tpuasm_tools.LccProbe()
    seed = probe.host[:8]
    names = ('getrngseed', '第 1 条 vrng', 'getrngseed', '第 2 条 vrng', 'getrngseed')
    for slot in ('va0', 'va1'):
        tiles = probe.run_tiles(body(slot, 'v10'), repeats=2)[:, 0, :5]
        reference = expected(seed)
        print(f'## {slot}：状态取自输入的前两个 sublane')
        for name, got, want in zip(names, tiles[0], reference):
            print(f'  {name}：与模型一致的 word {int(np.sum(got == want))} / 1024')
        print(f'  两次运行逐位相同：{bool(np.array_equal(tiles[0], tiles[1]))}')
    print('## getrngseed 读回的状态：只有前两个 sublane 有效，读回时重复 4 次')
    tiles = probe.run_tiles(body('va0', 'v10'))[0, 0]
    print(f'  读回的 sublane 0、1 与输入相同：{bool(np.array_equal(tiles[0][:2], seed[:2]))}；sublane 2–7 是 0、1 的重复：{bool(np.array_equal(tiles[0], np.tile(seed[:2], (4, 1))))}')
    print('## 全零状态')
    zero = bundle('va0: vxor.8x128.u32 v15, v10, v10') + GAP
    tiles = probe.run_tiles(zero + body('va0', 'v15'))[0, 0]
    print(f'  两条 vrng 的输出全为 0：{bool(np.all(tiles[1] == 0) and np.all(tiles[3] == 0))}')
    print('## 只有第 5 个生成器的状态非零（lane 10、11 的 sublane 0、1）')
    host = np.zeros_like(probe.host)
    host[:2, 10:12] = probe.host[:2, 10:12]
    probe.x = jax.device_put(host, jax.local_devices()[0])
    tiles = probe.run_tiles(body('va0', 'v10'))[0, 0]
    nonzero = sorted(set(np.nonzero(tiles[1])[1].tolist()))
    print(f'  第 1 条 vrng 输出中非零的 lane：{nonzero}；各 sublane 都非零：{bool(np.all(tiles[1][:, 10:12].any(axis=1)))}')
    # 用主机模型逐步推进第 5 个生成器，第 i 步的 64 位输出应等于 sublane i 的两个 lane 拼成的数。
    s0, s1 = rng_oracle.state_from_tile(host[:8])
    steps = []
    with np.errstate(over='ignore'):
        x, y = s0[5], s1[5]
        for _ in range(8):
            x, y = y, (x ^ (x << np.uint64(23))) ^ y ^ ((x ^ (x << np.uint64(23))) >> np.uint64(17)) ^ (y >> np.uint64(26))
            steps.append(int((y + x) & rng_oracle.MASK))
    got = [int(tiles[1][sublane, 10]) | (int(tiles[1][sublane, 11]) << 32) for sublane in range(8)]
    for sublane in range(8):
        print(f'  sublane {sublane}：0x{got[sublane]:016x}，第 {sublane + 1} 步 0x{steps[sublane]:016x}')

if __name__ == '__main__':
    main()
