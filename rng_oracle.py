"""TensorCore 硬件随机数生成器的主机端模型：64 个 xorshift128+ 生成器，每个占相邻两个 lane。"""
import numpy as np

MASK = np.uint64(0xFFFFFFFFFFFFFFFF)

def state_from_tile(tile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """setrngseed 从一个 u32[8,128] 的前两个 sublane 取状态：生成器 k 的 s0 来自 lane 2k，s1 来自 lane 2k+1，sublane 0 是低 32 位、sublane 1 是高 32 位。"""
    words = tile[:2].astype(np.uint64)
    state = words[0] | (words[1] << np.uint64(32))
    return state[0::2].copy(), state[1::2].copy()

def state_view(s0: np.ndarray, s1: np.ndarray) -> np.ndarray:
    """getrngseed 写出的 u32[8,128]：偶数 sublane 是低 32 位、奇数 sublane 是高 32 位，偶数 lane 是 s0、奇数 lane 是 s1，重复 4 次。"""
    state = np.empty(128, np.uint64)
    state[0::2], state[1::2] = s0, s1
    rows = np.stack([state & np.uint64(0xFFFFFFFF), state >> np.uint64(32)]).astype(np.uint32)
    return np.tile(rows, (4, 1))

def vrng(s0: np.ndarray, s1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """一条 vrng.8x128.u32：每个生成器连续推进 8 步，第 i 步的 64 位输出写进 sublane i 的两个 lane。返回 (输出, 新 s0, 新 s1)。"""
    s0, s1 = s0.copy(), s1.copy()
    out = np.empty((8, 128), np.uint32)
    with np.errstate(over='ignore'):
        for sublane in range(8):
            x, y = s0, s1
            s0 = y
            x = x ^ ((x << np.uint64(23)) & MASK)
            s1 = x ^ y ^ (x >> np.uint64(17)) ^ (y >> np.uint64(26))
            z = (s1 + y) & MASK
            out[sublane, 0::2] = (z & np.uint64(0xFFFFFFFF)).astype(np.uint32)
            out[sublane, 1::2] = (z >> np.uint64(32)).astype(np.uint32)
    return out, s0, s1
