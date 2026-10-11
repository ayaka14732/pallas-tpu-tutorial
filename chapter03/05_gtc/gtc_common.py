"""本节脚本共用：从主机只读地读出一颗芯片 6 个 ICI 端口的 GTC 寄存器（角色与计数），以及找出一个 JAX device 对应哪个 /dev/accel*。"""
import mmap
import os
from pathlib import Path

import numpy as np

BAR2 = 0x10000000  # /sys/class/accel/accel*/bar_offsets 给出的 BAR2 在 /dev/accel* 中的映射起点
PORT_BASE = 0x37B80000  # 端口 0 的 GTC 寄存器组在 BAR2 中的偏移
PORT_STRIDE = 0x200000  # 相邻端口的寄存器组相隔这么多
# 角色寄存器的取值。global leader：整棵树的时间源；local leader：这颗芯片从这个端口接收上游的时间；follower：其余端口。
ROLES = {0: 'global leader', 1: 'local leader', 2: 'follower'}

def read_ports(chip: int) -> list[tuple[int, int]]:
    """本 host 第 chip 颗芯片（/dev/accel{chip}）的 6 个端口各自的 (角色, 计数)。设备以只读方式打开，寄存器所在的页以只读方式映射，每个寄存器读一次 64 位。"""
    ports = []
    descriptor = os.open(f'/dev/accel{chip}', os.O_RDONLY)
    for port in range(6):
        page = mmap.mmap(descriptor, 4096, mmap.MAP_SHARED, mmap.PROT_READ, offset=BAR2 + PORT_BASE + port * PORT_STRIDE)
        words = np.frombuffer(page, dtype=np.uint64)
        # 偏移 0x00 是角色，0x28 是这个端口的 GTC 计数。
        ports.append((int(words[0]), int(words[5])))
        del words
        page.close()
    os.close(descriptor)
    return ports

def describe(ports: list[tuple[int, int]]) -> str:
    return '、'.join(f'端口 {port} {ROLES[role]}' for port, (role, _) in enumerate(ports))

def interrupt_counts() -> list[int]:
    """本 host 四颗芯片各自的中断总数（/sys/class/accel/accel*/interrupt_counts 各行之和）。"""
    return [sum(int(line.split(':')[1]) for line in Path(f'/sys/class/accel/accel{chip}/interrupt_counts').read_text().splitlines() if ':' in line) for chip in range(4)]

def accel_index(device) -> int:
    """JAX device 对应的 /dev/accel* 编号。device 没有给出这个编号的属性（local_hardware_id 不是它），所以实测：只在这个 device 上反复运行一个小程序，看哪颗芯片的中断数在增加。"""
    import jax
    import jax.numpy as jnp
    x = jax.device_put(jnp.ones((8, 128), jnp.float32), device)
    step = jax.jit(lambda x: x + 1.0)
    step(x).block_until_ready()
    before = interrupt_counts()
    for _ in range(64):
        x = step(x)
    x.block_until_ready()
    increases = [after - earlier for after, earlier in zip(interrupt_counts(), before)]
    chip, = [chip for chip, increase in enumerate(increases) if increase >= 64]
    return chip
