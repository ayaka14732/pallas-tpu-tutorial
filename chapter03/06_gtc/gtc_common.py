"""本节脚本共用：从主机只读地读出一颗芯片 6 个 ICI 端口的 GTC 寄存器（角色与计数）。"""
import mmap
import os

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
