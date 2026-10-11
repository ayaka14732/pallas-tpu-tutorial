"""时钟同步树的角色：在 TPU v4 (2x2x2) 上，从主机只读地读出每颗芯片 6 个 ICI 端口的 GTC 角色寄存器与计数，比较 JAX 会话建立之前与之后。运行方式：podrun -- /srv/workspace/venv/bin/python 本文件。"""
import time

from gtc_common import ROLES, accel_index, describe, read_ports

def main() -> None:
    # 会话建立之前：还没有 import jax，本进程没有初始化 TPU。
    before = [read_ports(chip) for chip in range(4)]
    import jax
    jax.distributed.initialize()
    lines = []
    for device in jax.local_devices():
        # device 的编号不是 /dev/accel* 的编号，要实测对应关系。
        chip = accel_index(device)
        first = read_ports(chip)
        time.sleep(0.01)
        second = read_ports(chip)
        idle = sorted({count for _, count in before[chip]})
        roles = [role for role, _ in first]
        selected, = [port for port, role in enumerate(roles) if role != 2]
        advancing = all(later > earlier for (_, earlier), (_, later) in zip(first, second))
        lines.append(
            f'device {device.id}，坐标 {tuple(device.coords)}，进程 {device.process_index}，/dev/accel{chip}\n'
            f'  会话之前：6 个端口的角色 {sorted({ROLES[role] for role, _ in before[chip]})}，计数 {idle}\n'
            f'  会话之中：{describe(first)}\n'
            f'  6 个端口的计数相隔 10 ms 两次读数都在增加：{advancing}；端口 {selected} 增加 {second[selected][1] - first[selected][1]}'
        )
    # 每个进程把自己 4 颗芯片的结果一次打印出来。
    print('\n'.join(lines), flush=True)

if __name__ == '__main__':
    main()
