"""两个 host 的 TPU v4 (2x2x2)：用 podrun 在每个 host 上各启动一个进程，列出 8 颗芯片的坐标、所属进程与 jax.make_mesh 的顺序。运行方式：podrun -- /srv/workspace/venv/bin/python chapter02/05_multi_chip/03_devices_2x2x2.py"""
import jax

def main() -> None:
    # 多 host 时，每个进程在访问设备之前都要先调用它，进程之间才能互相发现。
    jax.distributed.initialize()
    if jax.process_index() != 0:
        return
    print(f'进程数 {jax.process_count()}，全部 device {len(jax.devices())} 个，本进程的 device {len(jax.local_devices())} 个')
    for device in jax.devices():
        print(f'  id={device.id}，coords={device.coords}，process_index={device.process_index}')
    mesh = jax.make_mesh((8,), ('device',))
    print('jax.make_mesh((8,), ...) 的 device id：', [device.id for device in mesh.devices.flat])
    print('对应的芯片坐标：', [tuple(device.coords) for device in mesh.devices.flat])

if __name__ == '__main__':
    main()
