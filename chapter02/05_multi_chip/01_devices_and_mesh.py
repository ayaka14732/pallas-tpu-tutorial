"""本 host 的四颗芯片：device 的编号、芯片坐标与 jax.make_mesh 给出的顺序。"""
import tpu_init
tpu_init.initialize_local_chips()

import jax
import numpy as np

def main() -> None:
    print('jax.devices()：')
    for device in jax.devices():
        print(f'  id={device.id}，coords={device.coords}，core_on_chip={device.core_on_chip}，num_cores={device.num_cores}，process_index={device.process_index}')
    mesh = jax.make_mesh((4,), ('device',))
    print('jax.make_mesh((4,), ...) 中第 i 个位置的 device id：', [device.id for device in mesh.devices.flat])
    print('对应的芯片坐标：', [tuple(device.coords) for device in mesh.devices.flat])
    mesh2d = jax.make_mesh((2, 2), ('x', 'y'))
    print('jax.make_mesh((2, 2), ...) 的 device id：', np.vectorize(lambda device: device.id)(mesh2d.devices).tolist())

if __name__ == '__main__':
    main()
