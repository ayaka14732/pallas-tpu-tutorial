"""用 tpuasm 实现 Mosaic 拒绝的运行时页号：以页号为常数 2 的版本为载体，把主机地址计算中的常数 2 换成从 SMEM 读出的 page[0]。"""
import tpu_init
tpu_init.initialise_one_chip()

import importlib

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np

import tpuasm_tools

pages = importlib.import_module('02_pallas_host_dynamic_page')

def main() -> None:
    mesh = jax.make_mesh((1,), ('device',))
    host = jax.NamedSharding(mesh, P(), memory_kind='pinned_host')
    x = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
    page = jnp.array([2, 0], jnp.int32)
    carrier = tpuasm_tools.compile(pages.build(mesh, False), x, page, mesh=mesh, out_shardings=host)
    serialized = tpuasm_tools.serialize(carrier)
    listing = tpuasm_tools.kernel_listing(serialized, pallas_only=True)
    print('## 载体（页号为常数 2）中，主机地址的计算与发给主机的请求')
    print('\n'.join(line.split('#')[0].rstrip() for line in listing.splitlines() if any(key in line for key in ('sadd.s32 s25', 'vsyncset', 'vint', 'sfence'))))

    fence_pc, = tpuasm_tools.find_bundles(serialized, 's0: sfence')
    address_pc, = tpuasm_tools.find_bundles(serialized, 'sadd.s32 s25, 2, s2')
    # 页号 page[0] 已由 DMA 搬到 SMEM 地址 0x3e；在 sfence 之后把它读进未使用的 s14，再用它代替常数 2。
    patched = tpuasm_tools.edit_bundles(serialized, {address_pc: ('sadd.s32 s25, 2, s2', 'sadd.s32 s25, s14, s2')})
    patched = tpuasm_tools.insert_bundles(patched, {fence_pc + 1: '{ s1: sld s14, [smem:0x3e] }'})
    function = tpuasm_tools.load(patched, carrier)
    for target in range(4):
        # 每次的数据都不同，排除输出 buffer 中残留上一次结果的可能。
        value = x + 1000 * target
        result = np.asarray(function(value, jnp.array([target, 0], jnp.int32)))
        np.testing.assert_array_equal(result[target], np.asarray(value))
    print('\n## 改写后：sfence 之后插入 { s1: sld s14, [smem:0x3e] }，地址计算改为 sadd.s32 s25, s14, s2')
    print('page[0] 分别为 0、1、2、3 时，对应的页都写入了正确的数据')

if __name__ == '__main__':
    main()
