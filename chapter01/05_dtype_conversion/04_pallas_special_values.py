"""类型转换的边界情况：f32 → int32 的越界与 NaN，f32 → bf16 的舍入、NaN、非规格化数与溢出，int32 → f32 的舍入，bf16 → f32 的非规格化数。每组与 XLA 的 astype 比较，并列出按位的结果。"""
import tpu_init
tpu_init.initialize_one_chip()

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from conversion_common import convert

def run(mesh: jax.sharding.Mesh, host: np.ndarray, dtype: jnp.dtype, count: int) -> tuple[np.ndarray, bool]:
    """转换 host（只有第 0 行的前 count 个元素是测试值），返回这些元素的结果，以及整个数组是否与 XLA 一致。"""
    x = jnp.asarray(host)
    result = np.asarray(convert(mesh, x, dtype)(x))
    expected = np.asarray(jax.jit(lambda x: x.astype(dtype))(x))
    return result[0, :count], bool(np.array_equal(result.view(np.uint8), expected.view(np.uint8)))

def place(values: np.ndarray, rows: int) -> np.ndarray:
    host = np.zeros((rows, 128), values.dtype)
    host[0, :len(values)] = values
    return host

def main() -> None:
    mesh = jax.make_mesh((1,), ('device',))

    print('## f32 → int32')
    values = np.array([3e9, -3e9, np.inf, -np.inf, np.nan, 2147483520.0, 2147483648.0, -2147483648.0], np.float32)
    result, same = run(mesh, place(values, 8), jnp.int32, len(values))
    for value, converted in zip(values, result):
        print(f'  {float(value):>14.10g} → {int(converted)}')
    print(f'  与 XLA 按位一致：{same}')

    print('## f32 → bf16（按位）')
    words = np.array([0x3F808000, 0x3F818000, 0x3F808001, 0x3F817FFF, 0x7FC00001, 0x7F800001, 0xFF800001, 0x007FFFFF, 0x807FFFFF, 0x80000001, 0x00800000, 0x7F7FFFFF, 0x7F7F7FFF], np.uint32)
    notes = ('恰在中间，低位为偶', '恰在中间，低位为奇', '略大于中间', '略小于中间', '静默 NaN，带载荷', '发信号 NaN', '负的发信号 NaN', '最大的非规格化数', '绝对值最大的负非规格化数', '绝对值最小的负非规格化数', '最小的规格化数', '最大的有限数', '最大有限 bf16 加不到半个单位')
    result, same = run(mesh, place(words, 16).view(np.float32), jnp.bfloat16, len(words))
    reference = words.view(np.float32).astype(ml_dtypes.bfloat16)
    for word, converted, nearest, note in zip(words, result.view(np.uint16), reference.view(np.uint16), notes):
        print(f'  0x{int(word):08x} → 0x{int(converted):04x}（按 IEEE 就近舍入、保留非规格化数应为 0x{int(nearest):04x}）  {note}')
    print(f'  与 XLA 按位一致：{same}')

    print('## int32 → f32')
    integers = np.array([16777217, 16777218, 16777219, -16777217, 33554433, 33554435, 2147483647], np.int32)
    result, same = run(mesh, place(integers, 8), jnp.float32, len(integers))
    for integer, converted in zip(integers, result):
        print(f'  {int(integer):>11} → {float(converted):.1f}')
    print(f'  与 XLA 按位一致：{same}')

    print('## bf16 → f32（按位）')
    halves = np.array([0x0001, 0x8001, 0x007F, 0x7FC1, 0x7F81], np.uint16)
    result, same = run(mesh, place(halves, 16).view(ml_dtypes.bfloat16), jnp.float32, len(halves))
    for half, converted in zip(halves, result.view(np.uint32)):
        print(f'  0x{int(half):04x} → 0x{int(converted):08x}')
    print(f'  与 XLA 按位一致：{same}')

if __name__ == '__main__':
    main()
