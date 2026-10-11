"""单 TC 的类型转换：bf16↔f32、f32↔int32、int8↔int32，各打印清单中的转换指令。"""
import tpu_init
tpu_init.initialize_one_chip()

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from conversion_common import convert
import tpuasm_tools

def main() -> None:
    mesh = jax.make_mesh((1,), ('device',))
    rng = np.random.default_rng(0)
    floats = (rng.standard_normal((32, 128)) * 100).astype(np.float32)
    # 含 ±0.5、±1.5、±2.5 等恰在两整数中间的值，以确定 f32→int32 的舍入方式。
    floats[0, :8] = [0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 2.7, -2.7]
    cases = (
        ('bf16[16,128] → f32', floats[:16].astype(ml_dtypes.bfloat16), jnp.float32),
        ('f32[16,128] → bf16', floats[:16], jnp.bfloat16),
        ('f32[8,128] → int32', floats[:8], jnp.int32),
        ('int32[8,128] → f32', np.round(floats[:8]).astype(np.int32), jnp.float32),
        ('int8[32,128] → int32', np.clip(floats, -128, 127).astype(np.int8), jnp.int32),
        ('int32[32,128] → int8', np.clip(floats, -128, 127).astype(np.int32), jnp.int8),
    )
    for name, host, dtype in cases:
        x = jnp.asarray(host)
        try:
            compiled = convert(mesh, x, dtype)
        except Exception as error:
            print(f'## {name}：编译失败')
            print(str(error).splitlines()[0])
            print()
            continue
        result = np.asarray(compiled(x))
        expected = np.asarray(jax.jit(lambda x: x.astype(dtype))(x))
        print(f'## {name}：与 XLA 的 astype 逐元素一致：{bool(np.array_equal(result, expected))}')
        if name == 'f32[8,128] → int32':
            print('f32 输入：', host[0, :8].tolist(), '→ int32：', result[0, :8].tolist())
        listing = tpuasm_tools.kernel_listing(compiled)
        counts = tpuasm_tools.count_mnemonics(listing)
        print('向量指令：', {mnemonic: count for mnemonic, count in sorted(counts.items()) if mnemonic.startswith('v') and not mnemonic.startswith(('vsync', 'vwait', 'vtrace'))})
        print('\n'.join(line for line in listing.splitlines() if any(f'{slot}: ' in line for slot in ('vld', 'vst', 'va0', 'va1', 'vx0', 'vx1', 'vr0', 'vr1'))))
        print()

if __name__ == '__main__':
    main()
