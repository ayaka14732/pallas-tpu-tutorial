"""MXU 上的矩阵乘法：从 bf16[16,128] @ bf16[128,128] 出发，只改 RHS 的存放方向、dtype、M、K，统计矩阵指令并检查数值。"""
import tpu_init
tpu_init.initialise_one_chip()

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import ml_dtypes
import numpy as np

import mxu_common
import tpuasm_tools

def main() -> None:
    info = pltpu.get_tpu_info()
    print(f'pltpu.get_tpu_info()：num_mxus={info.num_mxus}，mxu_column_size={info.mxu_column_size}，num_accumulators={info.num_accumulators}')
    rng = np.random.default_rng(0)
    # (名称, M, K, N, 输入 dtype, RHS 是否以 (N,K) 存放, jnp.dot 的 precision)
    cases = (
        ('bf16[16,128] @ bf16[128,128]', 16, 128, 128, ml_dtypes.bfloat16, False, None),
        ('bf16[16,128] @ bf16[128,128]ᵀ（RHS 以 (N,K) 存放）', 16, 128, 128, ml_dtypes.bfloat16, True, None),
        ('f32[16,128] @ f32[128,128]', 16, 128, 128, np.float32, False, None),
        ('f32[16,128] @ f32[128,128]，precision=HIGHEST', 16, 128, 128, np.float32, False, jax.lax.Precision.HIGHEST),
        ('bf16[128,128] @ bf16[128,128]', 128, 128, 128, ml_dtypes.bfloat16, False, None),
        ('bf16[16,256] @ bf16[256,128]', 16, 256, 128, ml_dtypes.bfloat16, False, None),
        ('bf16[16,128] @ bf16[128,256]', 16, 128, 256, ml_dtypes.bfloat16, False, None),
        ('int8[32,128] @ int8[128,128]', 32, 128, 128, np.int8, False, None),
    )
    for name, m, k, n, dtype, transposed, precision in cases:
        lhs = rng.integers(-8, 8, (m, k)).astype(dtype)
        rhs_math = rng.integers(-8, 8, (k, n)).astype(dtype)
        if dtype == np.float32:
            # 带非零低位尾数的 f32，区分“按 f32 计算”与“先舍入成 bf16 再计算”。
            lhs = (lhs + rng.uniform(-0.5, 0.5, (m, k))).astype(np.float32)
            rhs_math = (rhs_math + rng.uniform(-0.5, 0.5, (k, n))).astype(np.float32)
        rhs = rhs_math.T.copy() if transposed else rhs_math
        out_dtype = jnp.int32 if dtype == np.int8 else jnp.float32

        def body(lhs_vmem, rhs_vmem, o_vmem) -> None:
            right = rhs_vmem[...].T if transposed else rhs_vmem[...]
            o_vmem[...] = jnp.dot(lhs_vmem[...], right, preferred_element_type=out_dtype, precision=precision)

        try:
            compiled = mxu_common.compile_kernel(body, jnp.asarray(lhs), jnp.asarray(rhs), jax.ShapeDtypeStruct((m, n), out_dtype))
        except Exception as error:
            print(f'## {name}：编译失败')
            print(str(error).splitlines()[0][:300])
            print()
            continue
        result = np.asarray(compiled(jnp.asarray(lhs), jnp.asarray(rhs)))
        exact = lhs.astype(np.float64) @ rhs_math.astype(np.float64)
        print(f'## {name}')
        if dtype == np.float32:
            # 误差以结果的最大绝对值归一化；分别与三种参考比较，判断两侧输入在哪里被舍入。
            bf16 = lambda a: a.astype(ml_dtypes.bfloat16).astype(np.float64)
            references = (
                ('f64 精确结果', exact),
                ('只有 RHS 先舍入成 bf16', lhs.astype(np.float64) @ bf16(rhs_math)),
                ('两侧都先舍入成 bf16', bf16(lhs) @ bf16(rhs_math)),
            )
            scale = np.max(np.abs(exact))
            for label, reference in references:
                print(f'  与“{label}”的最大相对误差：{np.max(np.abs(result - reference)) / scale:.1e}')
        else:
            print(f'  与精确结果逐元素一致：{bool(np.array_equal(result, exact.astype(result.dtype)))}')
        print(mxu_common.summary(tpuasm_tools.kernel_listing(compiled)))
        print()

if __name__ == '__main__':
    main()
