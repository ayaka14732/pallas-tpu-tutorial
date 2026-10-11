"""本节各实验共用：用循环移位与选择合成的 sublane gather。"""
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

def sublane_gather_by_rotation(x: jax.Array, indices: jax.Array) -> jax.Array:
    """用 8 次“比较并选择”合成 sublane gather：第 k 轮的候选是 x 向上循环移动 k 行的结果。"""
    sublane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 0)
    result = jnp.zeros_like(x)
    rotated = x
    for k in range(8):
        # 此时 rotated[s, l] = x[(s + k) % 8, l]。
        result = jnp.where(indices == (sublane + k) % 8, rotated, result)
        rotated = pltpu.roll(rotated, 7, axis=0)
    return result
