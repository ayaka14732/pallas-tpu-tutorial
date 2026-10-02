"""本节实验共用：转置的实验条件，以及按 XLU 队列统计清单中的提交与取回。"""
import re

import jax.numpy as jnp

# (名称, dtype, shape, 同时转置的矩阵个数)
CASES = (
    ('i32[128,128]', jnp.int32, (128, 128), 1),
    ('2 个 i32[128,128]', jnp.int32, (128, 128), 2),
    ('3 个 i32[128,128]', jnp.int32, (128, 128), 3),
    ('bf16[128,128]', jnp.bfloat16, (128, 128), 1),
    ('f32[256,256]', jnp.float32, (256, 256), 1),
    ('f32[129,129]', jnp.float32, (129, 129), 1),
    ('bf16[129,129]', jnp.bfloat16, (129, 129), 1),
)

def summary(listing: str) -> str:
    """统计 load/store、各 XLU 的提交（按目的队列 trf0/trf1）与取回（按来源队列），以及其他向量指令。"""
    counts: dict[str, int] = {}
    for match in re.finditer(r'(vld|vst|vx0|vx1|vr0|vr1|va0|va1): (?:@!?p[0-9]+ )?([a-z][\w.]*) ?([^;}#]*)', listing):
        slot, mnemonic, operands = match[1], match[2], match[3]
        if mnemonic.startswith(('vxpose', 'vsupp')):
            key = f'提交到 {operands.split(",")[0].strip()}（{mnemonic}）'
        elif mnemonic == 'vpop.8x128':
            key = f'从 {operands.split(",")[1].strip()} 取回'
        else:
            key = mnemonic
        counts[key] = counts.get(key, 0) + 1
    return '\n'.join(f'  {key}：{value}' for key, value in sorted(counts.items()))
