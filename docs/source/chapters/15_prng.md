# 第 15 章：TPU 上的随机数生成 (PRNG)

## 动机

随机数在 LLM 推理和训练中有多种用途：
- **Dropout**：训练时随机丢弃神经元
- **采样**：推理时从 logits 中采样 token（top-p, top-k）
- **随机舍入**：bf16 训练中的 stochastic rounding
- **稀疏注意力**：随机选择注意力块

## TPU PRNG 硬件

TPU 有专用的 PRNG 硬件单元，可以高吞吐量地生成伪随机数。与 GPU 不同，TPU 的 PRNG 是**有状态的**——硬件维护一个内部状态，每次调用自动推进。

## 基本用法

### 设置种子

```python
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def kernel_with_prng(x_ref, out_ref):
    # 设置 PRNG 种子
    pltpu.prng_seed(42)

    # 生成随机 bits
    random_bits = pltpu.prng_random_bits(x_ref.shape)

    # 转换为 [0, 1) 的均匀分布
    # random_bits 是 uint32，需要手动转换
    uniform = random_bits.astype(jnp.float32) / jnp.float32(2**32)

    out_ref[...] = x_ref[...] * (uniform > 0.5).astype(x_ref.dtype)
```

### 有状态随机数 API

```python
def dropout_kernel(x_ref, out_ref, rate=0.1):
    pltpu.prng_seed(seed)

    # 直接使用有状态 API
    mask = pltpu.stateful_bernoulli(x_ref.shape, prob=1.0 - rate)
    scale = 1.0 / (1.0 - rate)

    out_ref[...] = jnp.where(mask, x_ref[...] * scale, 0.0)
```

## 与 JAX Key 的互操作

```python
def kernel(x_ref, key_ref, out_ref):
    # 将 JAX PRNG key 转换为 Pallas 可用的种子
    pallas_key = pltpu.to_pallas_key(key_ref[...])
    pltpu.prng_seed(pallas_key)

    # 生成随机数
    bits = pltpu.prng_random_bits(x_ref.shape)
    ...
```

## Stochastic Rounding

在混合精度训练中，将 fp32 梯度累加到 bf16 参数时，标准的 round-to-nearest 会引入系统性偏差。Stochastic rounding 通过随机选择向上或向下舍入来消除偏差：

```python
def stochastic_round_kernel(x_fp32_ref, out_bf16_ref):
    pltpu.prng_seed(seed)

    x = x_fp32_ref[...]
    # stochastic_round 自动使用硬件 PRNG
    out_bf16_ref[...] = pltpu.stochastic_round(x, pltpu.prng_random_bits(x.shape))
```

## 与 GPU 的对比

| 维度 | GPU (cuRAND) | TPU (Pallas PRNG) |
| :--- | :--- | :--- |
| 状态管理 | 无状态（counter-based）| 有状态（硬件维护）|
| 并行性 | 每个线程独立的 state | 全局 state，硬件并行生成 |
| 典型算法 | Philox | 硬件实现（不公开）|
| 性能 | 需要显式管理 offset | 自动推进，低开销 |

## 注意事项

1. **种子必须在 kernel 内设置**：不能依赖外部状态
2. **确定性**：相同种子 + 相同 shape → 相同结果
3. **跨核一致性**：Megacore 模式下，不同核心需要不同种子以避免相关性
4. **性能**：PRNG 生成本身很快，但后续的比较和 mask 操作是 VPU 操作
