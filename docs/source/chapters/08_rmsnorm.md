# 第 8 章：RMSNorm

## 什么是 RMSNorm

RMSNorm（Root Mean Square Layer Normalization）是 LLaMA、Qwen、Gemma 等现代大模型中替代 LayerNorm 的归一化操作。相比 LayerNorm，RMSNorm 去掉了均值中心化步骤，只保留方差归一化：

```
LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
RMSNorm(x)   = x / sqrt(mean(x^2) + eps) * weight
             = x * rsqrt(mean(x^2) + eps) * weight
```

去掉均值中心化的好处：减少一次归约操作（不需要计算 mean），同时实验表明对模型质量几乎没有影响。

对于一个形状为 `(batch_size, seq_len, hidden_dim)` 的输入张量，RMSNorm 沿 `hidden_dim` 维度做归约。`weight` 是一个可学习的 `(hidden_dim,)` 向量。

## 算术强度分析

假设 bf16 输入，hidden_dim = d，处理一行：

| 操作 | 数据量 / 计算量 |
| :--- | :--- |
| 读取 x | 2d bytes |
| 读取 weight | 2d bytes |
| 写入 output | 2d bytes |
| 总内存访问 | 6d bytes |
| x^2 | d 次乘法 |
| sum(x^2) | d 次加法 |
| mean = sum / d | 1 次除法 |
| rsqrt(mean + eps) | 1 次 rsqrt |
| x * rsqrt * weight | 2d 次乘法 |
| 总计算量 | ~4d FLOPs |

**算术强度 = 4d / 6d ≈ 0.67 FLOPs/byte**

对比 TPU v5e 的 Roofline：
- 计算峰值：197 TFLOPS (bf16)
- HBM 带宽：820 GB/s
- 平衡点：197000 / 820 ≈ 240 FLOPs/byte

RMSNorm 的算术强度（0.67）远低于平衡点（240），因此是**极端 memory-bound** 的操作。优化目标不是减少计算，而是**最大化 HBM 带宽利用率**。

## 第一版：最简单的实现

先写一个能跑的版本，不考虑任何优化：

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import functools

def rmsnorm_kernel_v1(x_ref, weight_ref, out_ref, *, eps: float):
    """最简单的 RMSNorm：每次处理一行"""
    x = x_ref[...].astype(jnp.float32)  # 提升到 fp32 避免精度问题
    w = weight_ref[...].astype(jnp.float32)

    # 计算 mean(x^2)
    mean_sq = jnp.sum(x * x) / x.shape[0]

    # 计算 1/sqrt(mean_sq + eps)
    inv_rms = 1.0 / jnp.sqrt(mean_sq + eps)

    # 归一化并乘以 weight
    out_ref[...] = (x * inv_rms * w).astype(out_ref.dtype)

def rmsnorm_v1(x: jax.Array, weight: jax.Array, eps: float = 1e-6):
    """x: (num_rows, hidden_dim), weight: (hidden_dim,)"""
    num_rows, hidden_dim = x.shape

    return pl.pallas_call(
        functools.partial(rmsnorm_kernel_v1, eps=eps),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[
            pl.BlockSpec((None, hidden_dim), lambda i: (i, 0)),
            pl.BlockSpec((hidden_dim,), lambda i: (0,)),
        ],
        out_specs=pl.BlockSpec((None, hidden_dim), lambda i: (i, 0)),
        grid=(num_rows,),
    )(x, weight)

# 测试
x = jax.random.normal(jax.random.key(0), (32, 4096), dtype=jnp.bfloat16)
w = jnp.ones(4096, dtype=jnp.bfloat16)
result = rmsnorm_v1(x, w)
```

**这个版本的问题：**
1. `1.0 / jnp.sqrt(...)` 在 VPU 上是多周期操作（除法很慢）
2. 每次只处理一行，DMA 启动开销占比大
3. `BlockSpec` 中 `None` 维度的 squeeze 语义需要理解

## BlockSpec 中的 None 维度

```python
pl.BlockSpec((None, hidden_dim), lambda i: (i, 0))
```

这里 `None` 表示该维度被 **squeeze**：
- 原始数组形状：`(num_rows, hidden_dim)`
- block_shape 中 `None` 对应的维度在 Ref 中被去掉
- kernel 收到的 `x_ref` 形状是 `(hidden_dim,)` 而不是 `(1, hidden_dim)`

这避免了 singleton dimension 的性能问题（第 1 章提到的 `(1, 1)` padding 到 `(8, 128)` 的问题）。

## 第二版：使用硬件近似指令

TPU 提供硬件近似倒数平方根指令，精度约 12 bits（对 bf16 输出足够）：

```python
def rmsnorm_kernel_v2(x_ref, weight_ref, out_ref, *, eps: float):
    x = x_ref[...].astype(jnp.float32)
    w = weight_ref[...].astype(jnp.float32)

    mean_sq = jnp.sum(x * x) / x.shape[0]

    # 使用 jax.lax.rsqrt 代替 1.0 / sqrt(...)
    # 在 TPU 上，rsqrt 编译为单周期硬件指令
    inv_rms = jax.lax.rsqrt(mean_sq + eps)

    out_ref[...] = (x * inv_rms * w).astype(out_ref.dtype)
```

**`jax.lax.rsqrt` vs `1.0 / jnp.sqrt()`：**

| 方式 | TPU 指令 | 周期数 | 精度 |
| :--- | :--- | :--- | :--- |
| `1.0 / jnp.sqrt(x)` | sqrt + div（两次 VPU 多周期操作）| ~10+ 周期 | 完全精度 |
| `jax.lax.rsqrt(x)` | 单条 rsqrt 指令 | 1 周期 | ~12 bits |

对于 bf16 输出（只有 7 bits 有效位），12 bits 精度绰绰有余。

## TPU 上的归约性能

TPU 的归约操作有方向性差异，这是由硬件寄存器布局决定的：

```
向量寄存器 (VREG): 8 sublanes × 128 lanes (对于 32-bit 值)

数组最后两维映射到寄存器：
  倒数第二维 → sublane 方向 (8)
  最后一维   → lane 方向 (128)
```

| 归约方向 | 硬件操作 | 相对代价 |
| :--- | :--- | :--- |
| 沿前导维度（非最后两维）| 多个寄存器块逐元素累加 | 最快（纯 VPU）|
| 沿倒数第二维（sublane 维）| XLU 跨 sublane 归约 | 中等 |
| 沿最后一维（lane 维）| XLU 跨 lane 归约 | 最慢 |

对于 RMSNorm，归约维度是 hidden_dim。如果 hidden_dim 是数组的最后一维（通常如此），那么归约需要跨 lane 通信。

但由于 RMSNorm 是 memory-bound 的，归约本身不是瓶颈——即使用最慢的跨 lane 归约，计算时间仍然远小于 DMA 时间。真正的优化方向是减少 HBM 访问。

## 第三版：多行批处理

如果每次只处理一行（hidden_dim = 4096 → 8KB bf16），DMA 的启动开销和 4KiB 对齐粒度会浪费带宽。一次处理多行可以增加每次 DMA 的数据量：

```python
ROWS_PER_BLOCK = 8  # 一次处理 8 行

def rmsnorm_kernel_v3(x_ref, weight_ref, out_ref, *, eps: float):
    """批处理版本：一次处理 ROWS_PER_BLOCK 行"""
    # x_ref: (ROWS_PER_BLOCK, hidden_dim)
    x = x_ref[...].astype(jnp.float32)
    w = weight_ref[...].astype(jnp.float32)

    # 沿 hidden_dim 归约，保留 batch 维度
    # mean_sq: (ROWS_PER_BLOCK,)
    mean_sq = jnp.sum(x * x, axis=-1) / x.shape[-1]
    inv_rms = jax.lax.rsqrt(mean_sq + eps)

    # inv_rms: (ROWS_PER_BLOCK,) → 需要广播到 (ROWS_PER_BLOCK, hidden_dim)
    # 注意：(ROWS_PER_BLOCK, 1) 的 singleton 会被 padding
    # 所以用显式广播而不是 keepdims=True
    out_ref[...] = (x * inv_rms[:, None] * w[None, :]).astype(out_ref.dtype)

def rmsnorm_v3(x: jax.Array, weight: jax.Array, eps: float = 1e-6):
    num_rows, hidden_dim = x.shape
    assert num_rows % ROWS_PER_BLOCK == 0

    return pl.pallas_call(
        functools.partial(rmsnorm_kernel_v3, eps=eps),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[
            pl.BlockSpec((ROWS_PER_BLOCK, hidden_dim), lambda i: (i, 0)),
            pl.BlockSpec((hidden_dim,), lambda i: (0,)),
        ],
        out_specs=pl.BlockSpec((ROWS_PER_BLOCK, hidden_dim), lambda i: (i, 0)),
        grid=(num_rows // ROWS_PER_BLOCK,),
    )(x, weight)

# 测试
x = jax.random.normal(jax.random.key(0), (32, 4096), dtype=jnp.bfloat16)
w = jnp.ones(4096, dtype=jnp.bfloat16)
result = rmsnorm_v3(x, w)
```

**关于 singleton dimension 的陷阱：**

```python
# 危险：keepdims=True 产生 (8, 1) 形状
mean_sq = jnp.mean(x * x, axis=-1, keepdims=True)  # shape: (8, 1)
# 最后一维为 1，会被 padding 到 128，浪费 128 倍寄存器空间

# 安全：先 squeeze 再广播
mean_sq = jnp.mean(x * x, axis=-1)  # shape: (8,)
result = x * inv_rms[:, None]  # 编译器生成高效的广播代码
```

## 第四版：流水线化

对于大 batch（如 seq_len = 2048），可以用 `emit_pipeline` 让 DMA 和计算重叠：

```python
def rmsnorm_pipeline(x: jax.Array, weight: jax.Array, eps: float = 1e-6):
    num_rows, hidden_dim = x.shape
    BLOCK_ROWS = 8

    def kernel(x_hbm_ref, w_hbm_ref, o_hbm_ref):
        def body(x_ref, w_ref, o_ref):
            x = x_ref[...].astype(jnp.float32)
            w = w_ref[...].astype(jnp.float32)
            mean_sq = jnp.sum(x * x, axis=-1) / hidden_dim
            inv_rms = jax.lax.rsqrt(mean_sq + eps)
            o_ref[...] = (x * inv_rms[:, None] * w[None, :]).astype(o_ref.dtype)

        pltpu.emit_pipeline(
            body,
            grid=(num_rows // BLOCK_ROWS,),
            in_specs=[
                pl.BlockSpec((BLOCK_ROWS, hidden_dim), lambda i: (i, 0)),
                pl.BlockSpec((hidden_dim,), lambda i: (0,)),
            ],
            out_specs=[pl.BlockSpec((BLOCK_ROWS, hidden_dim), lambda i: (i, 0))],
        )(x_hbm_ref, w_hbm_ref, o_hbm_ref)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
    )(x, weight)

# 测试
x = jax.random.normal(jax.random.key(0), (2048, 4096), dtype=jnp.bfloat16)
w = jnp.ones(4096, dtype=jnp.bfloat16)
result = rmsnorm_pipeline(x, w)
```

**流水线的效果：**
- 不流水线：总时间 = N × (DMA_load + compute + DMA_store)
- 流水线后：总时间 ≈ N × max(DMA_load, compute, DMA_store)

由于 RMSNorm 是 memory-bound 的，compute << DMA，所以流水线后总时间 ≈ N × DMA 时间，计算完全被隐藏。

## 第五版：weight 的广播优化

注意到 `weight` 在每次迭代中都是相同的。在 `emit_pipeline` 中，如果 `BlockSpec` 的 `index_map` 对所有迭代返回相同的索引，编译器会自动识别并只做一次 DMA：

```python
# weight 的 BlockSpec：index_map 不依赖 i
pl.BlockSpec((hidden_dim,), lambda i: (0,))
# 编译器识别到每次迭代都访问同一块数据 → 只 DMA 一次，后续迭代复用 VMEM 中的副本
```

这是 TPU 顺序执行模型的优势：编译器可以静态分析哪些块在连续迭代中被重复访问。

## 算子融合：RMSNorm + Linear

RMSNorm 通常紧跟一个线性层。如果分开执行：

```
x → [RMSNorm kernel] → y (写回 HBM) → [MatMul kernel] → z
```

中间结果 `y` 需要写回 HBM 再读回来，浪费两次 HBM 访问（写 + 读 = 4d bytes）。

融合后：

```
x → [RMSNorm + MatMul kernel] → z  (y 留在 VMEM，不经过 HBM)
```

在 `emit_pipeline` 中实现融合：

```python
def fused_rmsnorm_linear_kernel(
    x_hbm_ref, w_norm_hbm_ref, w_linear_hbm_ref, o_hbm_ref
):
    def body(x_ref, w_norm_ref, w_linear_ref, o_ref):
        # RMSNorm（结果留在 VMEM）
        x = x_ref[...].astype(jnp.float32)
        w = w_norm_ref[...].astype(jnp.float32)
        mean_sq = jnp.sum(x * x, axis=-1) / x.shape[-1]
        inv_rms = jax.lax.rsqrt(mean_sq + 1e-6)
        normed = x * inv_rms[:, None] * w[None, :]

        # Linear（直接用 VMEM 中的 normed 结果）
        # normed: (BLOCK_ROWS, hidden_dim)
        # w_linear: (hidden_dim, out_dim) — 这里简化
        o_ref[...] = jnp.dot(normed, w_linear_ref[...]).astype(o_ref.dtype)

    pltpu.emit_pipeline(
        body,
        grid=(...),
        in_specs=[...],
        out_specs=[...],
    )(x_hbm_ref, w_norm_hbm_ref, w_linear_hbm_ref, o_hbm_ref)
```

融合的收益：
- 省去 y 的 HBM 写入 + 读取 = 节省 4d bytes/row 的 HBM 带宽
- 对于 hidden_dim = 4096, batch = 2048：节省 2048 × 4096 × 4 = 32MB 的 HBM 流量

## 与 GPU 实现的对比

| 维度 | GPU (CUDA) | TPU (Pallas) |
| :--- | :--- | :--- |
| 归约实现 | Warp shuffle + shared memory block reduce | 向量寄存器内归约（`jnp.sum`，硬件自动）|
| 同步 | `__syncthreads()` 在 block reduce 前后 | 不需要（单线程顺序执行）|
| 内存访问 | 需要 coalesced access 对齐 | DMA 自动处理对齐 |
| Block size | 受限于 shared memory (~160KB) | 受限于 VMEM (16MB+)，可以很大 |
| 向量化 | 需要手动 `float4` 加载 | 自动（8×128 寄存器天然向量化）|
| 融合策略 | 需要手写融合 kernel 或用 Triton | emit_pipeline 中自然组合 |

**GPU 上 RMSNorm 的典型实现：**
- 一个 CUDA block 处理一行
- 每个线程加载若干元素，做局部平方和
- Warp shuffle 做 warp 内归约
- Shared memory 做 block 内归约
- 最终一个线程算出 rsqrt，广播给所有线程
- 所有线程并行做归一化

**TPU 上的实现：**
- 一个 grid 步骤处理若干行
- `jnp.sum` 直接在 VREG 中完成（硬件自动处理跨 lane 归约）
- 不需要手动管理线程同步
- 重点放在 DMA 流水线和算子融合上

## BLOCK_ROWS 的选择

| BLOCK_ROWS | 每次 DMA 数据量 (d=4096, bf16) | 分析 |
| :--- | :--- | :--- |
| 1 | 8KB | 太小，DMA 启动开销占比大 |
| 4 | 32KB | 可以，但仍然偏小 |
| 8 | 64KB | 较好的平衡点 |
| 16 | 128KB | 更好的 DMA 效率 |
| 32 | 256KB | 接近最优 |
| 128 | 1MB | DMA 效率高，但 VMEM 占用大 |

选择原则：
1. 每次 DMA 数据量 >> 4KiB（DMA 粒度），否则浪费带宽
2. BLOCK_ROWS × hidden_dim × dtype_size × 3（输入 + 输出 + 中间结果）< VMEM 容量
3. 足够多的 grid 步骤（≥ 4）让流水线充分展开

对于 hidden_dim = 4096, bf16：
- VMEM = 16MB (v4) → 最大约 BLOCK_ROWS = 16MB / (4096 × 2 × 3) ≈ 682 行
- 实际取 32-64 行即可，留出 VMEM 给双缓冲

## 练习

1. 实现一个 RMSNorm kernel，处理 `(2048, 8192)` 的 bf16 输入，用 `emit_pipeline` 流水线化
2. 比较 `BLOCK_ROWS = 4` 和 `BLOCK_ROWS = 32` 的性能差异（用 JAX profiler）
3. 尝试将 RMSNorm 与一个简单的逐元素操作（如 SiLU 激活）融合
