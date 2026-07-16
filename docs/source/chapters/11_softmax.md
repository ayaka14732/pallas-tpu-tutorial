# 第 11 章：Softmax

## 算子分析

Softmax 是注意力机制的核心组件：

```
softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

减去 max(x) 是为了数值稳定性——防止 exp 溢出。

**算术强度分析**（对长度为 N 的向量）：
- 第一遍：求 max（N 次比较）
- 第二遍：减 max + exp + 求和（3N FLOPs）
- 第三遍：除以 sum（N 次除法）
- 总计 ≈ 5N FLOPs，内存访问 ≈ 4N bytes（读）+ 4N bytes（写）
- AI ≈ 0.6 FLOPs/byte → **极端 memory-bound**

## 朴素实现（两遍扫描）

如果整行能装入 VMEM，直接实现：

```python
def softmax_kernel(x_ref, out_ref):
    x = x_ref[...].astype(jnp.float32)

    # Pass 1: max
    m = jnp.max(x, axis=-1, keepdims=True)

    # Pass 2: exp, sum, normalize
    exp_x = jnp.exp(x - m)
    l = jnp.sum(exp_x, axis=-1, keepdims=True)
    out_ref[...] = (exp_x * pl.reciprocal(l, approx=True)).astype(out_ref.dtype)
```

## 在线 Softmax（Online Softmax）

当序列长度太大无法一次装入 VMEM 时，需要在线 Softmax 算法（Milakov et al., 2018）。这也是 FlashAttention 的核心技术。

在线 Softmax 维护两个 running state：
- `m`：当前已见数据的最大值
- `l`：当前已见数据的指数和（相对于当前 m）

**更新规则**（处理新块 B 时）：
```
m_new = max(m_old, max(B))
correction = exp(m_old - m_new)
l_new = l_old * correction + sum(exp(B - m_new))
```

关键洞察：当 m 更新时，之前所有的 exp 值都需要乘以修正因子 `exp(m_old - m_new)`。

## 在线 Softmax 的 Pallas 实现

```python
def online_softmax_kernel(x_ref, out_ref, m_scratch, l_scratch):
    BLOCK = 128
    seq_len = x_ref.shape[0]
    num_blocks = seq_len // BLOCK

    # 初始化 running state
    m_scratch[...] = jnp.full((), -jnp.inf, dtype=jnp.float32)
    l_scratch[...] = jnp.zeros((), dtype=jnp.float32)

    # Pass 1: 计算全局 max 和 sum（在线）
    @pl.loop(0, num_blocks)
    def _(i):
        chunk = x_ref[i*BLOCK:(i+1)*BLOCK].astype(jnp.float32)
        m_old = m_scratch[...]
        m_new = jnp.maximum(m_old, jnp.max(chunk))
        correction = jnp.exp(m_old - m_new)
        l_scratch[...] = l_scratch[...] * correction + jnp.sum(jnp.exp(chunk - m_new))
        m_scratch[...] = m_new

    # Pass 2: 归一化
    m_final = m_scratch[...]
    inv_l = pl.reciprocal(l_scratch[...], approx=True)

    @pl.loop(0, num_blocks)
    def _(i):
        chunk = x_ref[i*BLOCK:(i+1)*BLOCK].astype(jnp.float32)
        out_ref[i*BLOCK:(i+1)*BLOCK] = (
            jnp.exp(chunk - m_final) * inv_l
        ).astype(out_ref.dtype)
```

## 在线 Softmax 在 FlashAttention 中的应用

FlashAttention 的核心思想：不需要先计算完整的 attention score 矩阵再做 softmax，而是在分块计算 QK^T 的同时，用在线 Softmax 逐步更新输出。

```python
# FlashAttention 核心循环（伪代码）
m = -inf
l = 0
o = 0

for k_block in range(num_kv_blocks):
    s = q @ k_block.T           # 局部 attention score
    m_new = max(m, max(s))
    correction = exp(m - m_new)

    o = o * correction          # 修正之前的输出
    p = exp(s - m_new)
    o = o + p @ v_block         # 累加当前块贡献

    l = l * correction + sum(p)
    m = m_new

o = o / l  # 最终归一化
```

这个模式在第 11 章（FlashAttention）中会完整实现。

## TPU 上的 exp 性能

`jnp.exp` 在 TPU 上由 VPU 的超越函数单元执行，吞吐量远低于简单算术运算。

优化策略：
1. **减少 exp 调用次数**：在线 Softmax 中每个元素只需一次 exp
2. **使用近似**：对于不需要高精度的场景，可以使用多项式近似
3. **与 MXU 操作交错**：在 FlashAttention 中，exp 计算（VPU）可以与 V 矩阵乘法（MXU）重叠

## 数值稳定性

在 bf16 下：
1. **始终在 fp32 下计算 exp 和 sum**：避免 bf16 精度损失
2. **减去 max 是必须的**：防止 exp 上溢（bf16 最大值约 3.4e38）
3. **最终结果可以转回 bf16**：softmax 输出范围是 [0, 1]，bf16 足够

```python
x = x_ref[...].astype(jnp.float32)  # 提升到 fp32
m = jnp.max(x)
exp_x = jnp.exp(x - m)
l = jnp.sum(exp_x)
out_ref[...] = (exp_x * pl.reciprocal(l, approx=True)).astype(jnp.bfloat16)
```

## 与 GPU 实现的对比

| 维度 | GPU (CUDA) | TPU (Pallas) |
| :--- | :--- | :--- |
| 归约 | Warp shuffle + block reduce | 向量寄存器内归约 |
| 在线算法 | FlashAttention (Tri Dao) | 相同算法，不同实现 |
| 内存层级 | SRAM → Registers | VMEM → VREGs |
| 并行度 | 多 warp 并行处理不同行 | 单核顺序处理 + 流水线 |
| exp 性能 | SFU（Special Function Unit）| VPU 超越函数单元 |
