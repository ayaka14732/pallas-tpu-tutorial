# 第 7 章：性能分析与调优

## Roofline 模型

Roofline 模型将 kernel 分为两类：

- **Compute-bound**：计算时间 > 内存传输时间。瓶颈在 MXU/VPU 算力。
- **Memory-bound**：内存传输时间 > 计算时间。瓶颈在 HBM 带宽。

**算术强度（Arithmetic Intensity）** = FLOPs / Bytes transferred

对于 TPU v5e（约 197 TFLOPS bf16，约 1.6 TB/s HBM 带宽）：
- 平衡点 = 197T / 1.6T ≈ 123 FLOPs/byte
- 矩阵乘法 (M=N=K=4096)：AI ≈ 682 → **compute-bound**
- 逐元素操作（如 RMSNorm）：AI ≈ 0.25 → **memory-bound**
- FlashAttention：介于两者之间，取决于序列长度和 head dim

## JAX Profiler

```python
import jax

# 预热（触发 JIT 编译）
z = jax.jit(my_pallas_fn)(inputs)
z.block_until_ready()

# 捕获 Trace
with jax.profiler.trace("/tmp/jax-trace"):
    for _ in range(10):
        z = jax.jit(my_pallas_fn)(inputs)
        z.block_until_ready()
```

查看：
```bash
pip install tensorboard-plugin-profile
tensorboard --logdir=/tmp/jax-trace
```

## TPU Profile 中的关键指标

1. **MXU Utilization**：MXU 实际执行乘法的时间占比。理想值 > 80%。
2. **Memory Bandwidth Utilization**：HBM 带宽利用率。
3. **Infeed/Outfeed Stalls**：等待数据传入/传出的时间。
4. **Idle Time**：计算单元空闲时间。

## CostEstimate

`pl.CostEstimate` 向编译器提供 kernel 的预期成本信息：

```python
cost = pl.CostEstimate(
    flops=2 * M * N * K,
    transcendentals=0,
    bytes_accessed=2*M*K + 2*K*N + 4*M*N,
)

pl.pallas_call(kernel_fn, ..., cost_estimate=cost)
```

## Interpret 模式

在性能分析之前，先用 interpret 模式验证正确性：

```python
result = pl.pallas_call(
    my_kernel,
    ...,
    interpret=True,  # CPU 上模拟执行
)(inputs)
```

Interpret 模式下可以使用 `pdb`、`print()` 等标准调试工具。

## 常见性能问题诊断

### MXU 利用率低

**原因**：
- Block size 太小（< 128），MXU 无法充分流水线化
- 矩阵维度不是 128 的倍数，大量 padding
- VPU 操作阻塞了 MXU

**解决**：
- 增大 block size（BM, BK, BN ≥ 128）
- 确保维度对齐到 128
- 将 VPU 操作与 MXU 操作交错执行

### Memory-bound

**原因**：
- 算术强度太低
- Block size 太小，DMA 启动开销大
- 流水线未生效

**解决**：
- 算子融合
- 增大 block size
- 确保流水线正确工作（检查 emit_pipeline 配置）

### VMEM OOM

**原因**：
- Block size 太大，双缓冲超出 VMEM 容量
- 中间变量太多

**解决**：
- 减小 block size
- 使用 `pltpu.CompilerParams(vmem_limit_bytes=...)` 调试
- 减少同时存活的中间变量

### 寄存器溢出（Register Spill）

**原因**：
- 最后两维有 singleton dimension，导致 padding 到 8×128
- 中间变量太多，VREG 不够用

**解决**：
- 避免最后两维的 singleton dimension
- 减少同时存活的中间变量
- 重新组织计算顺序

## CompilerParams 调优选项

```python
pltpu.CompilerParams(
    vmem_limit_bytes=8 * 1024 * 1024,  # 限制 VMEM（调试 OOM）
    opt_level=3,                        # 优化级别（0-3）
    allow_input_fusion=True,            # 允许融合输入 DMA
    disable_bounds_checks=True,         # 禁用边界检查（生产环境）
    disable_semaphore_checks=True,      # 禁用信号量检查（生产环境）
    internal_scratch_in_bytes=0,        # 内部 scratch 大小
)
```

## Block Size 调优策略

1. **从大开始**：选择能装入 VMEM 的最大 block size
2. **检查对齐**：确保所有维度满足 MXU 对齐要求（8/128 的倍数）
3. **验证流水线**：确认计算时间 ≥ DMA 时间
4. **Profile**：用 JAX profiler 验证 MXU 利用率和 HBM 带宽
5. **迭代**：根据 profile 结果调整

## 算子融合

Memory-bound 操作的主要优化手段是**算子融合**——将多个操作合并为一个 kernel，减少 HBM 读写次数。

例如 `LayerNorm = mean → subtract → square → mean → rsqrt → multiply → add`，如果每步都读写 HBM，带宽浪费极大。融合为一个 Pallas kernel 后，数据只需从 HBM 读一次、写一次。

这也是为什么 RMSNorm、Softmax 等操作值得手写 Pallas kernel——XLA 的自动融合不一定能达到最优。

## 与 GPU Profiling 的对比

| 工具 | GPU | TPU |
| :--- | :--- | :--- |
| 硬件 profiler | Nsight Compute | TPU Profiler (TensorBoard) |
| 关键指标 | SM Occupancy, Memory Throughput | MXU Utilization, HBM BW |
| 瓶颈分析 | Roofline (同) | Roofline (同) |
| 调优手段 | 调 block size, 调 occupancy | 调 block size, 调流水线 |
| 编译器辅助 | 有限（PTX 级别）| 强（Mosaic 自动优化）|
