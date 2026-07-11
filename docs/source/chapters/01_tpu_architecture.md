# 第 1 章：TPU 硬件架构

## TPU 不是 GPU

TPU 和 GPU 在设计哲学上有根本分歧。GPU 是一台拥有数千个并发执行单元的**延迟隐藏机器**——通过海量线程的快速切换来掩盖内存访问延迟。TPU 则是一台拥有超宽向量寄存器的**吞吐量优先的顺序机器**——通过软件流水线和异步执行单元来实现计算与传输的重叠。

这个区别直接决定了 kernel 的编写方式：

| 维度 | GPU (CUDA) | TPU (Pallas) |
| :--- | :--- | :--- |
| 并行模型 | 数万线程并发，硬件调度器自动切换 warp | 单线程顺序执行，软件显式编排流水线 |
| 延迟隐藏 | 靠线程数量（occupancy）| 靠 DMA/MXU 异步执行与计算重叠 |
| 内存访问 | 计算指令可直接发起全局内存读写 | 计算指令**不能**直接访问 HBM，必须经过 DMA |
| 同步机制 | `__syncthreads()`、原子操作 | 信号量（Semaphore）|
| 矩阵乘法 | Tensor Core（需要特定数据布局）| MXU 脉动阵列（128x128，自动流水线）|
| 典型块大小 | 16-256（受限于共享内存 ~160KB）| 512-2048+（VMEM 16MB+）|

## 芯片级架构：TensorCore 与 SparseCore

一块 TPU 芯片包含两类处理单元：**TensorCore** 和 **SparseCore**。

```
┌─────────────────────────────────────────────────────────────┐
│                        TPU Chip                              │
│                                                             │
│  ┌─────────────────────┐    ┌─────────────────────────────┐│
│  │     TensorCore(s)   │    │         SparseCore(s)       ││
│  │  ┌───────┐ ┌──────┐│    │  ┌──────────┐  ┌─────────┐ ││
│  │  │  MXU  │ │ VPU  ││    │  │  Scalar  │  │ Vector  │ ││
│  │  │128x128│ │      ││    │  │ Subcore  │  │Subcore 0│ ││
│  │  └───────┘ └──────┘│    │  │  (SMEM)  │  │ (VMEM)  │ ││
│  │  ┌───────┐ ┌──────┐│    │  └──────────┘  ├─────────┤ ││
│  │  │  XLU  │ │Scalar││    │                 │ Vector  │ ││
│  │  │       │ │ Unit ││    │  ┌──────────┐  │Subcore 1│ ││
│  │  └───────┘ └──────┘│    │  │  Shared  │  │ (VMEM)  │ ││
│  │  ┌─────────────────┐│    │  │   VMEM   │  ├─────────┤ ││
│  │  │      VMEM       ││    │  └──────────┘  │   ...   │ ││
│  │  │    (16MB+)      ││    │                 │Subcore N│ ││
│  │  └─────────────────┘│    │                 │ (VMEM)  │ ││
│  └─────────────────────┘    │                 └─────────┘ ││
│                              └─────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │                        HBM                               ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### TensorCore

TensorCore 负责**密集计算**（矩阵乘法、向量运算）。这是大多数 Pallas kernel 运行的地方。

TensorCore 内部包含：
- **MXU**（Matrix Multiply Unit）：128×128 脉动阵列
- **VPU**（Vector Processing Unit）：逐元素运算
- **XLU**（Cross-Lane Unit）：跨 lane 操作（转置、归约）
- **Scalar Unit**：标量运算、控制流、DMA 发起
- **VMEM**：向量内存（16MB+），所有计算操作数必须在此

### SparseCore

SparseCore 负责**稀疏/随机访问操作**。它是 TPU 的独特设计，GPU 上没有对应物。

SparseCore 内部包含：
- **Scalar Subcore**：标量运算、动态索引、DMA 发起（有自己的 SMEM）
- **Vector Subcore 0..N**：SIMD 向量运算（每个有自己的 VMEM）
- **Shared VMEM**：所有 Vector Subcore 共享的内存空间

SparseCore 擅长的操作：
- Gather / Scatter（按索引取数据）
- 排序、去重、直方图
- Ragged 操作（不规则长度的批处理）
- 中低计算量 + 频繁数据通信的场景

在 Pallas 中：
- TensorCore kernel 使用 `pltpu.TensorCoreMesh`
- SparseCore kernel 使用 `plsc.ScalarSubcoreMesh` 或 `plsc.VectorSubcoreMesh`
- 两者可以**同时执行**（XLA 自动调度），实现计算重叠

```python
from jax.experimental.pallas import tpu_sc as plsc

# SparseCore 的 gather 比 TensorCore 的 jnp.take 快 4x+
# 因为 SparseCore 硬件专门优化了随机内存访问
```

### TPU 代次规格

| 属性 | TPU v4 | TPU v5p | TPU v6e (Trillium) | TPU 7x (Ironwood) |
| :--- | :--- | :--- | :--- | :--- |
| SparseCores / Chip | 4 | 4 | 2 | 2 (4 physical) |
| Vector Subcores / SparseCore | 16 | 16 | 16 | 16 |
| SC SIMD Width | 8 | 8 | 8(F32)/16(BF16) | 16(F32)/32(BF16) |
| TensorCores / Chip | 2 | 2 | 1 | - |
| HBM 容量 | 32 GB | 96 GB | 32 GB | 192 GB |

可以通过 `pltpu.get_tpu_info()` 在运行时查询硬件信息。

## 内存层级

TPU 的内存层级与 GPU 有本质不同。GPU 的共享内存（Shared Memory）通常只有 48-164KB，而 TPU 的 VMEM 有 **16MB 以上**。这意味着 TPU kernel 可以使用远大于 GPU 的 block size，从而减少 HBM 访问次数。

### TensorCore 内存空间

| 内存空间 | Pallas 常量 | 容量 | 访问方式 | 用途 |
| :--- | :--- | :--- | :--- | :--- |
| **HBM** | `pltpu.HBM` | 32-192GB | 仅通过 DMA | 主存储，存放完整张量 |
| **VMEM** | `pltpu.VMEM` | 16MB+ | VPU/MXU 直接读写 | 所有计算操作数 |
| **SMEM** | `pltpu.SMEM` | 数百 KB | Scalar Unit 直接读写 | 控制流、动态索引 |
| **VREG** | - | 8×128 (32-bit) | 计算单元直接操作 | 实际执行计算 |

### SparseCore 内存空间

| 内存空间 | Pallas 常量 | 用途 |
| :--- | :--- | :--- |
| **VMEM** | `pltpu.VMEM` | 每个 Vector Subcore 的本地内存 |
| **VMEM_SHARED** | `pltpu.VMEM_SHARED` | 所有 Vector Subcore 共享（也叫 SPMEM）|
| **SMEM** | `pltpu.SMEM` | Scalar Subcore 的内存 |

关键约束：

1. **HBM 不可直接计算**。所有数据必须先 DMA 到 VMEM，再加载到 VREG 进行计算。这与 GPU 不同——CUDA kernel 可以直接从 Global Memory 读取（虽然慢）。
2. **VMEM 与 HBM 之间的 DMA 以 4KiB 为粒度**。小于 4KiB 的传输会浪费带宽。
3. **SMEM 支持标量随机访问**。这使得动态索引（如 page table 查找）可以在标量核心上高效完成，而不需要占用向量核心。

## TensorCore 计算单元

**MXU（Matrix Multiply Unit）**：一个 128×128 的脉动阵列（Systolic Array）。每个时钟周期可以完成一次 128×128 的乘加操作。MXU 是 TPU 算力的主要来源——在 v5e 上提供约 197 TFLOPS (bf16)。MXU 支持的精度包括 bf16、fp32（通过累加）、int8。MXU 的输入必须满足特定的 tiling 约束（最内两维对齐到 128 和 8）。

**VPU（Vector Processing Unit）**：处理逐元素运算（加减乘除、激活函数、类型转换等）。VPU 操作的寄存器大小为 8×128（对于 32-bit 值）。VPU 的吞吐量远低于 MXU，因此 kernel 优化的核心原则是：**尽可能让 MXU 保持忙碌，将 VPU 操作隐藏在 MXU 计算的延迟中**。

**XLU（Cross-Lane Unit）**：处理跨 lane 的操作，如转置、排列（permute）、归约（reduce）。XLU 操作通常比 VPU 更昂贵。

## 数组布局与 Tiling

Pallas kernel 中的数组布局直接影响生成代码的质量。TPU 的向量寄存器是 2D 的（8 sublanes × 128 lanes，对于 32-bit 值），数组的**最后两个维度**会被映射到这个 2D 寄存器上。

这带来以下硬性约束：

1. **最后两维的大小应为 8 和 128 的倍数**。不满足时，编译器会自动 padding，浪费寄存器空间。两个 `(1, 1)` 数组相加的代价与两个 `(8, 128)` 数组相加完全相同。
2. **涉及最后两维的 reshape 可能不被支持**。某些跨 sublane/lane 的 reshape 无法在硬件上高效实现。
3. **最后两维上的 singleton dimension 极其浪费**。一个 `(8, 128, 1, 1)` 数组会被 padding 为 `(8, 128, 8, 128)`，占用 1024 倍的寄存器。
4. **归约（reduction）在最后一维（lane 维）上最高效**。跨 sublane 的归约需要 XLU 参与，代价更高。

对于矩阵乘法，MXU 要求：
- LHS 的收缩维度（contraction dim）位于倒数第二维（sublane 维），大小为 8 的倍数
- RHS 的收缩维度位于倒数第二维，大小为 128 的倍数
- 输出的最后一维为 128 的倍数

## 网格执行顺序

TPU 上的 Pallas grid 默认按**字典序顺序执行**，而非并行。这是与 GPU 最大的编程模型差异之一。

顺序执行带来的优势：
1. **自动 VMEM 复用**：连续的 grid 迭代如果访问相同的输入块，编译器会跳过第二次 DMA 传输。
2. **无竞态条件**：多次迭代可以安全地写入同一个输出位置（如归约操作），无需原子操作。
3. **状态传递**：可以通过 scratch buffer 在迭代之间传递状态（如 online softmax 的 running max）。

通过 `pltpu.CompilerParams(dimension_semantics=["parallel", "arbitrary"])` 可以将某些 grid 维度标记为可并行执行（利用多核）。`"arbitrary"` 表示该维度必须顺序执行。

## 与 GPU 优化思路的对比

在 GPU 上优化 kernel 的核心思路是：提高 occupancy（让更多 warp 同时驻留以隐藏延迟）、合并内存访问（coalesced access）、减少 bank conflict。

在 TPU 上，优化思路完全不同：
1. **最大化 MXU 利用率**：确保矩阵乘法的维度足够大（至少 128×128），让 MXU 的脉动阵列充分流水线化。
2. **重叠计算与传输**：通过软件流水线，让 DMA 传输和 MXU/VPU 计算同时进行。
3. **减少 VPU 瓶颈**：VPU 操作（如 softmax 中的 exp、除法）应尽量与 MXU 操作重叠。
4. **利用大 VMEM**：选择足够大的 block size 以减少 HBM 往返次数。
5. **避免寄存器溢出**：虽然 VMEM 很大，但 VREG 仍然有限。过多的中间变量会导致溢出到 VMEM。
6. **考虑 SparseCore 卸载**：如果有 gather/scatter 或 ragged 操作，考虑将其卸载到 SparseCore，让 TensorCore 专注于密集计算。
