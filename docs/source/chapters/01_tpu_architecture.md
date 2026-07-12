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

以下规格来自 JAX 源码中的 `jax/_src/pallas/mosaic/tpu_info.py`。需要特别注意：`TpuInfo` 中的 VMEM、SMEM、算力和带宽字段是**按 TensorCore 记录**的；下表中 HBM、带宽和峰值算力为了便于和公开芯片规格对齐，按物理 TensorCore 数量换算为**每芯片总量**。GB/TB 使用十进制单位，MiB/KiB 使用二进制单位。

**芯片与内存**

| TPU | 物理 TensorCore 数 / 芯片 | Lite 芯片 | 支持 Megacore | HBM / 芯片 | HBM 带宽 / 芯片 | VMEM / TensorCore | SMEM / TensorCore |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| v2 | 2 | 否 | 否 | 16 GB | 0.716 TB/s | 16 MiB | 16 KiB |
| v3 | 2 | 否 | 否 | 34.4 GB | 0.825 TB/s | 16 MiB | 16 KiB |
| v4i | 1 | 是 | 否 | 8.59 GB | 0.614 TB/s | 16 MiB | 1 MiB |
| v4 | 2 | 否 | 是 | 34.4 GB | 1.23 TB/s | 16 MiB | 1 MiB |
| v5e | 1 | 是 | 否 | 17.2 GB | 0.820 TB/s | 128 MiB | 1 MiB |
| v5p | 2 | 否 | 是 | 103 GB | 2.46 TB/s | 64 MiB | 1 MiB |
| v6e | 1 | 是 | 否 | 34.4 GB | 1.64 TB/s | 128 MiB | 1 MiB |
| 7x | 2 | 否 | 否 | 206 GB | 7.40 TB/s | 64 MiB | 1 MiB |
| 8i | 2 | 否 | 否 | 309 GB | 8.60 TB/s | 192 MiB | 1 MiB |

TPU v4 和 v4i 还具有 CMEM 这一内存空间，但在其他代次（包括更新代次）的 TPU 中没有，因此本教程不涉及 CMEM。

从第 7 代开始，TPU 的芯片代号改用不带 `v` 前缀的格式。例如，第 4 代写作 `TPU v4`，而第 7 代写作 `tpu7x`。

**TensorCore 计算参数**

| TPU | sublane 数 | lane 数 | MXU 列宽 | MXU 数 / TensorCore | 累加器数 / MXU | BF16 峰值 / 芯片 | INT8 峰值 / 芯片 | FP8 峰值 / 芯片 | INT4 峰值 / 芯片 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| v2 | 8 | 128 | 128 | 1 | 0 | 46 TFLOPS | - | - | - |
| v3 | 8 | 128 | 128 | 2 | 0 | 140 TFLOPS | - | - | - |
| v4i | 8 | 128 | 128 | 4 | 0 | 137 TFLOPS | - | - | - |
| v4 | 8 | 128 | 128 | 4 | 0 | 275 TFLOPS | - | - | - |
| v5e | 8 | 128 | 128 | 4 | 0 | 197 TFLOPS | 394 TFLOPS | - | 788 TFLOPS |
| v5p | 8 | 128 | 128 | 4 | 0 | 459 TFLOPS | 918 TFLOPS | - | 1.84 PFLOPS |
| v6e | 8 | 128 | 256 | 2 | 0 | 920 TFLOPS | 1.84 PFLOPS | 920 TFLOPS | 3.68 PFLOPS |
| 7x | 8 | 128 | 256 | 2 | 128 | 2.31 PFLOPS | - | 4.60 PFLOPS | - |
| 8i | 8 | 128 | 256 | 2 | 256 | 1.101 PFLOPS | - | 8.808 PFLOPS | - |

**SparseCore 参数**

| TPU | SparseCore 数 / 芯片 | Vector Subcore 数 / SparseCore | SparseCore lane 数 | SparseCore VMEM / Vector Subcore | SparseCore DMA 粒度 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| v2 | - | - | - | - | - |
| v3 | - | - | - | - | - |
| v4i | - | - | - | - | - |
| v4 | - | - | - | - | - |
| v5e | - | - | - | - | - |
| v5p | 4 | 16 | 8 | 512 KiB | 32 B |
| v6e | 2 | 16 | 8 | 256 KiB | 32 B |
| 7x | 2 | 16 | 16 | 512 KiB | 32 B |
| 8i | 1 | 4 | 16 | 512 KiB | 64 B |

`tpu_info.py` 还提供了 `get_sublane_tiling(dtype)` 和 `infer_tiling(...)`。对本教程最重要的结论是：TensorCore 的默认 compact tiling 仍围绕 `(8, 128)` 展开；从 generation 6 开始 MXU column size 变为 256；从 generation 7 开始，低 bitwidth dtype 的 sublane tiling 会默认使用更大的 second-minor tiling。

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
