# TPU 硬件架构研究笔记

## 芯片级架构

每个 TPU 芯片包含多个 TensorCore 和（在较新代次中）SparseCore。

### TensorCore

- **VPU (Vector Processing Unit)**：128 lanes 宽的 SIMD 引擎
- **MXU (Matrix Multiply Unit)**：128×128 脉动阵列，每周期产出一个 (8, 128) 的结果 tile
- **VMEM**：向量内存，16MB (v4) 到 192MB (v7)
- **SMEM**：标量内存，少量，用于索引和控制
- **执行模型**：单线程顺序执行（不是 GPU 的 SIMT）

### SparseCore（v5e+ 引入）

- **Scalar Subcore**：标量处理，有自己的 SMEM
- **Vector Subcore 0..N**：各自有独立的 VMEM（256-512KB per subcore）
- **VMEM_SHARED**：所有 subcore 共享的内存区域
- Scalar 和 Vector Subcore 并行执行，通过信号量同步

### 代次规格

| 代次 | TensorCore VMEM | MXU 大小 | HBM | SparseCore |
| :--- | :--- | :--- | :--- | :--- |
| v4 | 16 MB | 128×128 | 32 GB HBM2e | 无 |
| v5p | 48 MB | 128×128 | 95 GB HBM2e | 无 |
| v5e | 32 MB | 128×128 | 16 GB HBM2e | 有 |
| v6e | 64 MB | 128×128 | 32 GB HBM2e | 有 |
| v7 | 192 MB | 128×128 | 192 GB HBM3e | 有 |

### Tile 对齐约束

VMEM 中所有数据必须对齐到原生 tile：
- float32/int32: (8, 128) = 8 sublanes × 128 lanes
- bfloat16/float16: (16, 128)
- int8: (32, 128)

**不能在 VMEM 中存储标量。** 最小分配单位是一个 tile。

### 内存层级

```
HBM (32-192 GB, ~1-2 TB/s)
  ↕ DMA
VMEM (16-192 MB, ~数十 TB/s 内部带宽)
  ↕ load/store
VREG (寄存器文件，128 lanes × 8 sublanes)
  ↕
VPU / MXU (计算)
```

### DMA 特性

- DMA 引擎独立于计算引擎，可以并行工作
- 这是流水线（pipelining）的基础：DMA 搬运下一块数据的同时，计算引擎处理当前数据
- DMA 粒度：整个 tile 或更大的连续块
- 支持 strided access（非连续内存访问）

## 与 GPU 的核心差异

| 维度 | GPU (CUDA) | TPU (Pallas) |
| :--- | :--- | :--- |
| 执行模型 | SIMT：数千线程并行 | 单线程顺序 + SIMD |
| 内存管理 | 程序员管理 shared memory | 编译器/框架管理 VMEM（或手动 DMA） |
| 同步 | `__syncthreads()`, atomics | 信号量（跨 core），无需线程内同步 |
| 矩阵乘法 | Tensor Core (wmma) | MXU（脉动阵列，更大） |
| 带宽隐藏 | 大量线程切换隐藏延迟 | 软件流水线（双缓冲）隐藏 DMA 延迟 |
| 编程粒度 | thread/warp/block | grid iteration（一次处理一个 tile） |

## 关键设计模式

### 1. Grid + BlockSpec（自动 DMA）
最简单的模式。编译器根据 BlockSpec 自动生成 DMA 代码。

### 2. 手动 DMA + 双缓冲
性能关键路径。手动控制 DMA 时序，实现计算和传输重叠。

### 3. emit_pipeline（推荐的流水线方式）
介于自动和手动之间。声明式地描述流水线，编译器生成双缓冲代码。

### 4. Scratch Buffer（跨迭代状态）
利用顺序执行模型，在 grid 迭代之间维护状态（如归约的累加器）。

### 5. Scalar Prefetch
用 SMEM 存储索引信息，实现动态/稀疏访问模式。

### 6. MPMD（Scalar + Vector Subcore 协作）
SparseCore 上的高级模式：Scalar Subcore 做 DMA 调度，Vector Subcore 做计算。
