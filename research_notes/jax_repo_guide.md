# JAX 仓库中 Pallas TPU 相关内容导航

> 本文档帮助接手者快速定位 JAX 仓库中与 Pallas TPU 相关的所有重要文件。

## 官方文档源码

所有文档在 `jax/docs/pallas/` 目录下：

```
docs/pallas/
├── index.rst                    # Pallas 总目录
├── quickstart.md                # 快速入门（通用，非 TPU 特定）
├── grid_blockspec.md            # Grid 和 BlockSpec 详解
├── tpu/
│   ├── index.rst                # TPU 子目录
│   ├── details.rst              # TPU 核心概念（Ref 类型、内存模型）
│   ├── pipelining.md            # 流水线教程
│   ├── matmul.md                # 矩阵乘法教程
│   ├── sparse.rst               # Scalar Prefetch 和稀疏访问
│   ├── distributed.md           # 分布式 kernel（信号量、Remote DMA）
│   ├── core_map.md              # core_map 多核
│   ├── hardware.ipynb           # 硬件参考（notebook）
│   ├── prng.rst                 # 随机数生成
│   └── sparsecore.md            # SparseCore 架构和编程
```

## 源码实现

### 核心框架

```
jax/_src/pallas/
├── core.py                      # BlockSpec, GridSpec 等核心抽象
├── helpers.py                   # pl.loop, pl.when, pl.run_scoped 实现
├── mosaic/
│   ├── core.py                  # TPU 特定：MemorySpace, SemaphoreType, CompilerParams
│   ├── helpers.py               # sync_copy, core_barrier 实现
│   ├── pipeline.py              # emit_pipeline 完整实现（重要！）
│   ├── primitives.py            # make_async_copy, prng, matmul 等原语
│   ├── tpu_info.py              # 各代次 TPU 的硬件规格
│   ├── lowering.py              # Pallas → Mosaic IR 降低
│   └── sc_lowering.py           # SparseCore 降低
```

### 生产级 Kernel 实现

```
jax/experimental/pallas/ops/tpu/
├── flash_attention.py           # FlashAttention TPU 实现
├── paged_attention/
│   └── paged_attention_kernel.py  # Paged Attention（JAX 官方版本）
├── ragged_paged_attention/
│   └── kernel.py                # Ragged Paged Attention（JAX 仓库版本）
└── example_kernel.py            # 示例 kernel
```

### 测试文件（大量可运行示例）

```
tests/pallas/
├── tpu_pallas_test.py           # 主测试（数百个测试用例，覆盖所有基础 API）
├── tpu_pallas_pipeline_test.py  # 流水线测试（matmul 流水线、多级流水线）
├── tpu_pallas_mpmd_test.py      # MPMD 测试（Scalar/Vector Subcore 协作）
├── tpu_sparsecore_pallas_distributed_test.py  # SparseCore 分布式测试
├── tpu_pallas_distributed_test.py  # TensorCore 分布式测试
└── pallas_test.py               # 通用 Pallas 测试
```

## 重要测试用例索引

以下测试用例是理解特定功能的最佳参考：

### 基础 BlockSpec 和 Grid
- `tpu_pallas_test.py::PallasCallTest` - 基础 pallas_call 用法

### 流水线
- `tpu_pallas_pipeline_test.py::PallasPipelineTest::test_pipeline_matmul` - 流水线矩阵乘法
- `tpu_pallas_pipeline_test.py::PallasPipelineTest::test_double_buffered_pipeline` - 双缓冲

### 手动 DMA
- `tpu_pallas_test.py` 中搜索 `make_async_copy` - 异步 DMA 模式
- `tpu_pallas_test.py` 中搜索 `sync_copy` - 同步 DMA 模式

### 信号量和分布式
- `tpu_pallas_distributed_test.py` - 跨设备通信
- `tpu_pallas_mpmd_test.py::test_parallel_subkernels_semaphores` (约 line 755) - 信号量同步
- `tpu_sparsecore_pallas_distributed_test.py` (约 line 190-300) - SparseCore reduce-scatter

### SparseCore
- `tpu_pallas_mpmd_test.py::test_parallel_subkernels_with_kernel` (约 line 362) - 基础 MPMD
- `tpu_sparsecore_pallas_distributed_test.py` - 完整的 Scalar/Vector 协作

### Scalar Prefetch
- `tpu_pallas_test.py` 中搜索 `PrefetchScalarGridSpec` - 动态索引模式

## vLLM TPU Inference 仓库

外部仓库，包含生产级 RPA v3 kernel：
- URL: https://github.com/vllm-project/tpu-inference
- 关键文件: `tpu_inference/kernels/ragged_paged_attention/v3/kernel.py`
- 论文: arXiv:2604.15464

## 建议的阅读顺序

1. `docs/pallas/quickstart.md` → 理解 pallas_call 基本结构
2. `docs/pallas/grid_blockspec.md` → 理解 BlockSpec 语义
3. `docs/pallas/tpu/details.rst` → 理解 TPU 特有概念
4. `docs/pallas/tpu/pipelining.md` → 理解流水线
5. `jax/_src/pallas/mosaic/pipeline.py` → 看 emit_pipeline 实现
6. `docs/pallas/tpu/distributed.md` → 理解信号量和分布式
7. `docs/pallas/tpu/sparsecore.md` → 理解 SparseCore
8. `jax/experimental/pallas/ops/tpu/flash_attention.py` → 看生产 kernel
9. vLLM RPA v3 kernel → 最复杂的生产 kernel

## 已知问题和注意事项

1. **VMEM 标量限制**：不能在 VMEM 中存储标量或 (1,) 形状，必须用 tile 对齐的形状
2. **CMEM 废弃**：v5e/v6e/v7 上 CMEM 容量为 0，不要使用
3. **信号量归零规则**：kernel 结束时所有信号量必须为 0
4. **emit_pipeline 的 out_specs=[]**：当只需要 scratch 累加器而不需要每迭代输出时使用
5. **pl.when vs Python if**：trace 时必须用 `pl.when`，不能用 Python `if`
6. **sync_copy 不需要信号量**：`pltpu.sync_copy(src, dst)` 内部自动管理信号量
7. **BlockSpec 的 None 维度**：`None` 表示 squeeze（该维度不分块）
