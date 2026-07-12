# Pallas TPU API 完整清单

> 来源：`jax/jax/experimental/pallas/tpu.py` (pltpu) 和 `jax/jax/experimental/pallas/__init__.py` (pl)
>
> JAX 仓库版本：main branch (2025 年 7 月 clone)

## pltpu 命名空间 (`from jax.experimental.pallas import tpu as pltpu`)

### 内存空间常量

| 常量 | 说明 |
| :--- | :--- |
| `pltpu.VMEM` | TensorCore 的向量内存（16-192MB），用于向量/矩阵计算 |
| `pltpu.VMEM_SHARED` | SparseCore 的共享向量内存，Scalar/Vector Subcore 共享 |
| `pltpu.SMEM` | 标量内存，支持任意大小，用于索引/标量值 |
| `pltpu.HBM` | 高带宽内存（设备主存） |
| `pltpu.CMEM` | 通信内存（v5e/v6e/v7 上容量为 0，基本废弃） |
| `pltpu.HOST` | 主机内存 |
| `pltpu.SEMAPHORE` | 信号量内存空间 |

### Grid 维度语义

| 常量 | 说明 |
| :--- | :--- |
| `pltpu.PARALLEL` | 并行维度（编译器可重排序） |
| `pltpu.ARBITRARY` | 顺序维度（严格按序执行，用于有跨迭代依赖时） |
| `pltpu.CORE_PARALLEL` | 跨 core 并行 |
| `pltpu.SUBCORE_PARALLEL` | 跨 subcore 并行 |

### DMA 与内存操作

| API | 说明 |
| :--- | :--- |
| `pltpu.make_async_copy(src_ref, dst_ref, sem)` | 创建异步 DMA 拷贝对象，返回 `AsyncCopyDescriptor` |
| `pltpu.async_copy(src_ref, dst_ref, sem)` | 底层异步拷贝原语 |
| `pltpu.async_remote_copy(src_ref, dst_ref, send_sem, recv_sem, device_id=...)` | 跨设备异步 DMA |
| `pltpu.make_async_remote_copy(...)` | 创建跨设备异步拷贝描述符 |
| `pltpu.sync_copy(src_ref, dst_ref)` | 同步 DMA（内部自动分配信号量） |
| `pltpu.load(ref, ...)` | 从 Ref 加载数据 |
| `pltpu.store(ref, ...)` | 向 Ref 存储数据 |

### 信号量

| API | 说明 |
| :--- | :--- |
| `pltpu.SemaphoreType.DMA` | DMA 信号量类型 |
| `pltpu.SemaphoreType.REGULAR` | 常规信号量类型（用于跨 core/subcore 同步） |
| `pltpu.SemaphoreType.BARRIER` | 屏障信号量类型 |
| `pltpu.dma_semaphore` | DMA 信号量快捷方式 |
| `pltpu.get_barrier_semaphore()` | 获取全局屏障信号量 |

注意：`pl.semaphore_signal` / `pl.semaphore_wait` / `pl.semaphore_read` 在 `pl` 命名空间中。

### 流水线

| API | 说明 |
| :--- | :--- |
| `pltpu.emit_pipeline(body, grid, in_specs, out_specs, ...)` | 发射软件流水线（推荐方式） |
| `pltpu.emit_pipeline_with_allocations(...)` | 带显式分配的流水线 |
| `pltpu.BufferedRef` | 流水线中的缓冲引用 |
| `pltpu.BufferType` | 缓冲类型枚举 |
| `pltpu.PipelineStep` | 流水线步骤 |

### 矩阵运算（MXU 相关）

| API | 说明 |
| :--- | :--- |
| `pltpu.matmul_push_rhs(rhs_ref)` | 将 RHS 推入 MXU 的权重 FIFO |
| `pltpu.matmul_acc_lhs(lhs_ref, acc_ref)` | 用 LHS 和已推入的 RHS 做矩阵乘加 |
| `pltpu.matmul_pop(acc_ref)` | 弹出 MXU 累加器结果 |

### 随机数（PRNG）

| API | 说明 |
| :--- | :--- |
| `pltpu.prng_seed(seed)` | 设置硬件 PRNG 种子 |
| `pltpu.prng_random_bits(shape)` | 生成随机比特 |
| `pltpu.stochastic_round(x, bits)` | 随机舍入 |
| `pltpu.to_pallas_key(key)` | 将 JAX key 转换为 Pallas key |
| `pltpu.stateful_bernoulli(...)` | 有状态伯努利采样 |
| `pltpu.stateful_bits(...)` | 有状态随机比特 |
| `pltpu.stateful_normal(...)` | 有状态正态采样 |
| `pltpu.stateful_uniform(...)` | 有状态均匀采样 |
| `pltpu.sample_block(...)` | 采样一个块 |

### 数据操作

| API | 说明 |
| :--- | :--- |
| `pltpu.bitcast(x, dtype)` | 位转换（不改变底层比特） |
| `pltpu.roll(x, shift, axis)` | 沿轴滚动 |
| `pltpu.pack_elementwise(fn, x, y)` | 打包元素级操作（用于 bf16 配对） |
| `pltpu.unpack_elementwise(fn, x)` | 解包元素级操作 |
| `pltpu.touch(ref)` | 标记 ref 被访问（防止编译器优化掉） |
| `pltpu.trace_value(x, name)` | 调试用：在 trace 中打印值 |
| `pltpu.with_memory_space_constraint(x, memory_space)` | 强制数据在指定内存空间 |

### 硬件信息

| API | 说明 |
| :--- | :--- |
| `pltpu.get_tpu_info()` | 获取当前 TPU 硬件信息 |
| `pltpu.get_tpu_info_for_chip(chip)` | 获取指定芯片的硬件信息 |
| `pltpu.is_tpu_device(device)` | 检查是否为 TPU 设备 |
| `pltpu.TpuInfo` | TPU 信息数据类 |
| `pltpu.ChipVersion` | 芯片版本枚举 |
| `pltpu.Tiling` | Tiling 信息 |

### 编译器参数

| API | 说明 |
| :--- | :--- |
| `pltpu.CompilerParams(...)` | 编译器参数（dimension_semantics, collective_id, needs_layout_passes, use_tc_tiling_on_sc 等） |

### Mesh 与多核

| API | 说明 |
| :--- | :--- |
| `pltpu.TensorCoreMesh(...)` | TensorCore mesh |
| `pltpu.CoreType` | Core 类型枚举 |

### 辅助函数

| API | 说明 |
| :--- | :--- |
| `pltpu.core_barrier(...)` | 核间屏障同步 |
| `pltpu.run_on_first_core(fn)` | 只在第一个 core 上执行 |
| `pltpu.einshape(...)` | Einstein 形状推导 |

### 解释模式

| API | 说明 |
| :--- | :--- |
| `pltpu.InterpretParams` | 解释模式参数 |
| `pltpu.force_tpu_interpret_mode(...)` | 强制解释模式 |
| `pltpu.set_tpu_interpret_mode(...)` | 设置解释模式 |
| `pltpu.reset_tpu_interpret_mode_state()` | 重置解释模式状态 |

### SparseCore 相关 (`from jax.experimental.pallas import tpu_sc as plsc`)

| API | 说明 |
| :--- | :--- |
| `plsc.VectorSubcoreMesh(core_axis_name, subcore_axis_name, num_cores, num_subcores)` | Vector Subcore mesh |
| `plsc.ScalarSubcoreMesh(axis_name, num_cores)` | Scalar Subcore mesh |
| `plsc.parallel_loop(start, end, step)` | SparseCore 并行循环 |

---

## pl 命名空间 (`from jax.experimental import pallas as pl`)

### 核心 API

| API | 说明 |
| :--- | :--- |
| `pl.pallas_call(kernel, out_shape, grid, in_specs, out_specs, scratch_shapes, compiler_params)` | 调用 Pallas kernel |
| `pl.kernel(body, mesh, out_type, scratch_types, compiler_params)` | MPMD kernel（多 subcore） |
| `pl.BlockSpec(block_shape, index_map, memory_space)` | 块规格 |
| `pl.GridSpec(...)` | Grid 规格 |
| `pl.PrefetchScalarGridSpec(...)` | 带标量预取的 Grid 规格 |

### 控制流

| API | 说明 |
| :--- | :--- |
| `pl.program_id(axis)` | 获取当前 grid 坐标 |
| `pl.num_programs(axis)` | 获取 grid 大小 |
| `pl.when(condition)` | 条件执行（decorator） |
| `pl.loop(start, end, step)` | 循环（decorator） |
| `pl.run_scoped(fn, **refs)` | 作用域内分配临时 ref |
| `pl.select_ref(condition, ref_true, ref_false)` | 条件选择 ref |

### 索引

| API | 说明 |
| :--- | :--- |
| `pl.ds(start, size)` | 动态切片（dynamic slice） |
| `pl.dslice(start, size)` | 同 `pl.ds` |
| `pl.multiple_of(x, n)` | 标记 x 是 n 的倍数（帮助编译器优化） |

### 信号量

| API | 说明 |
| :--- | :--- |
| `pl.semaphore_signal(sem, inc=1, device_id=...)` | 信号量加 inc |
| `pl.semaphore_wait(sem, value)` | 等待信号量 >= value，然后减 value |
| `pl.semaphore_read(sem)` | 读取信号量当前值 |
| `pl.DeviceIdType.LOGICAL` | 逻辑设备 ID |
| `pl.DeviceIdType.MESH` | Mesh 设备 ID |

### 数学

| API | 说明 |
| :--- | :--- |
| `pl.dot(lhs, rhs, ...)` | 矩阵乘法（映射到 MXU） |

---

## 关键文件位置（JAX 仓库）

| 路径 | 内容 |
| :--- | :--- |
| `jax/experimental/pallas/tpu.py` | pltpu 公开 API 导出 |
| `jax/experimental/pallas/__init__.py` | pl 公开 API 导出 |
| `jax/_src/pallas/mosaic/core.py` | CompilerParams, MemorySpace, SemaphoreType 定义 |
| `jax/_src/pallas/mosaic/helpers.py` | sync_copy, core_barrier 实现 |
| `jax/_src/pallas/mosaic/pipeline.py` | emit_pipeline 实现 |
| `jax/_src/pallas/mosaic/primitives.py` | make_async_copy, prng, matmul 等原语 |
| `jax/_src/pallas/mosaic/tpu_info.py` | TPU 硬件信息（各代次容量等） |
| `jax/_src/pallas/helpers.py` | pl.loop, pl.when, pl.run_scoped 等 |
| `jax/experimental/pallas/ops/tpu/` | 生产级 kernel 实现 |
| `tests/pallas/tpu_pallas_test.py` | 主要 TPU Pallas 测试 |
| `tests/pallas/tpu_pallas_pipeline_test.py` | 流水线测试 |
| `tests/pallas/tpu_pallas_mpmd_test.py` | MPMD/SparseCore 测试 |
| `tests/pallas/tpu_sparsecore_pallas_distributed_test.py` | SparseCore 分布式测试 |
| `docs/pallas/tpu/` | 官方 TPU Pallas 文档源码 |
