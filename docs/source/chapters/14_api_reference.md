# 第 14 章：API 参考

本章列出 Pallas TPU 开发中常用的所有 API，按功能分类。

## jax.experimental.pallas (pl)

### 核心

| API | 说明 |
| :--- | :--- |
| `pl.pallas_call(kernel, out_shape, grid_spec, ...)` | 调用 Pallas kernel 的入口 |
| `pl.BlockSpec(block_shape, index_map, memory_space)` | 定义数据块的形状和索引映射 |
| `pl.GridSpec(grid, in_specs, out_specs, scratch_shapes)` | 定义 grid 和所有 BlockSpec |
| `pl.no_block_spec` | 表示该参数不需要 BlockSpec（整体传入）|

### 控制流

| API | 说明 |
| :--- | :--- |
| `pl.loop(start, end)(body_fn)` | Pallas 中的 for 循环（替代 `lax.fori_loop`）|
| `pl.when(condition)(body_fn)` | 条件执行（替代 `lax.cond`）|
| `pl.run_scoped(body_fn, *refs)` | 在作用域内分配临时 Ref |

### 索引

| API | 说明 |
| :--- | :--- |
| `pl.ds(start, size)` / `pl.dslice(start, size)` | 动态切片（运行时确定的 start）|
| `pl.Slice(start, size)` | 静态切片 |
| `pl.multiple_of(x, n)` | 向编译器声明 x 是 n 的倍数 |

### 内存与类型

| API | 说明 |
| :--- | :--- |
| `pl.MemorySpace.ANY` | 不指定内存空间 |
| `pl.Blocked` | 表示维度被分块 |
| `pl.Squeezed` / `pl.squeezed` | 表示维度被压缩（大小为 1）|
| `pl.Element` | 表示标量元素 |

### 原语

| API | 说明 |
| :--- | :--- |
| `pl.program_id(axis)` | 获取当前 grid 索引 |
| `pl.num_programs(axis)` | 获取 grid 维度大小 |
| `pl.debug_print(fmt, *args)` | 调试打印 |
| `pl.reciprocal(x, approx=True)` | 快速近似倒数 |
| `pl.delay(nanos)` | 插入延迟（调试用）|

### 工具函数

| API | 说明 |
| :--- | :--- |
| `pl.cdiv(a, b)` | 向上取整除法 |
| `pl.align_to(x, alignment)` | 对齐到指定值 |
| `pl.next_power_of_2(x)` | 下一个 2 的幂 |
| `pl.strides_from_shape(shape)` | 从 shape 计算 strides |

### 信号量

| API | 说明 |
| :--- | :--- |
| `pl.semaphore_read(sem)` | 读取信号量值 |
| `pl.semaphore_signal(sem, inc, ...)` | 信号量加 inc |
| `pl.semaphore_wait(sem, dec)` | 等待信号量 >= dec 并减去 |

## jax.experimental.pallas.tpu (pltpu)

### 编译器配置

| API | 说明 |
| :--- | :--- |
| `pltpu.CompilerParams(...)` | 编译器参数 |
| `pltpu.GridDimensionSemantics` | PARALLEL, ARBITRARY, CORE_PARALLEL, SUBCORE_PARALLEL |
| `pltpu.PrefetchScalarGridSpec(...)` | 支持标量预取的 GridSpec |
| `pltpu.CostEstimate(flops, bytes_accessed, transcendentals)` | 性能估算提示 |

### 内存空间

| API | 说明 |
| :--- | :--- |
| `pltpu.VMEM` | 向量内存（片上高速）|
| `pltpu.SMEM` | 标量内存（标量核心）|
| `pltpu.HBM` | 高带宽内存（片外）|
| `pltpu.CMEM` | 常量内存 |
| `pltpu.VMEM_SHARED` | 多核共享 VMEM |
| `pltpu.SEMAPHORE` | 信号量内存空间 |

### DMA 操作

| API | 说明 |
| :--- | :--- |
| `pltpu.make_async_copy(src_ref, dst_ref, sem)` | 创建异步 DMA 拷贝描述符 |
| `pltpu.make_async_remote_copy(src_ref, dst_ref, send_sem, recv_sem, device_id)` | 跨芯片远程 DMA |
| `pltpu.async_copy(src_ref, dst_ref, sem)` | 异步拷贝（旧 API）|
| `pltpu.sync_copy(src_ref, dst_ref)` | 同步拷贝 |
| `pltpu.load(ref)` | 从 Ref 加载 |
| `pltpu.store(ref, value)` | 向 Ref 存储 |

### 信号量类型

| API | 说明 |
| :--- | :--- |
| `pltpu.SemaphoreType.DMA` | DMA 完成信号量 |
| `pltpu.SemaphoreType.REGULAR` | 普通信号量 |
| `pltpu.SemaphoreType.BARRIER` | 屏障信号量 |
| `pltpu.dma_semaphore` | DMA 信号量的便捷构造 |

### 流水线

| API | 说明 |
| :--- | :--- |
| `pltpu.emit_pipeline(kernel, grid, in_specs, out_specs, ...)` | 自动生成双缓冲流水线 |
| `pltpu.emit_pipeline_with_allocations(...)` | 带自定义分配的流水线 |
| `pltpu.BufferedRef` | 流水线中的缓冲引用 |

### 数据操作

| API | 说明 |
| :--- | :--- |
| `pltpu.bitcast(ref, dtype)` | 类型重解释（不改变底层 bits）|
| `pltpu.roll(ref, shift, axis)` | 循环移位 |
| `pltpu.pack_elementwise(fn, *args)` | 打包逐元素操作 |
| `pltpu.stochastic_round(x, key)` | 随机舍入 |
| `pltpu.with_memory_space_constraint(ref, space)` | 强制内存空间 |

### 随机数

| API | 说明 |
| :--- | :--- |
| `pltpu.prng_seed(seed)` | 设置 PRNG 种子 |
| `pltpu.prng_random_bits(shape)` | 生成随机 bits |
| `pltpu.to_pallas_key(jax_key)` | JAX key → Pallas key |
| `pltpu.stateful_uniform(shape)` | 有状态均匀分布 |
| `pltpu.stateful_normal(shape)` | 有状态正态分布 |

### 多核

| API | 说明 |
| :--- | :--- |
| `pltpu.core_barrier()` | 核间屏障同步 |
| `pltpu.run_on_first_core(fn)` | 只在第一个核心执行 |
| `pltpu.get_barrier_semaphore()` | 获取屏障信号量 |
| `pltpu.TensorCoreMesh` | TensorCore 网格 |

### MXU 底层原语

| API | 说明 |
| :--- | :--- |
| `pltpu.matmul_push_rhs(rhs)` | 将 RHS 推入 MXU 权重寄存器 |
| `pltpu.matmul_acc_lhs(lhs)` | 用 LHS 累加到 MXU 累加器 |
| `pltpu.matmul_pop()` | 从 MXU 累加器弹出结果 |

这些底层原语允许手动控制 MXU 的流水线，在需要极致性能时使用。

### TPU 信息

| API | 说明 |
| :--- | :--- |
| `pltpu.get_tpu_info()` | 获取当前 TPU 信息 |
| `pltpu.ChipVersion` | 芯片版本枚举 |
| `pltpu.Tiling` | Tiling 信息 |

### 调试

| API | 说明 |
| :--- | :--- |
| `pltpu.InterpretParams` | 解释模式参数 |
| `pltpu.force_tpu_interpret_mode()` | 强制解释模式（CPU 上模拟 TPU）|
| `pltpu.set_tpu_interpret_mode(True/False)` | 设置解释模式 |

## CompilerParams 详细参数

```python
pltpu.CompilerParams(
    dimension_semantics=(...),      # Grid 维度语义
    allow_input_fusion=True,        # 允许输入融合
    vmem_limit_bytes=None,          # VMEM 使用限制
    collective_id=None,             # 集合通信 ID
    has_side_effects=False,         # 是否有副作用
    internal_scratch_in_bytes=0,    # 内部 scratch 大小
    disable_bounds_checks=False,    # 关闭边界检查
    disable_semaphore_checks=False, # 关闭信号量检查
    skip_device_barrier=False,      # 跳过设备屏障
    fuse_transposed_lhs_in_matmul=False,  # 融合转置
    opt_level=None,                 # 优化级别
)
```

## pallas_call 完整签名

```python
pl.pallas_call(
    kernel,                    # kernel 函数
    out_shape=...,             # 输出形状和类型
    grid_spec=None,            # GridSpec 或 PrefetchScalarGridSpec
    grid=None,                 # 简写 grid（与 grid_spec 二选一）
    in_specs=None,             # 输入 BlockSpec 列表
    out_specs=None,            # 输出 BlockSpec 列表
    scratch_shapes=None,       # Scratch buffer 形状列表
    input_output_aliases=None, # 输入输出别名（原地更新）
    compiler_params=None,      # 编译器参数
    interpret=False,           # 解释模式
)
```
