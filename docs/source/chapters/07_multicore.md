# 第 7 章：多核编程

前面的章节可以暂时把一个 Pallas kernel 想成“在一个 TPU 核心上处理一个 grid”。真实程序通常还有两层并行：

1. 一个 JAX device 内可能包含多个 TensorCore。Megacore、`dimension_semantics` 和 `pl.core_map` 处理这一层。
2. 一个程序可能使用多个 JAX device。`jax.shard_map` 处理这一层。

这两层可以组合，但不能混为一谈：`pl.core_map` 的一个实例仍然只属于当前 JAX device；`jax.shard_map` 的一个实例则只看到当前 device 的本地 shard。

```text
全局 jax.Array
    │
    │ jax.shard_map：按 device mesh 切成 shard
    ▼
每个 device 上的一份局部数组
    │
    │ pl.core_map / PARALLEL grid：在 device 内分工
    ▼
每个 TensorCore 的局部 VMEM、SMEM、VREG 和计算单元
```

本章先从一个 device 内的 Megacore 开始，再扩展到多个 device，最后把两层组合起来。所有标注“完整例子”的代码块都可以单独保存为 `.py` 文件运行；解释模式例子也可以在 CPU 上检查正确性。

## 先分清 grid、core 和 device

| 名称 | 由谁组织 | 典型索引 | 数据视角 |
| :--- | :--- | :--- | :--- |
| grid program | `pl.pallas_call` / `pltpu.emit_pipeline` | `pl.program_id(axis)` | 一个 `BlockSpec` 对应的块 |
| TensorCore | `pl.core_map` 或 Megacore grid 分区 | `jax.lax.axis_index("core")` | 同一 device 的 HBM，core 私有的 VMEM/SMEM |
| JAX device | `jax.shard_map` | `jax.lax.axis_index("device")` | 全局 `jax.Array` 的一个本地 shard |

“芯片”“TensorCore”和“JAX device”不总是一一对应：

- TPU v4、v5p 支持 Megacore，此时两个物理 TensorCore 可以作为一个 JAX device 暴露。
- 同样是双 TensorCore 芯片，也可能以 split 模式暴露成两个单 core device。
- v5e、v6e 等 lite 芯片每个 device 只有一个 TensorCore。

因此不要根据 TPU 型号或 `jax.device_count()` 猜 core 数量。应直接检查当前运行时：

```python
import jax
from jax.experimental.pallas import tpu as pltpu

if jax.devices()[0].platform == "tpu":
    info = pltpu.get_tpu_info()
    print("chip version:", info.chip_version)
    print("TensorCores in one JAX device:", info.num_cores)
    print("Megacore mode:", info.is_megacore)
    print("split-chip mode:", info.is_split_chip)
```

`info.num_cores` 是每个 JAX device 内可参与当前 Pallas 程序的 TensorCore 数量；它不是整个 pod 的 core 总数。

## 一个 device 内：Megacore

Megacore 中的多个 TensorCore：

- 共享当前 JAX device 的 HBM；
- 各自拥有 VMEM、SMEM、VREG、SREG 和计算单元；
- 可以同时执行互不依赖的工作；
- 对共享 HBM 的重叠写入仍然会产生竞态。

“共享 HBM”不等于“共享 VMEM”。一个 core 写入自己的 VMEM 后，另一个 core 不能直接用普通 load 读取它。需要通信时必须使用 remote DMA、信号量等显式机制。

### 最简单的方法：标记可并行的 grid 维度

TPU 上的 Pallas grid 默认保留顺序执行语义。`pltpu.CompilerParams(dimension_semantics=...)` 可以告诉编译器，哪些 grid 维度的不同 program 可以安全地分到不同 TensorCore：

- `pltpu.PARALLEL`：该维度的各次迭代互相独立，允许跨 core 并行。
- `pltpu.ARBITRARY`：不能假设各次迭代独立，不沿这一维做 Megacore 并行分区。

`dimension_semantics` 的长度必须等于 `grid` 的秩。它描述的是**依赖关系**，而不是性能愿望；只有能证明迭代互不影响时才能写 `PARALLEL`。

### 完整例子 1：自动把独立块分给多个 core

下面的加法有四个互不重叠的列块。真实 Megacore 会把这些 program 分给多个 TensorCore。`InterpretParams` 让同一份代码也能在 CPU 上验证结果；它只验证语义，不模拟真实并行速度。

```python
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


is_tpu_available = jax.devices()[0].platform == "tpu"


def add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]


block = pl.BlockSpec(
    block_shape=(8, 128),
    index_map=lambda j: (0, j),
)


def add(x, y):
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct.like(x),
        in_specs=[block, block],
        out_specs=block,
        grid=(x.shape[1] // 128,),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(pltpu.PARALLEL,),
        ),
        # CPU 使用解释模式；TPU 使用真正的 Mosaic lowering。
        interpret=False if is_tpu_available else pltpu.InterpretParams(),
    )(x, y)


x = jnp.arange(8 * 512, dtype=jnp.float32).reshape(8, 512)
y = jnp.full_like(x, 3)
z = jax.jit(add)(x, y)

np.testing.assert_array_equal(z, x + y)
print(z.shape, z[0, :4])  # (8, 512) [3. 4. 5. 6.]
```

这个例子的逻辑 grid 是 `(4,)`，四个 program 分别写：

```text
program 0 -> o[:,   0:128]
program 1 -> o[:, 128:256]
program 2 -> o[:, 256:384]
program 3 -> o[:, 384:512]
```

它们读取和写入的块都不重叠，因此执行次序不影响结果。编译器可以在两个 TensorCore 上分区这四个 program。

:::{note}
非 Megacore 平台仍可执行这段程序，但 `PARALLEL` 标注不会凭空增加 TensorCore。它的性能效果必须在 `pltpu.get_tpu_info().is_megacore` 为 `True` 的真实 TPU 上测量。
:::

### 哪些维度可以标成 `PARALLEL`

判断方法不是“这层循环很大”，而是交换任意两次迭代后结果是否仍然相同。

适合 `PARALLEL` 的情况：

- 每个 program 写不同的 batch、head 或输出 tile；
- 输入可以重叠读取，但输出不重叠；
- program 之间不通过 scratch buffer 传递状态。

通常应为 `ARBITRARY` 的情况：

- 多个 program 累加到同一个输出块；
- online softmax 的后一次迭代依赖前一次的 running max/sum；
- 不同 program 复用 scratch buffer 中的状态；
- 正确性依赖固定执行顺序。

例如分块矩阵乘法常写成：

```python
dimension_semantics = (
    pltpu.PARALLEL,   # M 方向：不同输出行块
    pltpu.PARALLEL,   # N 方向：不同输出列块
    pltpu.ARBITRARY,  # K 方向：同一输出块上的累加
)
```

前两个维度可以并行，因为 `(m_tile, n_tile)` 不同就会写不同的输出块。K 维的多次迭代更新同一个 accumulator，不能当作互相独立的 program。

以下模式则是错误的：

```python
# 所有 i 都映射到同一个输出块，却声称它们互相独立。
out_spec = pl.BlockSpec((8, 128), lambda i: (0, 0))
dimension_semantics = (pltpu.PARALLEL,)  # 竞态：多个 core 重叠写
```

解释模式主要检查数值语义，不是竞态检测器。错误的并行标注可能只在真实 Megacore 上暴露，而且每次运行的错误结果还可能不同。

### `emit_pipeline` 中的 Megacore

如果在 `pl.core_map` 内使用 `pltpu.emit_pipeline`，还要通过 `core_axis_name` 把 pipeline 的并行维度与 core mesh 连接起来：

```python
pltpu.emit_pipeline(
    body,
    grid=(m_blocks, n_blocks, k_blocks),
    in_specs=[...],
    out_specs=...,
    core_axis_name="core",
    dimension_semantics=(
        pltpu.PARALLEL,
        pltpu.PARALLEL,
        pltpu.ARBITRARY,
    ),
)(x_hbm_ref, y_hbm_ref, o_hbm_ref)
```

`dimension_semantics` 仍描述 grid 依赖；`core_axis_name="core"` 则说明用于执行这些并行 program 的 TensorCore mesh 轴。第 6 章介绍的多缓冲和 DMA 流水线仍在每个 core 的本地 VMEM 中发生。

## 显式控制：`pl.core_map`

`dimension_semantics` 适合“把独立 grid 自动分掉”。当程序需要下面的能力时，应使用更底层的 `pl.core_map`：

- 读取当前物理 core 的编号；
- 精确决定每个 core 处理哪个 HBM 区域；
- 给每个 core 分配独立的 VMEM、SMEM 和信号量；
- 做 core 间同步或 remote DMA；
- 在 core 内再启动 `emit_pipeline`。

先创建 TensorCore mesh：

```python
from jax.experimental.pallas import tpu as pltpu

# 真实 TPU：默认从当前 device 读取 num_cores。
core_mesh = pltpu.TensorCoreMesh(axis_name="core")
print(core_mesh.shape)  # 例如 OrderedDict([("core", 2)])
```

当前 JAX API 使用 `pltpu.TensorCoreMesh(...)`。旧代码中的 `pltpu.create_tensorcore_mesh(...)` 已进入弃用流程。

### `core_map` 的状态式接口

`pl.core_map` 的 body 有一个容易忽略的约束：body 本身不接收普通数组参数，也不返回数组。它通过闭包捕获 Ref，并原地修改 Ref：

```python
x_ref = jax.new_ref(x)
y_ref = jax.empty_ref(jax.ShapeDtypeStruct.like(x))

@pl.core_map(core_mesh)
def per_core():
    # 读取 x_ref，写入 y_ref
    ...

y = jax.freeze(y_ref)
```

`pl.run_scoped` 在每个 core 上分别分配 scratch Ref。也就是说，在两核 Megacore 上声明一个 `pltpu.VMEM((8, 128), dtype)`，实际会得到两份互相独立的 VMEM buffer，而不是一份共享 buffer。

### 完整例子 2：每个 core 处理不同的行

下面显式模拟两个 TensorCore。core 0 处理前四行，core 1 处理后四行；为了能看出分工，每个 core 把自己的编号加到结果中。

```python
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


is_tpu_available = jax.devices()[0].platform == "tpu"
NUM_CORES = 2
ROWS_PER_CORE = 4
core_mesh = pltpu.TensorCoreMesh(
    axis_name="core",
    num_cores=NUM_CORES,
)


@jax.jit
def add_core_id(x):
    # 这两个 Ref 位于当前 JAX device 的 HBM/ANY 空间。
    x_hbm_ref = jax.new_ref(x)
    y_hbm_ref = jax.empty_ref(jax.ShapeDtypeStruct.like(x))

    @pl.core_map(
        core_mesh,
        # InterpretParams 用于在 CPU 上模拟 TPU，不能传给真实 TPU。
        interpret=False if is_tpu_available else pltpu.InterpretParams(),
    )
    def per_core():
        core = jax.lax.axis_index("core")
        row_slice = pl.ds(core * ROWS_PER_CORE, ROWS_PER_CORE)

        def body(x_vmem_ref, y_vmem_ref, dma_sem):
            # 每个 core 只把自己的 HBM slice 搬进本地 VMEM。
            pltpu.async_copy(
                x_hbm_ref.at[row_slice], x_vmem_ref, dma_sem
            ).wait()

            y_vmem_ref[...] = x_vmem_ref[...] + core

            # 两个 core 写回互不重叠的 HBM slice。
            pltpu.async_copy(
                y_vmem_ref, y_hbm_ref.at[row_slice], dma_sem
            ).wait()

        # run_scoped 的三个 scratch 对每个 core 都各分配一份。
        pl.run_scoped(
            body,
            pltpu.VMEM((ROWS_PER_CORE, 128), x.dtype),
            pltpu.VMEM((ROWS_PER_CORE, 128), x.dtype),
            pltpu.SemaphoreType.DMA,
        )

    return jax.freeze(y_hbm_ref)


x = jnp.arange(8 * 128, dtype=jnp.int32).reshape(8, 128)
y = add_core_id(x)
expected = x + jnp.repeat(jnp.arange(2), 4)[:, None]

np.testing.assert_array_equal(y, expected)
print(y[:, 0])
# [  0 128 256 384 513 641 769 897]
```

这个例子包含 `core_map` 最重要的四个步骤：

1. `jax.new_ref` / `jax.empty_ref` 建立 device 级 HBM Ref。
2. `pl.core_map` 让同一个 body 在 core mesh 的每个 core 上执行。
3. `jax.lax.axis_index("core")` 决定当前 core 的工作范围。
4. `pl.run_scoped` 分配 core-local VMEM 和 DMA semaphore。

真实 TPU 上通常不应把 `NUM_CORES` 写死。可以改成：

```python
core_mesh = pltpu.TensorCoreMesh(axis_name="core")
num_cores = core_mesh.shape["core"]
assert x.shape[0] % num_cores == 0
```

上面的 CPU 解释模式故意显式写 `num_cores=2`，这样即使机器只有一个 CPU device，也能覆盖两条 core 路径。`interpret` 则必须根据 backend 选择：CPU 使用 `pltpu.InterpretParams()`，真实 TPU 使用 `False`（也就是正常的 Mosaic lowering）。如果在 TPU 上传入 `InterpretParams()`，解释器引入的 host callback effects 会进入 `mpmd_map` 的 TPU lowering，当前会触发 `tokens_out` 报错。

### 为什么输入输出用 HBM Ref

当前 TensorCore lowering 对多 core `core_map` 有一条重要限制：当 `num_cores > 1` 时，不支持把 VMEM array 直接作为 `core_map` 的输入或输出，因为 `core_map` 外部的操作无法知道 VMEM 数据在 core 之间如何分布。

稳定的结构是：

```text
core_map 外部：HBM/ANY Ref
       │
       │ 每个 core 自己发起 DMA
       ▼
core_map 内部：core-local VMEM scratch
       │
       │ 计算后 DMA 写回互不重叠的区域
       ▼
core_map 外部：HBM/ANY Ref
```

`pl.kernel(...)` 是这一套样板代码的便捷封装：它会包装输出 Ref、`core_map` 和 `run_scoped`。需要教学或精确理解生命周期时，显式写法更清楚；生产代码中使用 `pl.kernel` 通常更简洁。

### core 间通信只看框架

本章不展开 remote DMA 算法，但要先建立正确的同步顺序：

```text
所有 core 分配好本地 buffer
        ↓ barrier
发送方 start remote DMA
        ↓
接收方等待 recv semaphore
        ↓
使用收到的数据
```

目标 core 可以写成 `device_id={"core": dst_core}`。未写出的 mesh 轴保持不变，因此这表示当前 JAX device 内的另一个 core。跨 core 通信前通常先用 barrier 确保所有本地资源已经建立；涉及 barrier semaphore 时要给 `pltpu.CompilerParams` 设置唯一的 `collective_id`。

即使不通信，两个 core 同时写同一 HBM 区域也依然是竞态。最简单、最快的设计通常是让每个 core 拥有不重叠的输出 tile。

## 多个 device：`jax.shard_map`

`jax.shard_map` 让我们写“每个 device 执行的函数”。它是 rank-preserving map：全局数组沿 mesh 轴切成若干同秩的块，body 在每个 device 上接收一个本地块，输出块再按照 `out_specs` 组装成全局 `jax.Array`。

例如全局输入形状为 `(16,)`，device mesh 大小为 4，且 `in_specs=jax.P("device")`：

```text
shard_map 外部：x.shape       == (16,)
shard_map 内部：x_local.shape == (4,)
```

轴没有消失，所以它不同于会把映射轴去掉的 `vmap`。

### `Mesh` 和 `PartitionSpec`

一维 device mesh：

```python
mesh = jax.make_mesh((4,), ("device",))
```

常见的 `PartitionSpec` 写法：

| 写法 | 含义 |
| :--- | :--- |
| `jax.P("device")` | 数组第 0 维沿 `device` mesh 轴切分 |
| `jax.P(None, "device")` | 数组第 1 维沿 `device` 轴切分 |
| `jax.P()` | 不沿这个 mesh 轴切分；每个 device 获得相同值 |
| `jax.P("data", "model")` | 二维 mesh 上分别切分数组的两个维度 |

数组维度必须能被相应 mesh 轴大小整除。`jax.P()` 表示复制而不是“不给这个参数分配设备”。

### 完整例子 3：四个 CPU device 上的 rank-preserving map

JAX 可以在一个进程中创建多个虚拟 CPU device，适合本地学习 `shard_map`。`jax_num_cpu_devices` 必须在第一次查询 device 之前设置，所以请把下面代码作为独立脚本运行。

```python
import jax

# 必须早于 jax.devices()、jax.make_mesh() 和第一次 JAX 计算。
jax.config.update("jax_num_cpu_devices", 4)

import jax.numpy as jnp
import numpy as np


mesh = jax.make_mesh((4,), ("device",))
partition = jax.P("device")
sharding = jax.sharding.NamedSharding(mesh, partition)


@jax.jit
@jax.shard_map(
    mesh=mesh,
    in_specs=partition,
    out_specs=partition,
)
def add_device_id(x_local):
    device_id = jax.lax.axis_index("device")
    return x_local + device_id


host_x = np.arange(16, dtype=np.int32)

# 当前 JAX API 要求实参 sharding 与 in_specs 一致。
x = jax.device_put(host_x, sharding)
y = add_device_id(x)

expected = host_x + np.repeat(np.arange(4), 4)
np.testing.assert_array_equal(np.asarray(y), expected)

print("global shape:", y.shape)                    # (16,)
print("one local shape:", y.addressable_data(0).shape)  # (4,)
print(np.asarray(y))
# [ 0  1  2  3  5  6  7  8 10 11 12 13 15 16 17 18]
```

四个实例实际执行的是：

```text
device 0: [ 0,  1,  2,  3] + 0
device 1: [ 4,  5,  6,  7] + 1
device 2: [ 8,  9, 10, 11] + 2
device 3: [12, 13, 14, 15] + 3
```

`out_specs=jax.P("device")` 表示把四个长度为 4 的局部结果沿第 0 维连接，恢复成长度为 16 的全局数组。

:::{warning}
当前 `jax.shard_map` 会检查实参的物理 sharding。不要把普通的单 device/复制数组直接交给 `in_specs=jax.P("device")`，期待 `shard_map` 隐式重分片。先用 `jax.device_put(x, NamedSharding(mesh, spec))` 或 `jax.reshard` 明确布局。
:::

### `out_specs` 不只是输出布局

`out_specs` 同时声明各 device 的局部结果如何组成全局结果：

- mesh 轴出现在 `out_specs` 中：沿对应数组轴连接各局部结果。
- mesh 轴没有出现在 `out_specs` 中：承诺所有 device 在该 mesh 轴上的结果相同，只保留一份。

因此下面的 identity 是错误的：

```python
@jax.shard_map(
    mesh=mesh,
    in_specs=jax.P("device"),
    out_specs=jax.P(),  # 错误：不同 device 的 x_local 并不相同
)
def wrong(x_local):
    return x_local
```

默认的 `check_vma=True` 会跟踪 value 在哪些 manual mesh axis 上可能不同，并捕获这种错误。不要为了绕过报错随手设置 `check_vma=False`；错误的 `out_specs` 在关闭检查后可能静默地产生未定义结果。

### 完整例子 4：用 `psum` 做 device 间通信

`shard_map` body 可以用 mesh 轴名调用集合通信。下面把四个本地 shard 的对应位置相加：

```python
import jax

jax.config.update("jax_num_cpu_devices", 4)

import jax.numpy as jnp
import numpy as np


mesh = jax.make_mesh((4,), ("device",))
input_spec = jax.P("device")
input_sharding = jax.sharding.NamedSharding(mesh, input_spec)


@jax.jit
@jax.shard_map(
    mesh=mesh,
    in_specs=input_spec,
    out_specs=jax.P(),
)
def sum_corresponding_positions(x_local):
    # psum 后每个 device 都得到相同的长度 4 数组。
    return jax.lax.psum(x_local, "device")


host_x = np.arange(16, dtype=np.int32)
x = jax.device_put(host_x, input_sharding)
y = sum_corresponding_positions(x)

np.testing.assert_array_equal(y, np.array([24, 28, 32, 36]))
print(np.asarray(y))  # [24 28 32 36]
print(y.sharding)     # P(None,)：结果在 device 轴上复制
```

第一个元素 `24` 来自 `0 + 4 + 8 + 12`。`psum` 之后所有 device 都持有相同结果，所以 `out_specs=jax.P()` 是真实的承诺，VMA 检查也能通过。

常见集合通信包括：

| API | 作用 |
| :--- | :--- |
| `jax.lax.psum(x, axis_name)` | all-reduce sum |
| `jax.lax.all_gather(x, axis_name, ...)` | 收集各 device 的 shard |
| `jax.lax.psum_scatter(x, axis_name, ...)` | reduce-scatter |
| `jax.lax.ppermute(x, axis_name, perm)` | 按指定邻接关系置换数据 |
| `jax.lax.all_to_all(x, axis_name, ...)` | 在 device 间交换分块维度 |

集合通信的输入输出 shape 和 varying 状态各不相同。写 `out_specs` 时应根据 body **返回时**每个 device 实际持有的局部数组来推导，而不是根据函数输入猜测。

## 把 `shard_map` 和 Pallas 组合起来

常见的分布式 Pallas 程序结构是：

```python
@jax.jit
@jax.shard_map(...)
def per_device(x_local):
    return pallas_kernel(x_local)
```

`shard_map` 决定每个 device 获得哪些全局数据；Pallas kernel 只处理本地 shard。这样 BlockSpec 和 VMEM 容量都应根据**局部 shape**设计，而不是根据全局 shape 设计。

当前 Pallas 输出类型还不能完整表达 `shard_map` 的 manual-axis varying 信息，因此这个组合需要使用 `check_vma=False`。这是一项接口边界，不是关闭检查的一般建议：外层 `in_specs`、`out_specs` 仍需人工确保正确。

### 完整例子 5：device × core 两层映射

最后把两层组合起来：

- 外层创建四个虚拟 CPU device；
- `shard_map` 把全局 32 行切成每 device 8 行；
- 每个 device 内用 `core_map` 模拟两个 TensorCore；
- 每个 core 处理本地 shard 中的 4 行。

结果加上 `10 * device_id + core_id`，因此可以直接看出每一行由哪个两层 worker 处理。

```python
import jax

jax.config.update("jax_num_cpu_devices", 4)

import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


is_tpu_available = jax.devices()[0].platform == "tpu"

# 第一层：四个 JAX device。
device_mesh = jax.make_mesh((4,), ("device",))
partition = jax.P("device", None)
sharding = jax.sharding.NamedSharding(device_mesh, partition)

# 第二层：每个 device 内模拟两个 TensorCore。
core_mesh = pltpu.TensorCoreMesh(axis_name="core", num_cores=2)


@pl.kernel(
    mesh=core_mesh,
    # shard_map 内部看到的局部 shape 是 (8, 128)。
    out_type=jax.ShapeDtypeStruct((8, 128), jnp.int32),
    scratch_types=[
        pltpu.VMEM((4, 128), jnp.int32),
        pltpu.VMEM((4, 128), jnp.int32),
        pltpu.SemaphoreType.DMA,
    ],
    interpret=False if is_tpu_available else pltpu.InterpretParams(),
)
def identify_worker(
    x_hbm_ref,
    y_hbm_ref,
    x_vmem_ref,
    y_vmem_ref,
    dma_sem,
):
    # core_map 内仍然可以读取外层 shard_map 的 device 轴。
    device_id = jax.lax.axis_index("device")
    core_id = jax.lax.axis_index("core")
    row_slice = pl.ds(core_id * 4, 4)

    pltpu.async_copy(
        x_hbm_ref.at[row_slice], x_vmem_ref, dma_sem
    ).wait()
    y_vmem_ref[...] = x_vmem_ref[...] + 10 * device_id + core_id
    pltpu.async_copy(
        y_vmem_ref, y_hbm_ref.at[row_slice], dma_sem
    ).wait()


@jax.jit
@jax.shard_map(
    mesh=device_mesh,
    in_specs=partition,
    out_specs=partition,
    check_vma=False,  # 当前 Pallas + shard_map 组合所需，见上文。
)
def run_on_all_cores(x_local):
    return identify_worker(x_local)


host_x = np.zeros((32, 128), dtype=np.int32)
x = jax.device_put(host_x, sharding)
y = run_on_all_cores(x)

worker_ids = np.repeat(
    [0, 1, 10, 11, 20, 21, 30, 31], repeats=4
)
expected = np.broadcast_to(worker_ids[:, None], host_x.shape)
np.testing.assert_array_equal(np.asarray(y), expected)

print(np.asarray(y)[:, 0])
# [ 0  0  0  0  1  1  1  1 10 10 10 10 11 11 11 11
#  20 20 20 20 21 21 21 21 30 30 30 30 31 31 31 31]
```

这里有两个互不冲突的命名轴：

```text
device_id = 0, 1, 2, 3       # jax.shard_map 的 device mesh
core_id   = 0, 1              # pl.core_map 的 TensorCore mesh
worker    = (device_id, core_id)
```

在四个双核 Megacore device 上，逻辑上共有 `4 × 2 = 8` 个 core worker。外层每个 device 只看到自己的 8 行，内层每个 core 又只搬运其中 4 行到本地 VMEM。

在真实 TPU 上使用这个结构时需要做四项修改：

1. 删除 `jax_num_cpu_devices`，使用真实的 `jax.device_count()` 构造 device mesh。
2. 把 `core_mesh` 改成 `pltpu.TensorCoreMesh(axis_name="core")`，不要写死 core 数量。
3. 保持示例中的 backend 分支，使 TPU 选择 `interpret=False` 并生成真正的 TPU kernel。
4. 根据真实 device 数、每 device 的局部 shape 和 `core_mesh.shape["core"]` 重新计算输出类型及 VMEM scratch shape。

## 三种入口如何选择

| 需求 | 首选入口 | 原因 |
| :--- | :--- | :--- |
| 一个 device 内，独立输出 tile 自动并行 | `dimension_semantics=PARALLEL` | 改动最少，保留 BlockSpec/pipeline 写法 |
| 一个 device 内，需要 core id、私有 scratch 或通信 | `pl.core_map` | 显式 per-core SPMD 控制 |
| 多个 device 上切分全局数组 | `jax.shard_map` | 明确 global/local shape 和 device 布局 |
| 多 device 且每个 device 是 Megacore | 外层 `shard_map`，内层 `core_map`/Pallas | 两层职责清晰，可分别调试 |

实践中的推荐顺序是：

1. 先写单 device、单 core 正确版本。
2. 如果输出 tile 天然独立，先加 `PARALLEL`，不要立刻手写 core 分工。
3. 只有需要显式 core 身份或通信时才下沉到 `core_map`。
4. 最后用 `shard_map` 扩展到多个 device，并明确每个参数的全局和局部 shape。

## 常见错误

### 1. 把 core 数当成 device 数

`jax.device_count()` 不能告诉你一个 device 内有几个 TensorCore。Megacore 应检查 `pltpu.get_tpu_info().num_cores` 或 `TensorCoreMesh.shape`。

### 2. 把有依赖的 grid 维度标成 `PARALLEL`

如果多个 program 写同一输出块或共享跨迭代状态，标注 `PARALLEL` 会引入竞态。先从数据读写集合证明独立性。

### 3. 认为 Megacore 共享 VMEM

共享的是 HBM。VMEM、SMEM 和寄存器属于各自 TensorCore；跨 core 数据交换必须显式通信。

### 4. 忘记局部 shape

`shard_map` 内的 Pallas kernel 接收本地 shard。BlockSpec、scratch shape 和 grid 都要根据本地 shape 计算。

### 5. `in_specs` 与实参数组的物理 sharding 不一致

当前 JAX 会报错，而不是隐式搬运。调用前使用 `NamedSharding` 配合 `jax.device_put` 或 `jax.reshard`。

### 6. `out_specs=jax.P()` 却返回 device-varying 值

`P()` 是“各 device 结果相同”的承诺，不是“随便挑一个输出”。保持默认 `check_vma=True`，让类型系统检查这项承诺。

### 7. 把解释模式当性能模拟器

解释模式适合检查索引、shape 和数值；它不能证明真实 Megacore 的负载均衡、DMA 重叠或没有硬件竞态。最终必须在目标 TPU 上 profile。
