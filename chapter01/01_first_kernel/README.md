# 第一个 Pallas TPU kernel

本节从一个最小的完整 kernel 出发，逐行说明 Pallas TPU kernel 的写法：程序怎样限定在一颗芯片、一个 TensorCore 上，kernel 的参数从哪里来，Ref 与值有什么区别，数据怎样在 HBM 与 TC VMEM 之间移动。本章之后各节都沿用这里的写法，只说明每个实验改动了哪几行。

本节实验：最小 kernel [源码](01_pallas_scale.py)、[输出](01_pallas_scale.txt)；值与 Ref [源码](02_pallas_values_and_refs.py)、[输出](02_pallas_values_and_refs.txt)；检查开关 [源码](03_pallas_checks.py)、[输出](03_pallas_checks.txt)。

## 完整的程序

下面的程序把一个 `f32[8,128]` 乘以 2，省略了 import 和数值检查：

```python
import tpu_init
tpu_init.initialise_one_chip()  # 必须在 import jax 之前

mesh = jax.make_mesh((1,), ('device',))
tc_mesh = pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)

@jax.shard_map(
    mesh=mesh,
    in_specs=P(),
    out_specs=P(),
    check_vma=False,
)
def scale(x: jax.Array) -> jax.Array:
    @pl.kernel(
        out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
        mesh=tc_mesh,
        scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.SemaphoreType.DMA),
        name='scale',
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            disable_semaphore_checks=True,
        ),
    )
    def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, sem: Ref) -> None:
        pltpu.async_copy(x_hbm, x_vmem, sem).wait()
        x_vmem[...] = x_vmem[...] * 2.0
        pltpu.async_copy(x_vmem, o_hbm, sem).wait()

    return kernel(x)

x = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
y = jax.jit(scale)(x)
```

程序分为三层：最外层决定用几颗芯片，中间一层（`shard_map`）决定每颗芯片执行什么，最内层（`pl.kernel`）是 TensorCore 上运行的 kernel。下面逐层说明。

## 第一层：runtime 打开几颗芯片

`tpu_init.initialise_one_chip()` 设置三个环境变量：

```python
os.environ['TPU_CHIPS_PER_PROCESS_BOUNDS'] = '1,1,1'
os.environ['TPU_PROCESS_BOUNDS'] = '1,1,1'
os.environ['TPU_VISIBLE_CHIPS'] = '0'
```

TPU runtime 在 JAX 第一次访问设备时读取这些变量，所以必须在 `import jax` 之前设置。设置后，`jax.devices()` 只返回一个 device，即本 host 的第 0 颗芯片。改用 `initialise_one_chip(1)` 就换成第 1 颗芯片；不同芯片的进程可以同时运行。第二章用 `initialise_local_chips()` 打开本 host 的全部四颗芯片。

这一层只决定 runtime 打开哪些芯片，不决定程序用芯片上的几个 TensorCore。

## 第二层：shard_map 与两级 mesh

程序有两个 mesh。`jax.make_mesh((1,), ('device',))` 是 device mesh，由这一颗芯片组成。`pltpu.TensorCoreMesh(axis_name='tc', num_cores=1)` 是 TensorCore mesh，描述 kernel 在一颗芯片内部用几个 TensorCore。

> 暂且可以理解为：一颗 TPU v4 芯片有两个 TensorCore，它们执行同一份程序。`num_cores=1` 时只有 TensorCore 0 执行 kernel 主体，TensorCore 1 跳过主体。第二章第 1 节详细介绍这种称为 Megacore 的组织方式，以及 `num_cores=2` 的写法。

`jax.shard_map` 把函数 `scale` 变成“device mesh 中每个 device 各执行一份”的程序。`in_specs=P()` 和 `out_specs=P()` 表示输入输出都不切分，每个 device 拿到完整的数组。这里只有一个 device，`shard_map` 并非必需，`pl.kernel` 也可以直接在 `jax.jit` 中调用。本教程统一写在 `shard_map` 里，因为第二章跨芯片时，同一写法只需改 mesh 和 `in_specs`/`out_specs`。

`check_vma=False` 关闭 `shard_map` 对“值在各 device 之间是否不同”的类型检查。打开这项检查（`check_vma=True`）时，Pallas 要求在 `out_type` 的 `jax.ShapeDtypeStruct` 上用 `manual_axis_type` 声明输出沿哪些 mesh 轴变化，否则报错。本教程不使用这项检查。

原生 XLA 的 baseline 不需要这一层，但也要限定为一个 TensorCore，否则编译器会自己把工作分给两个 TensorCore，与本章的 Pallas kernel 不可比。做法是在 runtime 层再加一个参数，让一颗芯片的两个 TensorCore 各作为一个 device 出现（称为 split-chip）；XLA 的程序默认在第 0 个 device 上运行，也就只用 TensorCore 0。本章的 XLA 实验都调用 `tpu_init.initialise_one_core()`，它在 `initialise_one_chip()` 的基础上多设一个环境变量：

```python
os.environ['LIBTPU_INIT_ARGS'] = '--deepsea_chip_config_name=legacy'
```

这样编译出的 XLA 程序是纯粹的单 TensorCore 程序。本章的 Pallas kernel 仍用默认模式加 `num_cores=1`；第二章第 1 节比较这两种组织方式。

## 第三层：pl.kernel 的参数

`pl.kernel` 是一个装饰器，被装饰的 `kernel` 函数就是 TensorCore 执行的代码。它的参数：

| 参数 | 含义 |
| --- | --- |
| `out_type` | 输出的 shape 和 dtype。可以是一个 `jax.ShapeDtypeStruct`，也可以是它们组成的 tuple |
| `mesh` | kernel 在哪个 mesh 上执行，这里是只含一个 TensorCore 的 `tc_mesh` |
| `scratch_types` | kernel 运行期间临时使用的 buffer 和信号量，kernel 结束即释放 |
| `name` | kernel 的名字，出现在 HLO 和机器清单中，方便查找 |
| `compiler_params` | 传给 Mosaic 编译器的参数，见本节最后一小节 |

装饰后的 `kernel` 像普通 JAX 函数一样调用：`kernel(x)` 返回一个 `out_type` 形状的数组。

被装饰的函数本身不返回任何值，它的参数全部是 Ref，按以下顺序排列：

1. 每个输入一个 Ref：`x_hbm`；
2. 每个输出一个 Ref：`o_hbm`；
3. `scratch_types` 中的每一项一个 Ref：`x_vmem` 和 `sem`。

Ref 是一块内存的引用，不是数组的值。输入输出 Ref 指向的就是输入输出数组所在的内存。`TensorCoreMesh` 的默认内存空间是 `ANY`，即数组原来在哪里就留在哪里；JAX 数组在 HBM 中，所以 `x_hbm` 和 `o_hbm` 都指向 HBM。kernel 要产生输出，就必须把结果写进 `o_hbm`。

`scratch_types` 中：

- `pltpu.VMEM(shape, dtype)` 在 TC VMEM 中分配一个 buffer；
- `pltpu.SemaphoreType.DMA` 分配一个 DMA 信号量，用于等待 DMA 完成；`pltpu.SemaphoreType.DMA((n,))` 分配 n 个。

## kernel 主体：DMA、读 Ref、写 Ref

kernel 主体只有三行。

第一行把输入从 HBM 搬到 TC VMEM：

```python
pltpu.async_copy(x_hbm, x_vmem, sem).wait()
```

TensorCore 的向量单元只能读写 TC VMEM，不能直接访问 HBM，所以 HBM 中的数据必须先用 DMA 搬进来。`pltpu.async_copy(源, 目的, 信号量)` 发起一次 DMA 并立即返回一个描述符，DMA 在后台进行；`.wait()` 等到这次 DMA 完成。发起和等待可以分开写，中间插入其他工作，本章第 3 节会用到这一点。

第二行是计算：

```python
x_vmem[...] = x_vmem[...] * 2.0
```

`x_vmem[...]` 出现在右边时是读：把整个 Ref 的内容读成一个值。`x_vmem[...] = 值` 是写：把值写进 Ref。读出的值是一个普通的 JAX 数组，可以用 `jnp` 的运算处理；它存放在 TensorCore 的向量寄存器 TC VREG 中。`[...]` 表示整个 Ref，也可以写 `x_vmem[0:8, :]` 只读写一部分。

第三行把结果从 TC VMEM 搬回 HBM 的输出 Ref：

```python
pltpu.async_copy(x_vmem, o_hbm, sem).wait()
```

## 编译与调用

调用 `jax.jit(scale)(x)` 时，JAX 先追踪 `scale` 得到计算图，Pallas 把 `kernel` 交给 Mosaic 编译成 TensorCore 程序，再嵌入整个 XLA 程序。实验打印的编译后 HLO 中，整个 kernel 是一条 `custom-call`：

```text
ENTRY %main.1 (x_hbm.1: f32[8,128]) -> f32[8,128] {
ROOT %scale.1 = f32[8,128]{1,0:T(8,128)} custom-call(%x_hbm.1), custom_call_target="tpu_custom_call", ...
```

输出的 layout 写作 `{1,0:T(8,128)}`：数组在 HBM 中按 `8×128` 的 tile 存放。本章第 3、4 节会反复用到这一点。

实验脚本不直接调用 `jax.jit`，而是用 [`tpuasm_tools.compile`](../../tpuasm_tools.py)。两者的编译结果相同，后者在编译时保留源码位置，以便下一节读机器清单。

## 值与 Ref：读写 Ref 才产生 load/store

本小节实验[源码](02_pallas_values_and_refs.py)、[输出](02_pallas_values_and_refs.txt)。

相对于基本形式，输入改为 `f32[16,128]`，多分配一个输出 buffer `o_vmem`，kernel 主体的计算改为：

```python
x = x_vmem[...]
y = x * x
o_vmem[...] = y * x + y + x
o_vmem[0:8, :] = x_vmem[8:16, :]
```

清单中的向量指令只有：

```text
vld  v0 ← [vmem:0x8]          # x 的第 8–15 行
vmul v1 = v0 * v0              # y
vst  [vmem:0x10] ← v0          # o 的第 0–7 行 = x 的第 8–15 行
vmul v2 = v1 * v0
vadd v3 = v2 + v1
vadd v4 = v3 + v0
vst  [vmem:0x18] ← v4          # o 的第 8–15 行
```

这个结果说明了三件事：

- 值 `x`、`y` 只存在于 TC VREG 中。连续运算不需要为中间值分配 TC VMEM，也不产生额外的 load/store。
- 第四行的 `x_vmem[8:16, :]` 没有产生新的 `vld`，编译器复用了已经读进 `v0` 的值。
- `o_vmem` 的第 0–7 行先写多项式、后被覆盖，编译器删掉了前一次写，连 x 第 0–7 行的读取和计算也一并删掉了。

kernel 主体仍然由编译器优化，写下的源码不等于实际执行的指令。所以本教程的每个实验都要读机器清单核对。

## CompilerParams：越界检查与信号量检查

本小节实验[源码](03_pallas_checks.py)、[输出](03_pallas_checks.txt)。

本教程的 kernel 一律设置：

```python
compiler_params=pltpu.CompilerParams(
    disable_bounds_checks=True,
    disable_semaphore_checks=True,
)
```

实验把输入 DMA 改为只搬 `x_hbm.at[pl.ds(8, 8)]`（第 8–15 行，`.at[]` 的写法见本章第 3 节），在四种参数下编译同一个 kernel：

| 参数 | kernel 段 bundle 数 | `shalt` 条数 |
| --- | ---: | ---: |
| 默认参数 | 54 | 5 |
| 只关闭越界检查 | 34 | 1 |
| 只关闭信号量检查 | 50 | 4 |
| 两者都关闭 | 30 | 0 |

`shalt` 让 TensorCore 停机。默认参数下：

- 每次 DMA 之前，都有一串标量比较和一条带谓词的 `shalt`，注释为 `BoundsCheck ... for dma.hbm_to_vmem`。这是越界检查：DMA 的地址超出 buffer 时停机，而不是悄悄读写别处的内存。
- kernel 结束前有一条 `shalt`，注释为 `Semaphore (scratch argument 1) has a nonzero value upon exit from a Mosaic kernel`。这是信号量检查：若退出时信号量不为 0，说明某次 DMA 没有等待，或者 signal 与 wait 没有配对。

这两类检查在调试时有用，但它们是额外的标量指令和分支，与要研究的硬件操作无关。本教程关闭它们，使机器清单只剩下 kernel 本身。代价是：写错地址或漏写 `.wait()` 时，kernel 不会停机报错，而是读到错误的数据或在下一次运行时出错。新写的 kernel 出现异常结果时，应先打开这两项检查重新运行。
