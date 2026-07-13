# 第 3 章：Pallas 控制流

#### `pl.when`：只在条件成立时执行一段副作用代码

`pl.when(condition)` 是 Pallas 提供的条件执行 helper，形式上是一个 decorator。它最适合表达“如果条件成立，就做一次写入、初始化、DMA 或信号量操作；否则什么都不做”的场景。

```python
def kernel(x_ref, o_ref, scratch_ref):
    i = pl.program_id(0)
    n = pl.num_programs(0)

    @pl.when(i == 0)
    def _init():
        scratch_ref[...] = jnp.zeros(scratch_ref.shape, scratch_ref.dtype)

    scratch_ref[...] += x_ref[...]

    @pl.when(i == n - 1)
    def _store():
        o_ref[...] = scratch_ref[...]
```

JAX 源码中的 `pl.when` 实现很简单：如果 `condition` 是 Python `bool`，它等价于普通的 `if condition: f()`；如果 `condition` 是 JAX/Pallas 的数组标量，它会生成一个 `jax.lax.cond(condition, f, lambda: None)`。这意味着：

- `pl.when` 可以接受 `pl.program_id(...) == ...` 这样的运行时条件。
- 被装饰的函数用于执行副作用，不应该依赖它的返回值。
- 分支中的 Python 代码仍然会被 trace；真正受条件控制的是生成出来的 kernel 操作。

`pl.when` 在 Pallas TPU kernel 中非常常见。例如矩阵乘法按 `K` 分块累加时，通常只在第一个 `K` program 清零累加器，只在最后一个 `K` program 写回输出：

```python
k = pl.program_id(2)
num_k = pl.num_programs(2)

@pl.when(k == 0)
def _zero_acc():
    acc_ref[...] = jnp.zeros(acc_ref.shape, acc_ref.dtype)

acc_ref[...] += jnp.dot(lhs_ref[...], rhs_ref[...])

@pl.when(k == num_k - 1)
def _write_back():
    out_ref[...] = acc_ref[...].astype(out_ref.dtype)
```

如果有两个互斥的副作用分支，可以写成两个 `pl.when`：

```python
is_even = jax.lax.rem(pl.program_id(0), 2) == 0

@pl.when(is_even)
def _even_path():
    o_ref[...] = x_ref[...] + 1

@pl.when(~is_even)
def _odd_path():
    o_ref[...] = x_ref[...] - 1
```

这种写法清晰，但要记住两个分支都会被 trace，只是在运行时只执行满足条件的操作。

#### `jax.lax.cond`：根据条件选择一个值

如果分支需要返回值，而不是只做一次条件副作用，应使用 `jax.lax.cond`。它的两个分支都必须返回相同结构、相同 shape 和 dtype 的值。

```python
def kernel(x_ref, o_ref):
    i = pl.program_id(0)

    def first_program():
        return jnp.zeros(x_ref.shape, x_ref.dtype)

    def other_program():
        return x_ref[...] + 1

    o_ref[...] = jax.lax.cond(i == 0, first_program, other_program)
```

`lax.cond` 适合表达“计算 A 或计算 B，然后把结果继续参与后续表达式”的情况。例如最后一个 `K` block 需要 mask 掉越界元素，而其他 block 可以直接计算：

```python
def last_block():
    x = jnp.where(x_mask, x_ref[...], 0)
    y = jnp.where(y_mask, y_ref[...], 0)
    return jnp.dot(x, y)

def normal_block():
    return jnp.dot(x_ref[...], y_ref[...])

acc_ref[...] += jax.lax.cond(k == num_k - 1, last_block, normal_block)
```

当分支里主要是 `Ref` 写入时，`pl.when` 通常更贴近意图；当分支要产生一个数组值并继续参与计算时，`lax.cond` 通常更自然。

#### `pl.loop` 与 `jax.lax.fori_loop`：运行时循环

如果循环次数是 Python 常量，普通 Python `for` 可以用于展开少量重复代码：

```python
for mxu in range(num_mxus):  # num_mxus 是 Python int
    ...
```

但如果循环边界或循环变量要参与 kernel 中的 traced 计算，就应该使用 `pl.loop` 或 `jax.lax.fori_loop`。`pl.loop` 是 Pallas 提供的 decorator 形式，本质上会包装成 `lax.fori_loop`；它支持 `step`、`unroll`，也支持携带一个 loop carry。

没有 carry、只做副作用时：

```python
@pl.loop(0, num_steps)
def _(t):
    @pl.when(t == 0)
    def _first_step():
        scratch_ref[...] = jnp.zeros(scratch_ref.shape, scratch_ref.dtype)

    scratch_ref[...] += x_ref[...]
```

需要累积一个值时：

```python
acc = pl.loop(
    0,
    4,
    init_carry=jnp.zeros(x_ref.shape, x_ref.dtype),
)(lambda t, carry: carry + x_ref[...] * t.astype(x_ref.dtype))

o_ref[...] = acc
```

也可以直接使用 `jax.lax.fori_loop`：

```python
def body(t, carry):
    return carry + x_ref[...] * t.astype(x_ref.dtype)

o_ref[...] = jax.lax.fori_loop(
    0,
    4,
    body,
    jnp.zeros(x_ref.shape, x_ref.dtype),
)
```

无论使用哪一种写法，loop carry 的结构、shape 和 dtype 都必须在每次迭代中保持一致。需要按条件提前停止时，可以使用 `jax.lax.while_loop`，但同样要保持 carry 类型稳定。

#### `pl.select_ref`：运行时选择 Ref

普通数组值可以用 `jax.lax.cond`、`jnp.where` 或 `jax.lax.select` 选择；但是 `Ref` 不是普通数组值，不能简单地从 `jax.lax.cond` 返回一个 `Ref` 再交给 DMA。Pallas 为这类场景提供 `pl.select_ref(idx, *refs)`，用于根据运行时标量索引选择一个 Ref。

更直白地说，`select_ref` 选择的是一组已经准备好的候选 Ref，而不是返回一个可以像普通数组一样继续计算的值。候选 Ref 本身可以先通过 `.at[...]` 取子引用，例如 `pl.select_ref(i, x0_ref.at[...], x1_ref.at[...])`；选出来的 Ref 通常应当直接作为 DMA 的源或目标使用。

典型用途是循环中从多个 HBM Ref 中选择一个，拷贝到同一个输出区域：

```python
def kernel(x0_ref, x1_ref, y_ref):
    def body(sem):
        @pl.loop(0, 2)
        def _(i):
            src_ref = pl.select_ref(i, x0_ref, x1_ref)
            dst_ref = y_ref.at[pl.ds(i * 128, 128)]
            pltpu.async_copy(src_ref, dst_ref, sem).wait()

    pl.run_scoped(body, pltpu.SemaphoreType.DMA)
```

如果只是选择数组值，不要用 `select_ref`；如果需要根据运行时条件选择 DMA 的源或目标 Ref，才考虑它。

#### 常用选择表

| 场景 | 推荐写法 | 说明 |
| :--- | :--- | :--- |
| 条件由 Python 常量、静态参数、shape 决定 | Python `if` / `for` | trace 时已经能决定分支或展开次数 |
| 根据 `program_id`、Ref 内容、运行时标量做一次条件写入 | `pl.when` | 适合初始化、写回、DMA、信号量等副作用 |
| 根据运行时条件选择一个数组值 | `jax.lax.cond` | 两个分支返回结构、shape、dtype 必须一致 |
| 根据运行时标量做固定次数循环 | `pl.loop` / `jax.lax.fori_loop` | carry 类型必须稳定，可用 `unroll` 控制展开 |
| 根据运行时条件继续或停止循环 | `jax.lax.while_loop` | 条件和 carry 都是 traced 值 |
| 根据运行时索引选择 DMA 使用的 Ref | `pl.select_ref` | 作用于 Ref，不用于普通数组值，主要用于 DMA 场景 |
| 对数组元素逐元素选择 | `jnp.where` / `jax.lax.select` | 不改变程序控制流，只做数据选择 |

### Pallas kernel 中的控制流选择

JAX 的 tracing 规则在 Pallas kernel 中同样成立：依赖 `program_id`、`num_programs` 或 Ref 内容的条件属于运行时条件，应使用 JAX/Pallas 控制流，而不是 Python `if`。

相反，编译期分支依赖 Python 常量、静态参数、`Ref.shape` 中的块形状，以及由它们计算出来的值。它们会在 trace 时决定，适合用普通 Python `if` / `for` 改变生成出来的 kernel 形态。例如 TPU matmul kernel 中经常会根据 `K` 是否能被块大小整除选择不同代码路径：

```python
def kernel(x_ref, y_ref, o_ref, acc_ref, *, k: int):
    k_index = pl.program_id(2)
    bk = x_ref.shape[1]      # 块形状，trace 时已知
    divisible_k = k % bk == 0  # k 是静态参数，trace 时已知

    if divisible_k:
        # 这个 Python if 是可以的，因为条件是编译期常量
        acc_ref[...] += jnp.dot(x_ref[...], y_ref[...])
    else:
        # 这里如果还要判断 k_index 是否是最后一块，
        # 就进入 kernel 运行时控制流。
        ...
```

## Ref 与 JAX 可变数组

`Ref` 支持的操作：

| 操作 | 语义 | 示例 |
| :--- | :--- | :--- |
| `ref[...]` | 加载整个块 | `x = x_ref[...]` |
| `ref[i]` | 加载第 i 个切片 | `row = x_ref[0]` |
| `ref[pl.ds(start, size)]` | 动态切片 | `x_ref[pl.ds(i*128, 128)]` |
| `ref[...] = val` | 写入整个块 | `z_ref[...] = result` |
| `ref.at[idx]` | 获取子引用（不触发加载）| `sub = ref.at[0, pl.ds(0, 64)]` |

`ref.at[...]` 与 `ref[...]` 的区别：`ref.at[...]` 返回的仍然是一个 `Ref`（子引用），不触发实际的内存加载。这在 DMA 操作中非常重要——你需要传递一个 `Ref` 给 `make_async_copy`，而不是一个已加载的值。

讨论 ref_addupdate lowering 的不可能性。
