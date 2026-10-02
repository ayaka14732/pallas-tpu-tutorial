# 类型转换

上一节看到，bf16 在 TC VREG 中两个元素共用一个 32 bit 位置，算术却在 f32 中进行。本节逐一看各种类型转换在硬件上是哪几条指令：bf16 与 f32、f32 与 int32、int8 与 int32，最后比较 XLA 与 Pallas 写回 bf16 结果的两种方式。

## 写法：kernel 中的 astype

本小节实验[源码](01_pallas_conversions.py)、[输出](01_pallas_conversions.txt)。

相对于第 1 节的最小 kernel，输出的 dtype 与输入不同，所以 `out_type` 和输出 buffer 改用目标 dtype，计算改为 `astype`：

```python
@pl.kernel(
    out_type=jax.ShapeDtypeStruct(x.shape, dtype),
    mesh=tc_mesh,
    scratch_types=(pltpu.VMEM(x.shape, x.dtype), pltpu.VMEM(x.shape, dtype), pltpu.SemaphoreType.DMA),
    ...
)
def kernel(x_hbm: Ref, o_hbm: Ref, x_vmem: Ref, o_vmem: Ref, sem: Ref) -> None:
    pltpu.async_copy(x_hbm, x_vmem, sem).wait()
    o_vmem[...] = x_vmem[...].astype(dtype)
    pltpu.async_copy(o_vmem, o_hbm, sem).wait()
```

脚本对六组 (输入 dtype, 目标 dtype) 各编译一次，并与 XLA 的 `x.astype(dtype)` 逐元素比较。六组结果都与 XLA 完全一致。

## bf16 → f32：一次 load、两次 unpack

`bf16[16,128]` 正好是一个打包的 TC VREG：

```text
vld     v0, [vmem:0x0]
vunpackl.8x128.f16.f32 v1, v0     # 第 0–7 行
vunpacku.8x128.f16.f32 v2, v0     # 第 8–15 行
vst     [vmem:0x8], v1
vst     [vmem:0x10], v2
```

两条 unpack 在同一个 bundle 中，分别占 `va0` 和 `va1`。元素数不变，每个元素从 16 bit 变成 32 bit，所以 TC VREG 数从 1 变成 2。若没有边界处理，N 个打包的输入 tile 对应 N 条 `vunpackl`、N 条 `vunpacku` 和 2N 个 f32 tile。

## f32 → bf16：两次 load、一次 pack

反向转换把两个 f32 TC VREG 合成一个：

```text
vld     v0, [vmem:0x0]
vld     v1, [vmem:0x8]
vpackc.8x128.f32.f16 v2, v1, v0
vst     [vmem:0x10], v2
```

`vpackc` 同时完成舍入。f32 → bf16 会丢掉低 16 位尾数，舍入发生在这条指令中；它的结果与 XLA 逐元素一致。若某个算法要求与参考实现逐 bit 相同，必须把这次舍入放在与参考实现相同的位置。

## f32 ↔ int32：一条 vcvt

```text
vcvt.8x128.f32.s32 v1, v0, 0xffffffff   # f32 → int32
vcvt.8x128.s32.f32 v1, v0               # int32 → f32
```

各一条指令。f32 → int32 的舍入方式由实验输入中恰在两整数中间的值确定：

```text
f32   ： [0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 2.7, -2.7]
int32 ： [0,   1,   2,   0,    -1,   -2,   2,   -2]
```

结果是向零截断，与 C 语言和 `astype` 的语义相同，不是四舍五入。需要其他舍入方式时，先在 f32 中用 `jnp.round` 等函数处理，再转换。`f32.s32` 形式的第三个操作数 `0xffffffff` 在本实验中保持不变，本节不推断它的含义。

## int8 ↔ int32：没有专用指令

上一节看到，int8 不能直接做向量运算，要先转换成 int32。`int8[32,128]` 是一个装了 4 个 int8 的打包 TC VREG，转换成 4 个 int32 TC VREG。TPU v4 没有 int8 的 unpack 指令，编译器的做法是：

```text
vld.sshfl.8x128 v5, [vmem:0x28], 0x11110000   # 按指定方式复制 sublane 读入
vshll.8x128.s32 v6, v5, v4                    # 把目标字节移到最高 8 位
vshra.8x128.s32 v7, v6, 0x18                  # 算术右移 24 位，完成符号扩展
```

每个输出 tile 重复一组，共 4 次 `vld.sshfl`、4 次 `vshll`、4 次 `vshra`。左移量 `v4` 随 sublane 变化，由 `vlaneseq` 等指令事先算出，用来选出每个 sublane 该取的那一个字节。

反方向 int32 → int8 同样没有专用指令，编译器用 `vshll`、`vand`、`vor` 拼出字节，借用 16 bit 的 `vpackc` 两两合并，最后用 `sm=3` 的部分写回。`int32[32,128]` 转 int8 共用了 8 条 `vpackc`、8 条 `vand`、4 条 `vor`、8 条 `vshll`。

可见 TPU v4 的向量单元以 32 bit 和 16 bit 为基本粒度，int8 只是一种存储格式。它的用处在于节省 HBM 和 TC VMEM 的容量与 DMA 带宽，以及送入矩阵单元（第 10 节），而不在向量运算。

## 只改实现入口：XLA 与 Pallas 写回 bf16 的方式

本小节 XLA 实验[源码](02_jax_bf16_scale_64x128.py)、[输出](02_jax_bf16_scale_64x128.txt)；Pallas 版本即上一节的[实验](../04_vector_layout/01_pallas_scale_64x128.py)。

`bf16[64,128] × 2` 在 XLA 和 Pallas 中都是 load、unpack、f32 乘法、pack、store，差别在 pack 和 store。XLA 把 64 行分给两个 TensorCore，下表把 Pallas 的计数也折算到 32 行，便于对照：

| 每 32 行 | `vld` | unpack | `vmul` | `vpackc` | `vst` |
| --- | ---: | ---: | ---: | ---: | ---: |
| XLA | 2 | 4 | 4 | 4 | 4（`sm=15`） |
| Pallas | 2 | 4 | 4 | 2 | 2 |

XLA 的写法是：

```text
vpackc.8x128.f32.f16 v8, 0.0, v6
vst [vmem:0x10, sm=15], v8
```

每个 f32 TC VREG 单独与常数 0.0 打包，结果只占第 0–3 个 sublane，于是用 `sm=15`（二进制 `1111`）只写这 4 个 sublane。Pallas 则把相邻两个 f32 TC VREG 合成一个满的打包 TC VREG，一次写满 8 个 sublane。两者都正确；Pallas 的写法少一半 pack 和 store。

这个对照说明，编译器的默认写法不是唯一写法，指令少也不自动等于更快。但它确实给手写 kernel 提供了一个选择：当一对 f32 结果同时可用时，合成一次满写回。
