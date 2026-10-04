# 类型转换

上一节看到，bf16 在 TC VREG 中两个元素共用一个 32 bit 位置，算术却在 f32 中进行。本节逐一看各种类型转换在硬件上是哪几条指令、按什么规则舍入、怎样处理越界与特殊值：bf16 与 f32、f32 与 int32、int8 与 int32，最后比较 XLA 与 Pallas 写回 bf16 结果的两种方式。其中 f32 → int32 的指令带有一个 Mosaic 不使用的操作数，用 tpuasm 改变它，可以得到硬件的随机舍入。

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

这个 kernel 写在 [conversion_common.py](conversion_common.py) 中，本节第 4 个实验也用它。脚本对六组 (输入 dtype, 目标 dtype) 各编译一次，并与 XLA 的 `x.astype(dtype)` 逐元素比较。六组结果都与 XLA 完全一致。

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

bf16 就是 f32 的高 16 bit，所以这个方向不需要舍入：unpack 把 16 bit 放到高位、低位补 0。边界情况的实验（见下文“特殊值”）证实它逐位不变：非规格化数 `0x0001` 变为 `0x00010000`，带载荷的 NaN `0x7f81` 变为 `0x7f810000`，都原样保留。

## f32 → bf16：两次 load、一次 pack

反向转换把两个 f32 TC VREG 合成一个：

```text
vld     v0, [vmem:0x0]
vld     v1, [vmem:0x8]
vpackc.8x128.f32.f16 v2, v1, v0
vst     [vmem:0x10], v2
```

`vpackc` 同时完成舍入。f32 → bf16 会丢掉低 16 位尾数，舍入发生在这条指令中；它的结果与 XLA 逐元素一致。若某个算法要求与参考实现逐 bit 相同，必须把这次舍入放在与参考实现相同的位置。舍入的具体规则见下文“特殊值”。

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

结果是向零截断，与 C 语言和 `astype` 的语义相同，不是四舍五入。需要其他舍入方式时，先在 f32 中用 `jnp.round` 等函数处理，再转换。`f32.s32` 形式比反方向多一个操作数 `0xffffffff`，下一小节专门研究它。

## 只改 vcvt 的第三个操作数：硬件的随机舍入

本小节实验[源码](03_tpuasm_vcvt_rounding.py)、[输出](03_tpuasm_vcvt_rounding.txt)。

tpuasm 的指令索引中，`vcvt.8x128.f32.s32` 的第三个操作数可以是 TC VREG、标量寄存器或 32 位立即数，本节的清单中 Mosaic 写的是立即数 `0xffffffff`。实验用第三章第 3 节的 `LccProbe` 把手写的片段插进一个载体 kernel 执行（这里只用它执行片段、取回结果，不读周期计数器）：先用 `vimm` 把一个 f32 常数的位型广播成整个 TC VREG，再用 `vcvt` 转换，第三个操作数取不同的值 T，写回后读出：

```python
body += bundle(f'va0: vimm.8x128.s32 v13, 0x{as_bits(value):x}') + bundle(f'va0: vcvt.8x128.f32.s32 v14, v13, {operand}') + bundle(f'vst: vst.8x128 [vmem:0x{tile * 8:x}], v14')
```

实验只改 `operand`：

| T | 0.25 | 0.5 | 0.75 | −0.25 | −0.5 | −0.75 | 2.5 | −2.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0xffffffff` | 0 | 0 | 0 | 0 | 0 | 0 | 2 | −2 |
| `0xc0000000` | 0 | 0 | 0 | 0 | 0 | 0 | 2 | −2 |
| `0xbfffffff` | 0 | 0 | 1 | 0 | 0 | −1 | 2 | −2 |
| `0x80000000` | 0 | 0 | 1 | 0 | 0 | −1 | 2 | −2 |
| `0x7fffffff` | 0 | 1 | 1 | 0 | −1 | −1 | 3 | −3 |
| `0x40000000` | 0 | 1 | 1 | 0 | −1 | −1 | 3 | −3 |
| `0x3fffffff` | 1 | 1 | 1 | −1 | −1 | −1 | 3 | −3 |
| `0x00000000` | 1 | 1 | 1 | −1 | −1 | −1 | 3 | −3 |

把 |x| 的小数部分写成 32 位定点数 F：0.25 是 `0x40000000`，0.5 是 `0x80000000`，0.75 是 `0xc0000000`。表中的 64 个结果全部符合一条规则：**F > T 时向远离零的方向进 1，否则向零截断。** 正负数对称，进位的边界是严格的大于：0.5 在 T = `0x80000000` 时不进位，在 `0x7fffffff` 时进位。T = `0xffffffff` 时没有任何 F 能大于它，所以 Mosaic 写的这个值就是向零截断。

第三个操作数换成 TC VREG 时，每个元素用各自的 T。载体在片段之前已把一个随机的 `u32` tile 读进 `v10`，以它为 T：

| x | 小数部分 | 1024 个元素中进 1 的个数 | 比例 |
| ---: | ---: | ---: | ---: |
| 2.25 | 0.25 | 236 | 0.230 |
| −2.25 | 0.25 | 236 | 0.230 |
| 0.75 | 0.75 | 755 | 0.737 |
| 0.5 | 0.5 | 476 | 0.465 |
| 1.125 | 0.125 | 123 | 0.120 |
| −7.875 | 0.875 | 885 | 0.864 |

每个元素都与上面的规则逐一吻合。T 在 32 位整数上均匀分布时，F > T 的概率恰好是 F / 2³²，即 x 的小数部分：进位的期望恰好补上截断丢掉的部分，舍入后的期望等于 x。这就是随机舍入（stochastic rounding），硬件在一条 `vcvt` 中完成，随机比特由第三个操作数提供。

> 暂且可以理解为：一个 TC VREG 的均匀随机 `u32` 可以由硬件随机数生成器的一条指令产生。第四章第 1 节详细介绍这条指令，第四章第 5 节介绍 Pallas 中 f32 → bf16 的随机舍入怎样手写。

## 特殊值：越界、NaN、非规格化数

本小节实验[源码](04_pallas_special_values.py)、[输出](04_pallas_special_values.txt)。

第一个实验的输入都是普通的数。本实验沿用同一个 kernel，只换输入：每种转换在第 0 行放入一组边界值，结果与 XLA 的 `astype` 按位比较。四组都与 XLA 按位一致；下面是硬件的具体行为。

**f32 → int32 饱和。** 超出 int32 范围的值不回绕，而是取最近的端点；NaN 变为最大值：

```text
3000000000 → 2147483647      -3000000000 → -2147483648
inf → 2147483647             -inf → -2147483648
nan → 2147483647
2147483520 → 2147483520      2147483648 → 2147483647
```

2147483520 是小于 2³¹ 的最大 f32，原样转换；2³¹ 本身比 int32 的最大值大 1，饱和为 2147483647。

**f32 → bf16 就近舍入、偶数优先。** 低 16 bit 恰为 `0x8000`（恰在两个 bf16 中间）时，`0x3f808000` 舍为 `0x3f80`，`0x3f818000` 进为 `0x3f82`：都取末位为偶数的一边。略大于或略小于中间的值按距离舍入。最大的有限数 `0x7f7fffff` 舍入后溢出为无穷大 `0x7f80`。

**f32 → bf16 中的 NaN 与非规格化数。** 这两处与 IEEE 的直接推论不同：

| f32 | 硬件 | IEEE 就近舍入（保留符号、载荷与非规格化数） |
| --- | --- | --- |
| `0x7fc00001`（带载荷的 NaN） | `0x7fc0` | `0x7fc0` |
| `0xff800001`（负号的 NaN） | `0x7fc0` | `0xffc0` |
| `0x007fffff`（最大的非规格化数） | `0x0000` | `0x0080` |
| `0x807fffff` | `0x8000` | `0x8080` |

所有 NaN 都变成同一个 `0x7fc0`，符号与载荷都不保留；f32 的非规格化数先被当作 0（保留符号），所以 `0x007fffff` 不会像 IEEE 那样进位成最小的规格化数 `0x0080`。反方向 bf16 → f32 则逐位保留，见上文。

**int32 → f32 就近舍入、偶数优先。** 2²⁴ 以上的整数不能被 f32 精确表示：16777217（2²⁴ + 1）舍为 16777216，16777219 进为 16777220，都取偶数一边；2147483647 舍入为 2147483648。

## int8 ↔ int32：没有专用指令

上一节看到，int8 不能直接做向量运算，要先转换成 int32。TPU v4 没有 int8 的 unpack 指令，编译器用读入时的 sublane 复制、移位拼出转换。要读懂这段清单，先要知道 int8 的打包格式。

本小节实验[源码](05_tpuasm_int8_unpack.py)、[输出](05_tpuasm_int8_unpack.txt)。

**int8 的打包格式。** 与第 4 节读 bf16 格式的方法相同，把第 r 行全为 r 的 `int8[32,128]` 用 `pltpu.bitcast` 按位读成 `u32[8,128]`：

```text
sublane 0：字节 0–3 = 第 [0, 1, 2, 3] 行
sublane 1：字节 0–3 = 第 [4, 5, 6, 7] 行
...
sublane 7：字节 0–3 = 第 [28, 29, 30, 31] 行
```

一个 TC VREG 装 32 行：第 s 个 sublane 的 32 bit 依次装第 4s、4s+1、4s+2、4s+3 行，第 4s 行在最低字节。它与 bf16 的 `vpackc` 格式是同一个规律：相邻的行放进同一个 32 bit 位置，行号小的在低位。

**int8 → int32 的清单。** `int8[32,128]` 是一个打包的 TC VREG，转换成 4 个 int32 TC VREG：

```text
vld     v0, [vmem:0x0]                         # 读入打包的 int8
vlaneseq.8x128.u32 v1                          # 计算移位量，只做一次
vshrl.8x128.s32 v2, v1, 0x4
vand.8x128.u32 v3, 0x18, v2
vsub.8x128.s32 v4, 24, v3
vst     [vmem:0x28], v0                        # 把输入写到一块临时区域
vld.sshfl.8x128 v5, [vmem:0x28], 0x11110000    # 第 0 个输出 tile：读第 0、1 个 sublane，各复制 4 份
vshll.8x128.s32 v6, v5, v4                     # 把目标字节移到最高 8 位
vshra.8x128.s32 v7, v6, 0x18                   # 算术右移 24 位，完成符号扩展
vld.sshfl.8x128 v8, [vmem:0x2a], 0x11110000    # 第 1 个输出 tile：读第 2、3 个 sublane
...
```

逐条用 tpuasm 单独执行，看每条做了什么：

- **`vld.sshfl` 读入时重排 sublane。** 立即数的 8 个十六进制位从低到高对应输出的第 0–7 个 sublane，每一位给出它从地址起的第几个 sublane 读：

  ```text
  地址 0x0，模式 0x76543210：[0, 1, 2, 3, 4, 5, 6, 7]   # 原样读入
  地址 0x0，模式 0x01234567：[7, 6, 5, 4, 3, 2, 1, 0]   # 倒序
  地址 0x0，模式 0x11110000：[0, 0, 0, 0, 1, 1, 1, 1]
  地址 0x2，模式 0x11110000：[2, 2, 2, 2, 3, 3, 3, 3]
  ```

  `vld.sshfl` 只能从 TC VMEM 读，所以编译器先把 `v0` 写到临时区域 `0x28`，再从 `0x28`、`0x2a`、`0x2c`、`0x2e` 读，每次取两个 sublane、各复制 4 份。清单中每次读之前都把同一个 `v0` 重写一遍，4 次写入的是同一个值，后 3 次是多余的。
- **移位量。** `vlaneseq` 给出每个元素的序号 `s × 128 + lane`；右移 4 位再与 `0x18` 相与，得到 `(s mod 4) × 8`；24 减去它，第 s 个 sublane 的左移量是 `[24, 16, 8, 0, 24, 16, 8, 0]`。
- **左移再算术右移。** 第 0 个输出 tile 的第 s 个 sublane 装的是打包 TC VREG 第 ⌊s/4⌋ 个 sublane 的 32 bit，其中第 `s mod 4` 个字节就是第 s 行。左移 `24 − 8 × (s mod 4)` 位把这个字节移到最高 8 位，算术右移 24 位把它放回最低 8 位，同时用符号位填满高 24 位：第 s 行的 int8 变成了 int32。

每个输出 tile 重复一组 `vld.sshfl`、`vshll`、`vshra`，共 4 组。

**int32 → int8。** 反方向同样没有专用指令。`int32[32,128]` 转 int8 共用了 8 条 `vpackc`、8 条 `vand`、4 条 `vor`、8 条 `vshll`，每个输入 TC VREG 一组：

```text
vshll.8x128.s32 v4, v0, 0x18                 # 每个元素的低 8 位移到最高 8 位，低 24 位为 0
vpackc.8x128.f32.f16 v5, 0.0, v4             # 第一次打包
vand.8x128.u32 v9, 0xff00, v5
vand.8x128.u32 v10, 0xff000000, v5
vshll.8x128.s32 v11, v9, 0x8
vor.8x128.u32 v13, v10, v11
vpackc.8x128.f32.f16 v15, 0.0, v13           # 第二次打包
vst     [vmem:0x28, sm=3], v15               # 只写第 0、1 个 sublane
```

编译器借用 f32 → bf16 的打包指令来搬运字节。按第 4 节的 `vpackc` 格式推导：

1. 左移 24 位后，每个 32 bit 的最高字节是这一行的 int8，低 24 位为 0。把它当作 f32，指数的最低位总是 0，所以永远不是 NaN 或无穷大；字节为 0 或 `0x80` 时是 ±0；低 16 位全为 0，舍入不改变任何 bit。所以这次 `vpackc` 只是取出每个元素的高 16 bit，按格式打包：结果的第 s 个 sublane（s = 0–3）装第 2s 行（低 16 bit）和第 2s+1 行（高 16 bit），两行的 int8 分别在第 8–15 位和第 24–31 位；第 4–7 个 sublane 来自常数 0.0。
2. 两条 `vand` 与 `vshll`、`vor` 把第 2s 行的字节从第 8–15 位挪到第 16–23 位，于是第 2s、2s+1 行的字节占据了高 16 bit。
3. 第二次 `vpackc` 再取高 16 bit 打包：结果的第 t 个 sublane（t = 0、1）装输入第 2t、2t+1 个 sublane 的高 16 bit，四个字节依次是第 4t、4t+1、4t+2、4t+3 行。这正是 int8 的打包格式。

一个 int32 TC VREG（8 行）只填满 int8 打包格式的 2 个 sublane，所以用 `sm=3`（二进制 `11`）只写第 0、1 个 sublane，4 个输入依次写到 `0x28`、`0x2a`、`0x2c`、`0x2e`，拼成一个完整的 int8 TC VREG，最后整体读出、写到输出。

可见 TPU v4 的向量单元以 32 bit 和 16 bit 为基本粒度，int8 只是一种存储格式。它的用处在于节省 HBM 和 TC VMEM 的容量与 DMA 带宽，以及送入矩阵单元（第 10 节），而不在向量运算。

## 只改实现入口：XLA 与 Pallas 写回 bf16 的方式

本小节 XLA 实验[源码](02_jax_bf16_scale_64x128.py)、[输出](02_jax_bf16_scale_64x128.txt)；Pallas 版本即上一节的[实验](../04_vector_layout/01_pallas_scale_64x128.py)。

`bf16[64,128] × 2` 在 XLA 和 Pallas 中都是 load、unpack、f32 乘法、pack、store，差别在 pack 和 store：

| | `vld` | unpack | `vmul` | `vpackc` | `vst` |
| --- | ---: | ---: | ---: | ---: | ---: |
| XLA | 4 | 8 | 8 | 8 | 8（`sm=15`） |
| Pallas | 4 | 8 | 8 | 4 | 4 |

XLA 的写法是：

```text
vpackc.8x128.f32.f16 v11, 0.0, v7
vst [vmem:0x20, sm=15], v11
```

每个 f32 TC VREG 单独与常数 0.0 打包，结果只占第 0–3 个 sublane，于是用 `sm=15`（二进制 `1111`）只写这 4 个 sublane。Pallas 则把相邻两个 f32 TC VREG 合成一个满的打包 TC VREG，一次写满 8 个 sublane。两者都正确；Pallas 的写法少一半 pack 和 store。

这个对照说明，编译器的默认写法不是唯一写法，指令少也不自动等于更快。但它确实给手写 kernel 提供了一个选择：当一对 f32 结果同时可用时，合成一次满写回。
