# 硬件随机数生成器

TensorCore 内部有一个随机数生成器，用三条向量指令操作。本节不经过任何 Pallas 接口，直接用 tpuasm 执行这三条指令，弄清生成器有多少状态、状态怎样排布、每条指令怎样推进状态，并与主机上的模型逐位比较。后面各节的 Pallas 接口、状态管理和计时都以这里的结论为基础。

## 三条指令

```text
setrngseed vN          # 用 TC VREG vN 中的值设定生成器的状态
getrngseed vN          # 把生成器的状态读进 vN，不改变状态
vrng.8x128.u32 vN      # 生成一个 u32[8,128] 写进 vN，并推进状态
```

三条都在 `va0` 或 `va1` 槽中发射（[tpuasm 指令索引](../../../tpuasm/docs/references/tpu_v4_tc_isa.md)）。

## 实验：与 xorshift128+ 模型逐位比较

本小节实验[源码](01_tpuasm_xorshift.py)、[输出](01_tpuasm_xorshift.txt)。

实验沿用第三章的 `LccProbe` 载体，但不读计数器：片段自己把结果用 `vst` 写进 TC VMEM 的第 1–8 个 tile，`run_tiles` 把这 8 个 tile 取回主机：

```python
text = bundle(f'{slot}: setrngseed {seed}') + GAP
for tile, instruction in enumerate(('getrngseed v11', 'vrng.8x128.u32 v11', 'getrngseed v11', 'vrng.8x128.u32 v11', 'getrngseed v11'), 1):
    text += bundle(f'{slot}: {instruction}') + GAP + store('v11', tile) + GAP
```

`v10` 中是载体的随机输入，作为种子装入。每两条指令之间隔 16 个 `vnop`，本节只看指令的作用，时序在第 6 节测量。

主机上的模型 [`rng_oracle.py`](../../rng_oracle.py) 由研究报告 57 的结论写成：

```python
def vrng(s0, s1):
    for sublane in range(8):
        x, y = s0, s1
        s0 = y
        x = x ^ (x << 23)
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26)
        z = s1 + y
        out[sublane, 0::2] = low32(z)
        out[sublane, 1::2] = high32(z)
```

这是 xorshift128+ 生成器（全部运算按 64 位无符号整数截断）。每个生成器有两个 64 位的状态字 `s0`、`s1`。

结果：`va0` 与 `va1` 两个槽中，三次读出的状态和两次生成的 tile 全部 1024 个 word 都与模型一致，两次运行逐位相同。

## 状态的排布

模型中的两个函数说明了状态怎样放进 TC VREG：

- **64 个生成器，每个占两个 lane。** 第 k 个生成器的 `s0` 对应 lane 2k，`s1` 对应 lane 2k + 1。
- **`setrngseed` 只读前两个 sublane。** sublane 0 是两个状态字的低 32 位，sublane 1 是高 32 位；sublane 2–7 被忽略。全部状态是 64 × 2 × 64 位 = 1 KiB，正好是一个 TC VREG 的两个 sublane。
- **`getrngseed` 把这两个 sublane 重复 4 次。** 输出中 sublane 2–7 是 sublane 0、1 的重复：

```text
读回的 sublane 0、1 与输入相同：True；sublane 2–7 是 0、1 的重复：True
```

- **`vrng` 的 8 个 sublane 是同一个生成器的连续 8 步。** sublane i 是第 i 步的 64 位输出，低 32 位在 lane 2k、高 32 位在 lane 2k + 1。8 个 sublane 不是 8 个独立的生成器；一条 `vrng` 让每个生成器前进 8 步。

## 全零状态

把状态全部设为 0（`vxor v15, v10, v10` 得到全零的 TC VREG，再 `setrngseed v15`）：

```text
两条 vrng 的输出全为 0：True
```

xorshift 的每一步都是移位和异或，全零状态的下一步仍是全零，生成器永远输出 0。所以直接装入的状态不能全为 0；后面会看到，Pallas 的 `prng_seed(0)` 不会出现这种情况，因为它不是把 0 直接装进状态。

## 小结

- 硬件生成器是 64 个并行的 xorshift128+，每次 `vrng` 产生 4 KiB 随机比特。
- 状态完全由 `setrngseed` 决定，同样的状态产生同样的序列；`getrngseed` 可以读出状态。
- xorshift128+ 是快速的统计生成器，不是密码学安全的随机数源；本章也不讨论它的统计质量。

这些结论与[研究报告 57](../../../pallas-tpu-readings-dev/research_reports/57_tpu_v4_rng.md) 一致，该报告还在本 host 四颗芯片的 `va0`、`va1` 上做了交叉检查。
