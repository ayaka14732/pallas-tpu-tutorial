# 硬件随机数生成器

TensorCore 内部有一个随机数生成器，用三条向量指令操作。本节不经过任何 Pallas 接口，直接用 tpuasm 执行这三条指令，弄清生成器有多少状态、状态怎样排布在 TC VREG 中、每条指令怎样推进状态，并与主机上的模型逐位比较。后面各节的 Pallas 接口、状态管理和计时都以这里的结论为基础。

## 三条指令

```text
setrngseed vN          # 用 TC VREG vN 中的值设定生成器的状态
getrngseed vN          # 把生成器的状态读进 vN，不改变状态
vrng.8x128.u32 vN      # 生成一个 u32[8,128] 写进 vN，并推进状态
```

三条都可以在 `va0` 或 `va1` 槽中发射（[tpuasm 指令索引](../../../tpuasm/docs/references/tpu_v4_tc_isa.md)），与向量加法、乘法等共用两个向量 ALU 槽。`vrng` 的结果直接写进 TC VREG，不像 EUP 或 MXU 那样经过一个结果队列再 `vpop`。

## 实验的写法

本小节实验[源码](01_tpuasm_xorshift.py)、[输出](01_tpuasm_xorshift.txt)。

实验沿用第三章的 `LccProbe` 载体，但不读计数器：插入的片段自己用 `vst` 把结果写进 TC VMEM 的第 1–8 个 tile，`run_tiles` 把这 8 个 tile 随载体的输出 DMA 取回主机。片段之前，载体已经把输入的第一个 tile 读进 `v10`；`LccProbe.host` 是这个输入在主机上的副本，用作种子：

```python
def body(slot: str, seed: str) -> str:
    text = bundle(f'{slot}: setrngseed {seed}') + GAP
    for tile, instruction in enumerate(('getrngseed v11', 'vrng.8x128.u32 v11', 'getrngseed v11', 'vrng.8x128.u32 v11', 'getrngseed v11'), 1):
        text += bundle(f'{slot}: {instruction}') + GAP + store('v11', tile) + GAP
    return text
```

装入种子之后依次：读状态 → tile 1，生成 → tile 2，读状态 → tile 3，生成 → tile 4，读状态 → tile 5。`GAP` 是 16 个 `vnop`，让每条指令的结果都确定写完；本节只看指令的作用，时序在第 6 节测量。实验依次改变三个条件：发射的槽（`slot`）、种子的内容、只让一个生成器的状态非零。

## 主机上的模型

主机模型 [`rng_oracle.py`](../../rng_oracle.py) 按[研究报告 57](../../../pallas-tpu-readings-dev/research_reports/57_tpu_v4_rng.md) 的结论写成，有三个函数：

- `state_from_tile(tile)`：`setrngseed` 从一个 `u32[8,128]` 中取出的状态；
- `state_view(s0, s1)`：`getrngseed` 写出的 `u32[8,128]`；
- `vrng(s0, s1)`：一条 `vrng` 的输出和推进后的状态。

`vrng` 的核心：

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

这是 xorshift128+ 生成器：每个生成器有两个 64 位的状态字 `s0`、`s1`，每一步由移位与异或得到新状态，输出新旧两个 `s1` 之和（全部运算按 64 位无符号整数截断）。`s0`、`s1` 在这里都是 64 个元素的数组，一次计算 64 个生成器。

## 结果：逐位一致

```text
## va0：状态取自输入的前两个 sublane
  getrngseed：与模型一致的 word 1024 / 1024
  第 1 条 vrng：与模型一致的 word 1024 / 1024
  getrngseed：与模型一致的 word 1024 / 1024
  第 2 条 vrng：与模型一致的 word 1024 / 1024
  getrngseed：与模型一致的 word 1024 / 1024
  两次运行逐位相同：True
```

`va1` 的结果完全相同。两次生成和三次读出的状态，全部 1024 个 word 都与模型一致；同一个种子运行两次，输出逐位相同。

## 状态的排布

### 64 个生成器，每个占两个 lane

只改种子：把输入清零，只保留 lane 10、11 的 sublane 0、1，即只让第 5 个生成器的状态非零：

```python
host = np.zeros_like(probe.host)
host[:2, 10:12] = probe.host[:2, 10:12]
probe.x = jax.device_put(host, jax.local_devices()[0])
```

```text
第 1 条 vrng 输出中非零的 lane：[10, 11]；各 sublane 都非零：True
sublane 0：0x6aa1f3bb621e99e6，第 1 步 0x6aa1f3bb621e99e6
sublane 1：0x2ef54d94e9c25bcc，第 2 步 0x2ef54d94e9c25bcc
...
sublane 7：0x6d679074c74708f7，第 8 步 0x6d679074c74708f7
```

输出中只有 lane 10、11 非零，其余 126 个 lane 都是 0：其他生成器的状态全为 0，输出也是 0（见下文）。每个 sublane 中，lane 10 是低 32 位、lane 11 是高 32 位，拼成的 64 位数正好是主机模型逐步推进第 5 个生成器时第 1 到第 8 步的输出。由此：

| 位置 | 含义 |
| --- | --- |
| lane 2k、2k + 1 | 第 k 个生成器（k = 0–63） |
| `setrngseed` 的 sublane 0、1 | 状态字的低 32 位、高 32 位；lane 2k 是 `s0`，lane 2k + 1 是 `s1` |
| `vrng` 输出的 sublane i | 每个生成器第 i + 1 步的 64 位输出，低 32 位在 lane 2k，高 32 位在 lane 2k + 1 |

一条 `vrng` 让 64 个生成器各走 8 步，8 个 sublane 是同一个生成器的连续 8 步，而不是 8 个独立的生成器。全部状态是 64 × 2 × 64 位 = 1 KiB，正好是一个 TC VREG 的两个 sublane。

### setrngseed 只读前两个 sublane

`setrngseed` 忽略 sublane 2–7。`getrngseed` 必须写满一个 TC VREG，就把这两个 sublane 重复 4 次：

```text
读回的 sublane 0、1 与输入相同：True；sublane 2–7 是 0、1 的重复：True
```

所以对 `getrngseed` 的结果再 `setrngseed`，得到的是同一个状态。第 2 节用这一点保存与恢复状态。

## 全零状态

把状态全部设为 0（`vxor v15, v10, v10` 得到全零的 TC VREG，再 `setrngseed v15`）：

```text
两条 vrng 的输出全为 0：True
```

xorshift 的每一步都是移位和异或，全零状态的下一步仍是全零，生成器永远输出 0。上面只让一个生成器非零的实验也说明，每个生成器的状态彼此独立：一个生成器的状态不影响其他 lane。所以直接装入的状态里，每一对 lane 都不能全为 0。第 3 节会看到，Pallas 的 `prng_seed(0)` 不会出现这种情况，因为它不是把 0 直接装进状态，而是先把种子与 lane 编号混合。

## 小结

- 硬件生成器是 64 个并行的 xorshift128+，每次 `vrng` 产生 4 KiB 随机比特，每个生成器走 8 步。
- 状态完全由 `setrngseed` 的前两个 sublane 决定，同样的状态产生同样的序列；`getrngseed` 读出状态而不改变它。
- 每个生成器的状态独立；全零的生成器永远输出 0。
- xorshift128+ 是快速的统计生成器，不是密码学安全的随机数源；本章也不讨论它的统计质量。

研究报告 57 还在本 host 四颗芯片的 `va0`、`va1` 上做了同样的检查，结论相同。
