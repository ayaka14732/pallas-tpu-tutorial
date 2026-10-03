# 随机数的计时

前几节用指令数比较了不同的生成方法。本节用第三章的方法实测：`vrng` 自身的时序，以及在 kernel 中连续生成时，各种方法每个 TC VREG 要多少周期。

## vrng 的时序

本小节实验[源码](01_tpuasm_vrng_timing.py)、[输出](01_tpuasm_vrng_timing.txt)。

与第三章第 6 节相同，用 `LccProbe` 插入手写片段，`setup` 中先 `setrngseed`。读数 `(R1 − R0, R2 − R0)`：

| 片段 | 读数 | 结论 |
| --- | --- | --- |
| `vrng` 后紧接使用其结果的 `vadd` | (3, 15) | 没有等待：结果下一个周期就可用 |
| 连续 k 条 `vrng`，k = 1、2、4、8、16 | R2 = 14、21、37、69、133 | 每多一条加 8 个周期 |
| `va0` 与 `va1` 交替，k = 2、8 | R2 = 21、69 | 与全在 `va0` 相同 |
| k 组“`vrng` + 一条无关的 `vadd`”，k = 2、4 | R2 = 22、38 | `vadd` 不增加时间 |
| `vrng` 后紧接 `getrngseed` 或 `setrngseed` | (3, 21) | 同样等 8 个周期 |
| `setrngseed` 后紧接 `vrng` | (3, 23) | `vrng` 晚 10 个周期 |

`vrng` 的发射间隔是 8 个周期：一条 `vrng` 让每个生成器连续走 8 步（第 1 节），每个周期一步。两个槽共用同一个生成器，换到 `va1` 不能并行。间隔中的空闲周期可以发射其他向量运算。按第三章第 6 节的模型，它与 XLU 的转置相似：一个单元每 8 个周期接收一次，期间不挡住其他指令。

由此，硬件生成器的吞吐上限是每 8 个周期 4 KiB，即每周期 512 字节，按 1.05 GHz 约 540 GB/s。

## 在 kernel 中连续生成

本小节实验[源码](02_pallas_generation_rate.py)、[输出](02_pallas_generation_rate.txt)。

kernel 在 `pl.loop` 中每次生成 `[64,128]`（8 个 TC VREG），结果异或进同一个 buffer，使编译器不能删去任何一次生成：

```python
@pl.loop(0, chunks)
def _(chunk: jax.Array) -> None:
    acc_vmem[...] = acc_vmem[...] ^ generate(method, chunk)
```

用第三章第 1 节的 XProf kernel 时间，取 `chunks = 72` 与 `chunks = 8` 之差，除以多生成的 512 个 TC VREG，消去种子混合、DMA 等固定开销：

| 方法 | 每个 TC VREG | 吞吐 |
| --- | ---: | ---: |
| 硬件 `vrng` | 8.4 周期 | 514 GB/s |
| `vrng` → 均匀分布（`stateful_uniform`） | 8.6 周期 | 499 GB/s |
| `vrng` → 正态分布（`stateful_normal`） | 36.9 周期 | 117 GB/s |
| threefry2x32 | 119.8 周期 | 36 GB/s |

- `vrng` 达到了 8 周期的发射间隔，均匀分布的 4 条变换指令完全藏在间隔里。
- 正态分布约 70 条向量运算（第 5 节），两个向量 ALU 槽每周期各一条，约 35 个周期，与实测的 36.9 相符：这时瓶颈从生成器变成了变换。
- threefry2x32 每个 TC VREG 约 138 条向量运算，实测约 120 个周期，比 `vrng` 慢约 14 倍。

对比第二章第 2 节：HBM 读入一个 TC VREG 约需 4 个周期。所以用 `vrng` 在 kernel 内生成随机数，比从 HBM 读入预先生成的随机数慢一倍；但它不占 HBM 带宽，可以与其他读写并行。而 threefry2x32 的生成比读入慢约 30 倍，只在需要与 `jax.random` 逐位一致、或需要按位置寻址时才值得。

这些计时补充了[研究报告 57](../../../pallas-tpu-readings-dev/research_reports/57_tpu_v4_rng.md) 未测的时序部分。
