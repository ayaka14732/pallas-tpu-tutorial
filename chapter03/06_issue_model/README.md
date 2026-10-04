# 发射模型与静态分析

第 4、5 节得到了两级发射的定性规则。本节把它们整理成一个可以计算的模型：给定一段实际执行的 bundle 序列，逐个 bundle 推算它的标量发射和向量发射时刻，从而预测 LCC 读数。模型需要的参数只有几类：VIF 的容量和释放延迟、每种结果的可用延迟、每个单元的发射间隔。本节先测出 EUP、MXU、XLU 和向量到标量通路的参数，再用模型预测第 4、5 节和本节全部手写片段的读数，与真机比较。

## 模型

参数来自 [tpu-v4-latency-numbers 第 0 节](../../../tpu-v4-latency-numbers/README.md#0-测量方法与发射模型)，它用 `vdelay` 人为堵住向量发射，直接测出了 VIF 的容量与释放延迟；本节在此基础上补上各个“提交—取回”通路。规则如下：

1. **标量发射**按顺序进行，没有阻塞时每周期一个 bundle。标量指令（包括读 LCC）在此执行。
2. **VIF**：含向量或 DMA 指令的 bundle 标量发射后进入 VIF。一项在向量发射后 **10 个周期**才在标量一侧释放；未释放的项达到 **20 个**时，任何 bundle 都不能标量发射。
3. **向量发射**按顺序进行，每周期最多一个 bundle，最早在标量发射后 1 个周期；含访问内存的 `vld`、`vst`、`cld` 的 bundle 最早在 2 个周期后。此外要等计分板放行：源操作数就绪、对应的单元和队列允许。模型中的 `ready`、`queues`、`last` 三张表就是对计分板的模拟。
4. **`sfence`** 所在 bundle 在向量一侧占一个位置，下一 bundle 要等 VIF 中全部项释放之后才标量发射。

单元与队列的参数：

| 通路 | 提交 → 可以取回 | 相邻两条提交指令的发射间隔 | 相邻两条取回指令的发射间隔 | 实验 |
| --- | ---: | ---: | ---: | --- |
| 向量运算写 TC VREG | 1（`vmul.f32`、`vrot.slane.down` 为 2） | 1 | — | 第 4、5 节，本节 01 |
| EUP：`vpow2`、`vrcp`、`vrsqrt`、`vtanh`、`vlog2` → `erf` | 7 | 2 | 1 | 本节 01 |
| MXU0：`vmatmul` → `mrf0` | 83 | 8 | 1 | 本节 01 |
| XLU：最后一次 `vxpose` → `trf0` | 6 | 8 | 8 | 本节 01 |
| XLU：`vadd.xlane` → `trf0` | 79 | 8 | — | 本节 01 |
| XLU：lane 循环移位 `vrot`、重排 `vperm` → `trf0` | 69 | 8 | — | 本节 01 |
| CMEM：`cld` → `crf` | 53 | 2 | 2 | 第 5 节，第二章第 3 节 |
| 向量 → 标量：`vpush` → `v2sf` → `spop` | 42，`spop` 在标量一侧等待 | — | — | 本节 01 |

`mrf2`、`mrf3` 的结果比 `mrf0`、`mrf1` 晚 18 个周期，即 101（[研究报告 55](../../../pallas-tpu-readings-dev/research_reports/55_tpu_v4_mrf_latency.md)）。

模型写在 [issue_model.py](issue_model.py) 中，不到一百行。核心是每个 bundle 的向量发射时刻：

```python
issue = max(vector_free, scalar + 1 + any(decode(item)[0].startswith(MEMORY) for item in vector))
for item in vector:
    mnemonic, operands = decode(item)
    if mnemonic.startswith('vpop'):
        events.append(('pop', operands[1]))
        issue = max(issue, queues[operands[1]][0])          # 等结果进入队列
    elif operands and kind(operands[0]) in PUSH_LATENCY:
        events.append(('push', operands[0]))
    unit = next((prefix for prefix in UNIT_INTERVAL if mnemonic.startswith(prefix)), None)
    if unit:
        events.append(('unit', unit))
    issue = max([issue, *(ready.get(source, 0) for source in operands[1:])])   # 等源操作数
for event, queue in events:
    issue = max(issue, last.get(f'{event} {queue}', -99) + (UNIT_INTERVAL[queue] if event == 'unit' else INTERVAL.get(f'{event} {kind(queue)}', 1)))   # 等单元
```

XLU 的延迟表 `XLU_LATENCY` 中，`vmax.index.xlane` 与 `vmin.xlane` 两项由第一章第 14 节测得，都是 79 个周期；那一节用模型预测了 top-k 的时间。`UNIT_INTERVAL` 记录不经过队列、但同一单元相邻两条有最小间隔的指令。目前只有一项：第四章第 6 节测得的 `vrng`，每 8 个周期一条；那一节用它解释了几种随机数分布的生成速度。

模型的输入是**动态执行的** bundle 序列：循环要按实际迭代次数展开。模型不处理 DMA 和 `vwait`，它们的时间取决于数据通路，由第二章的代价模型给出。标量一侧加入了第 4 节测得的 `sld` 规则：

```python
if mnemonic == 'sld':
    scalar = max(scalar, last_sld + SLD_INTERVAL)       # 相邻两条 sld 至少相隔 4 个周期
# 读取尚未就绪的标量寄存器时，整个 bundle 等待。
scalar = max([scalar, *(scalar_ready.get(source, 0) for source in sources)])
```

`sld` 写入的寄存器在 4 个周期后就绪（`SLD_LATENCY`）；其余标量指令的结果都按下一周期可用处理。有一点与预期不同：紧接在 `sld` 之后的 `vmov.8x128 v11, s24` 读同一个寄存器，读数里看不到任何等待，模型不加等待时恰好与真机一致（下表的“`sld` 与依赖 `sld` 结果的向量运算”）。第 4 节的实验确认，此时 `vmov` 读到的就是 `sld` 载入的新值，而不是旧值；`vmov` 读标量操作数的时刻本节没有进一步确定。

## 测量通路的参数

本小节实验[源码](01_tpuasm_result_latency.py)、[输出](01_tpuasm_result_latency.txt)。

**延迟。** 在发射指令与取回指令之间插入 d − 1 个 `vnop`：

```python
body = read_lcc(20) + bundle(issue) + bundle('misc: vnop') * (distance - 1) + bundle(fetch) + END
```

实验只改 `issue` 和 `fetch` 两条指令。若取回要等结果，向量一侧停在取回处，R2 不随 d 变化；d 大到结果已经就绪时，R2 开始每次加 1。以 `vpow2` 为例：

| d | 1 | 2 | 4 | 6 | 7 | 8 | 12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R2 − R0 | 20 | 20 | 20 | 20 | 21 | 22 | 26 |

d = 1 时，`vpow2` 在第 2 个周期向量发射，`vpop` 若在 L 个周期后发射，后面的 `sfence` 再过 1 + 10 个周期释放，所以 R2 − R0 = 13 + L，L = 7。五种 EUP 函数的读数完全相同，延迟都是 7。MXU 的 `vmatmul` 到 `vpop mrf0` 在 d ≤ 12 时都是 96，L = 83；XLU 从最后一次 `vxpose` 到第一次取回是 19，L = 6。MXU 和 XLU 的准备工作（装入权重、提交前 15 个 TC VREG）放在 `setup` 中，不计入区间。

`vpush` 到 `spop` 不同：`spop` 在标量一侧，等待时挡住的是标量发射，所以 R1 也跟着变。d ≤ 42 时 R1 − R0 都是 45，d = 44 起不再等待：`spop` 最早在 `vpush` 标量发射后 43 个周期才能执行。

**发射间隔。** 连续发射 k 条提交指令，看每多一条 R2 加多少：

| 序列 | k = 1 | 2 | 4 | 8 | 16 | 每次 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| k 次 `vpow2`（读数之后才取回） | 14 | 15 | 19 | 27 | 43 | 2 |
| k 次 `vmatmul`，再取回 k 次 | 96 | 104 | 120 | 152 | 254 | 8 |
| 转置完成后取回 k 次 `trf0` | 14 | 21 | 37 | 69 | 133 | 8 |

16 次 `vxpose` 之间插入 0、1、2、4、8 个 `vnop`，R2 为 133、134、135、137、157：间隔不足 8 个周期时，插入的 `vnop` 不增加时间。

两种转置的变体（第一章第 9 节）：

| 序列 | R2 − R0 |
| --- | ---: |
| bf16 打包：8 次 `vxpose.0.packed` | 125 |
| bf16 打包：8 次提交 + 8 次取回 | 259 |
| 16 次宽度为 8 的提交 + 1 次取回 | 139 |

打包的提交每 16 个周期一次，8 次打包提交加 8 次取回与 f32 的 16 次加 16 次一样是 259 个周期：XLU 每个周期处理 TC VREG 的一行，f32 的 TC VREG 有 8 行，打包的 bf16 有 16 行。宽度为 8 的转置提交时间不变，只是取回从 16 次减为 1 次。本节的模型只按 8 个周期的间隔处理 f32 的提交，没有包含打包的情况。XLU 每 8 个周期接收一个 TC VREG、送出一个 TC VREG，一次 `128×128` 的转置提交 16 次、取回 16 次，各占 16 × 8 = 128 个周期。

## 用模型预测，再与真机比较

本小节实验[源码](02_tpuasm_model_check.py)、[输出](02_tpuasm_model_check.txt)。

脚本把每段片段同时交给模型和真机：

```python
measured = sorted({tuple(row) for row in probe.run(program, setup=setup).tolist()})
reads = issue_model.replay(issue_model.parse(trace))
predicted = (reads[21] - reads[20], reads[22] - reads[20])
```

44 段片段中 41 段的 R1 − R0、R2 − R0 与模型完全一致，包括 VIF 积压、挡住标量发射的情况，以及第 4 节的 `sld` 规则。例如 16 组 `cld` + `vpop`：模型和真机都是 (337, 877)。R1 = 337 的来历是：R1 前面有 32 个进入 VIF 的 bundle，R1 要等到只剩 19 项未释放，也就是第 13 项（第 7 条 `cld`）释放；它在第 327 个周期向量发射，第 337 个周期释放。

不一致的 3 段：

| 片段 | 模型 | 实测 |
| --- | --- | --- |
| 1 组 `cld` + `vpop` | (3, 67) | (3, 66)、(3, 67) |
| 12 条 `vmatmul` → 12 条 `vpop` | (44, 184) | (44, 190) |
| 16 条 `vmatmul` → 16 条 `vpop` | (108, 216) | (108, 254) |

第一段的读数在 66 与 67 之间变化，模型给出其中之一。后两段说明 MXU 有模型没有包含的限制：连续 11 条 `vmatmul` 之后才取回时与模型一致，从第 12 条起，每多一条要多 8 个周期，相当于结果送出的间隔从 8 变成了 16。同样是 16 条 `vmatmul`，从第 9 条起与 `vpop` 交错时又与模型一致。编译器生成的代码总是把 `vmatmul` 与 `vpop` 交错排列（第一章第 10 节），手写时也应如此。

## 用模型读清单

有了这些参数，第一、二章中几处“暂且可以理解为”可以给出确切的答案。

**循环。** 计数循环每次迭代是 3 个标量 bundle 加 1 个延迟槽，实测 R1 − R0 = 4N + 2，与模型相同：分支除了延迟槽之外没有额外代价。所以循环的执行周期就是“循环体 bundle 数（含延迟槽）× 迭代次数”，再加上循环体内的等待（第一章第 7 节）。

**EUP。** `exp` 的一个 TC VREG 是 `vmul`、`vpow2`、`vpop` 三条，但 `vpop` 要在 `vpow2` 之后 7 个周期才能发射。单独计算一个 TC VREG 时，这 7 个周期是空等；连续计算多个 TC VREG 时，`vpow2` 每 2 个周期可以发射一条，延迟被流水掩盖。所以 EUP 的吞吐是每 2 个周期一个 TC VREG：16 条 `vpow2` 后接 16 条 `vpop`，实测 R2 − R0 = 59，其中 16 条 `vpow2` 占 32 个周期，16 条 `vpop` 紧随其后每周期一条（第一章第 6 节）。

**MXU。** `vmatmul` 每 8 个周期接收一个 LHS TC VREG（8 行），第一个结果 83 个周期后才能取回。所以第一个结果可以取回时，MXU 已经接收了约 `83 / 8 ≈ 10` 个 TC VREG：要让 MXU 不停，需要约 10 条 `vmatmul` 同时在途，再以每 8 个周期一条的节奏交替发射 `vmatmul` 与 `vpop`（第一章第 10 节）。上一小节的反例也在这个数附近：在途超过 11 条而不取回时，MXU 变慢。

**XLU。** 一次 `128×128` 的 f32 转置，从第一次提交到最后一次取回之后的读数，R2 − R0 = 259 个周期。两个转置分给 `trf0` 和 `trf1`、每次把两边的 `vpop` 放进同一个 bundle 时，总时间仍是 259，与一个相同；若先取完 `trf0` 再取 `trf1`，要 380 个周期，因为每个队列都要每 8 个周期才取一次。两个 XLU 能并行，前提是提交和取回都成对排列（第一章第 9 节）。

**归约。** XLU 的跨 lane 归约（第一章第 11 节）也是提交—取回：`vadd.xlane` 之后 79 个周期才能取回（`d` 到 40 时 R2 − R0 仍是 92），连续的 `vadd.xlane` 每 8 个周期发射一条（k 条 `vadd.xlane` 加 k 次取回为 `84 + 8k`）。sublane 方向的归约用 `vrot.slane.down`，它在向量 ALU 中执行，结果 2 个周期后可用（N 条相互依赖时 R2 − R0 = 2N + 11）。所以对单个 TC VREG：沿 lane 求和要等约 80 个周期，沿 sublane 的二叉树是 7 条移位加 3 条加法、约 17 个周期；但沿 lane 的归约每 8 个周期可以发射一条，许多 TC VREG 一起归约时，XLU 的吞吐并不差，而且不占用向量 ALU。

**向量 → 标量。** `spop` 最早在 `vpush` 标量发射之后 43 个周期执行，期间标量一侧停止发射（第一章第 11 节）。把向量的计算结果变成循环次数或分支条件，每次都要付这个代价。

**CMEM。** `cld` 之后 53 个周期才能 `vpop`，`cld` 每 2 个周期可以发射一次。串行地“读一个、取一个”时每组恰好 54 个周期，提前发出多个 `cld` 才能接近每 2 个周期一个 TC VREG（第二章第 3 节）。

## 模型的边界

模型可以预测不含 DMA 的直线代码。对整个 kernel，还需要把 DMA 的代价（第二章第 2 节）和各 TensorCore 之间的等待加进来。[研究报告 49](../../../pallas-tpu-readings-dev/research_reports/49_static_timing_model.md) 讨论了把这样的模型扩展成读取真实清单的静态计时器还缺什么：DMA 与向量工作的组合、并发 DMA、完整的 MXU 规则。本节的参数可以作为这项工作的输入。
