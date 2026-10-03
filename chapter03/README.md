# 第三章：静态分析与计时

前两章研究了每种操作由哪些指令完成、数据经过哪些通路，引用的时延都来自实测。本章讲这些数字是怎样测出来的，以及怎样从清单预先估算执行时间。

时间可以在三个层次上测：主机上的调用时间，XProf 中设备上的事件，kernel 内部的周期计数器。每个层次包含的东西不同，适用的精度也不同。本章从最粗的主机计时开始，逐步深入到单个周期，最后把各层次的结论整理成一个可以从清单推算周期数的发射模型。

读完本章，应当能够：

- 说清一个“时间”从哪里开始、到哪里结束，包含哪些与 kernel 无关的开销；
- 公平地比较两个实现，并从编译结果判断两边是否做了同一件事；
- 用 XProf 的区域和 kernel 内部的 LCC 测量一段代码，知道各自的边界在哪里；
- 理解 TensorCore 的两级发射，判断何时需要 `sfence`；
- 从清单推算一段代码的周期数，并用 LCC 验证；
- 用 GTC 比较不同 TensorCore、不同芯片上的事件。

## 本章目录

- [01 · 五种时间](01_five_times/README.md)
- [02 · 公平计时](02_fair_timing/README.md)
- [03 · XProf 与 vtrace](03_xprof_vtrace/README.md)
- [04 · LCC：指令级计时](04_lcc/README.md)
- [05 · sfence 与 VIF](05_sfence_vif/README.md)
- [06 · 发射模型与静态分析](06_issue_model/README.md)
- [07 · GTC 与多个 TensorCore 的计时](07_gtc/README.md)

## 本章的工具

第 4–7 节的实验都用 [`tpuasm_tools.LccProbe`](../tpuasm_tools.py)：用 Pallas 编译一个载体 kernel，再用 tpuasm 在其中插入手写的 bundle，读出 LCC 或 GTC。插入的片段不经过编译器，设备执行的就是写下的指令。第 1–3 节用 [`xprof_tools.py`](../xprof_tools.py) 采集并读取 XProf 的设备事件。

## 本章得到的发射参数

| 参数 | 值 | 位置 |
| --- | --- | --- |
| 标量发射 | 每周期一个 bundle | 第 4 节 |
| `sfence` 的代价（VIF 为空时） | 11 个周期 | 第 4、5 节 |
| VIF 容量 / 释放延迟 | 20 项 / 向量发射后 10 个周期 | 第 5、6 节 |
| `vmul.f32` 结果可用 | 2 个周期 | 第 5 节 |
| EUP 发射 → 取回 / 发射间隔 | 7 / 2 个周期 | 第 6 节 |
| MXU `vmatmul` → 取回 / 发射间隔 | 83（`mrf0`、`mrf1`）/ 8 个周期 | 第 6 节 |
| XLU 转置：提交、取回间隔 | 各 8 个周期 | 第 6 节 |
| CMEM `cld` → `vpop` | 约 54 个周期 | 第 5、6 节 |
| `vpush` → `spop` | 约 43 个周期 | 第 6 节 |
| LCC / GTC 频率 | 约 1.05 GHz / 约 11.2 GHz，比例 3 : 32 | 第 7 节 |
