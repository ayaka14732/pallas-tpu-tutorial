# 第四章：随机数

随机数在 TPU 上有两种来源。一种是 TensorCore 内部的硬件生成器：一条指令产生 4 KiB 随机比特，但它是有状态的，第 n 个结果取决于此前生成过多少次。另一种是计数器式生成器：随机数是 key 与位置的函数，可以在任何地方精确复现，但要用几十到上百条向量整数运算算出来。本章从硬件生成器的指令开始，弄清它的状态、作用域和 Pallas 接口，再看计数器式生成器的代价、从比特到各种分布的变换，最后实测各自的速度。

本章只讲生成随机数的硬件能力与基本操作，不实现具体的采样算法。

读完本章，应当能够：

- 直接使用 `setrngseed`、`getrngseed`、`vrng`，并在主机上逐位复现硬件生成器的输出；
- 判断随机数能否复现、多个 TensorCore 与芯片的序列是否重复，并通过种子、状态保存与恢复控制它们；
- 在 `prng_seed`、Pallas key、`sample_block` 与 `jax.random` 之间做出选择；
- 用少量指令把随机比特变成均匀、伯努利、Gumbel 等分布，以及在 TPU v4 上实现随机舍入；
- 估算并实测一种生成方式每个 TC VREG 的周期数。

## 本章目录

- [01 · 硬件随机数生成器](01_hardware_prng/README.md)
- [02 · 状态的作用域与生命周期](02_state_scope/README.md)
- [03 · Pallas 的硬件随机数接口](03_pallas_prng/README.md)
- [04 · 计数器式生成器](04_counter_based/README.md)
- [05 · 从比特到分布](05_distributions/README.md)
- [06 · 随机数的计时](06_timing/README.md)

## 本章的结论

| 问题 | 结论 | 位置 |
| --- | --- | --- |
| 硬件生成器是什么 | 64 个 xorshift128+，每个占两个 lane；一条 `vrng` 每个生成器走 8 步 | 第 1 节 |
| 状态属于谁 | 每个 TensorCore 一份，kernel 结束后保留；程序开头的前导每次启动用 runtime 的值、芯片与核编号重新装入 | 第 2 节 |
| 能否保存与恢复 | `getrngseed` 不改变状态，读出的状态用 `setrngseed` 装回可精确重放 | 第 2 节 |
| `prng_seed` 是否区分 TensorCore | 不区分：必须把核、芯片编号写进种子 | 第 3 节 |
| 计数器式生成器的代价 | threefry2x32 每个 TC VREG 约 138 条向量运算，Philox 约 277 条 | 第 4 节 |
| 随机舍入 | `pltpu.stochastic_round` 在 TPU v4 上不能编译；用整数加法实现，3 条指令 | 第 5 节 |
| 速度 | `vrng` 每 8 个周期一个 TC VREG（约 514 GB/s），threefry2x32 约 120 个周期 | 第 6 节 |
