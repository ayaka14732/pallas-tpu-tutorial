从零开始学习 JAX Pallas TPU Kernel 开发
=========================================

本教程面向有 JAX 使用经验、但没有底层 kernel 开发经验的开发者。
你将从最基础的概念出发，逐步掌握在 TPU 上使用 Pallas 编写高性能自定义算子的能力。

.. note::
   Pallas 仍处于实验阶段，API 可能会发生变化。但如果编译器接受了你的 kernel，
   它就 **必须** 返回正确的结果。

前置要求
--------

- 熟悉 JAX 基本用法（``jax.jit``、``jax.vmap``、``jax.numpy``）
- 了解 Python / NumPy 编程
- 不需要 CUDA 或 C++ 经验

.. toctree::
   :caption: 第一部分：基础概念
   :maxdepth: 2

   chapters/01_tpu_architecture
   chapters/02_pallas_hello_world
   chapters/03_grid_and_blockspec
   chapters/04_memory_spaces

.. toctree::
   :caption: 第二部分：流水线与性能优化
   :maxdepth: 2

   chapters/05_pipelining
   chapters/06_matmul
   chapters/07_profiling

.. toctree::
   :caption: 第三部分：经典算子实现
   :maxdepth: 2

   chapters/08_rmsnorm
   chapters/09_softmax
   chapters/10_scalar_prefetch_and_sparse

.. toctree::
   :caption: 第四部分：进阶主题
   :maxdepth: 2

   chapters/11_flash_attention
   chapters/12_distributed
   chapters/13_ragged_paged_attention
