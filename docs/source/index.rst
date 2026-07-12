Pallas TPU Kernel 开发教程
==========================

.. toctree::
   :maxdepth: 2

   chapters/00_preface

.. toctree::
   :caption: 第一部分：简介
   :maxdepth: 2

   chapters/01_tpu_architecture
   chapters/02_environment_setup

.. toctree::
   :caption: 第二部分：Pallas TPU 基础用法
   :maxdepth: 2

   chapters/03_grid_and_blockspec
   chapters/04_memory_spaces
   chapters/05_pipelining

.. toctree::
   :caption: 第三部分：流水线与性能优化
   :maxdepth: 2

   chapters/06_matmul
   chapters/07_profiling

.. toctree::
   :caption: 第四部分：基本算子实现
   :maxdepth: 2

   chapters/08_rmsnorm
   chapters/09_softmax
   chapters/10_scalar_prefetch_and_sparse

.. toctree::
   :caption: 第五部分：注意力机制实现
   :maxdepth: 2

   chapters/11_flash_attention
   chapters/12_distributed
   chapters/13_ragged_paged_attention

.. toctree::
   :caption: 第六部分：附录
   :maxdepth: 2

   chapters/14_api_reference
   chapters/15_prng
