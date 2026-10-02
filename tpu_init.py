"""控制 TPU runtime 启动哪些芯片；必须在 `import jax` 之前调用。

这里只决定 runtime 层面打开几颗芯片。程序在 SPMD 层面用几个 TensorCore，由 `jax.shard_map` 与 `pltpu.TensorCoreMesh` 决定。
"""
import os

def initialise_one_chip(chip: int = 0) -> None:
    """只打开本 host 的第 chip 颗芯片（0–3）；不同 chip 的进程可以同时运行。"""
    os.environ['TPU_CHIPS_PER_PROCESS_BOUNDS'] = '1,1,1'
    os.environ['TPU_PROCESS_BOUNDS'] = '1,1,1'
    os.environ['TPU_VISIBLE_CHIPS'] = str(chip)

def initialise_local_chips() -> None:
    """打开本 host 的全部四颗芯片（2x2x1）；在多 host 切片上也只用本 host，不调用 `jax.distributed.initialize()`。"""
    os.environ['TPU_CHIPS_PER_PROCESS_BOUNDS'] = '2,2,1'
    os.environ['TPU_PROCESS_BOUNDS'] = '1,1,1'
    os.environ['TPU_VISIBLE_CHIPS'] = '0,1,2,3'
