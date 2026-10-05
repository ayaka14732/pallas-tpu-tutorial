"""控制 TPU runtime 启动哪些芯片；必须在 `import jax` 之前调用。

这里只决定 runtime 层面打开几颗芯片。程序在 SPMD 层面用几个 TensorCore，由 `jax.shard_map` 与 `pltpu.TensorCoreMesh` 决定。
"""
import os

def initialize_one_chip(chip: int | None = None) -> None:
    """只打开本 host 的第 chip 颗芯片（0–3）；不同 chip 的进程可以同时运行。省略 chip 时读环境变量 TPU_TUTORIAL_CHIP，默认 0。"""
    if chip is None:
        chip = int(os.environ.get('TPU_TUTORIAL_CHIP', '0'))
    os.environ['TPU_CHIPS_PER_PROCESS_BOUNDS'] = '1,1,1'
    os.environ['TPU_PROCESS_BOUNDS'] = '1,1,1'
    os.environ['TPU_VISIBLE_CHIPS'] = str(chip)

def initialize_one_core(chip: int | None = None) -> None:
    """只打开一颗芯片，并让它的两个 TensorCore 各作为一个 device 出现（split-chip）。原生 XLA 的程序默认在第 0 个 device 上运行，只用一个 TensorCore。"""
    initialize_one_chip(chip)
    os.environ['LIBTPU_INIT_ARGS'] = f"{os.environ.get('LIBTPU_INIT_ARGS', '')} --deepsea_chip_config_name=legacy".strip()

def initialize_local_chips() -> None:
    """打开本 host 的全部四颗芯片（2x2x1）；在多 host 切片上也只用本 host，不调用 `jax.distributed.initialize()`。"""
    os.environ['TPU_CHIPS_PER_PROCESS_BOUNDS'] = '2,2,1'
    os.environ['TPU_PROCESS_BOUNDS'] = '1,1,1'
    os.environ['TPU_VISIBLE_CHIPS'] = '0,1,2,3'
