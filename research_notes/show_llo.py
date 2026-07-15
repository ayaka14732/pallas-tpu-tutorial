# ========================================================================
# Pallas call wrapper that assigns a unique kernel name,
# then locates and prints the compiled LLO from the Mosaic dumps.
# ========================================================================

import os
from pathlib import Path
import warnings

DUMP_DIR = Path("/tmp/mosaic_dumps")

# Based on https://openxla.org/xla/hlo_dumps#mosaic
os.environ["LIBTPU_INIT_ARGS"] = f"--xla_mosaic_dump_to={DUMP_DIR}"

def clear_compiled_llo() -> None:
    for path in DUMP_DIR.rglob("*.txt"):
        try:
            path.unlink()
        except OSError:
            warnings.warn(f"Failed to remove file {path}")

def print_compiled_llo(kernel_name: str) -> None:
    matched = []

    for path in DUMP_DIR.rglob("*post-finalize-llo.txt"):
        try:
            if kernel_name in path.read_text(encoding="utf-8", errors="ignore"):
                matched.append(path)
        except OSError:
            pass

    if matched:
        path = max(matched)
        print(f"------------------- File {path} -------------------")
        print(path.read_text(encoding="utf-8", errors="ignore").strip())

# ========================================================================
# Use libtpu's compile-only client to compile Pallas kernels for a mock
# TPU topology without an attached TPU.
# ========================================================================

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["TPU_ACCELERATOR_TYPE"] = "v5e-4"
os.environ["TPU_WORKER_HOSTNAMES"] = "localhost"

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental import topologies

topology = topologies.get_topology_desc("v5e-4", platform="tpu")
target = jax.sharding.SingleDeviceSharding(topology.devices[0])

def kernel(x_ref: jax.Ref, o_ref: jax.Ref) -> None:
    col = pl.program_id(0)
    row = pl.program_id(1)

    pl.debug_print("Executing grid ({}, {})", col, row)

    o_ref[...] = x_ref[...] * 2.0

def fn(x: jax.Array) -> jax.Array:
    N, M = x.shape
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec(block_shape=(8, 128), index_map=lambda i, j: (i, j))],
        out_specs=pl.BlockSpec(block_shape=(8, 128), index_map=lambda i, j: (i, j)),
        grid=(pl.cdiv(N, 8), pl.cdiv(M, 128)),
        name="tpu_kernel_test",
        # interpret=True,
    )(x)

x_spec = jax.ShapeDtypeStruct((24, 384), jnp.float32, sharding=target)
clear_compiled_llo()
jax.jit(fn, in_shardings=target, out_shardings=target).lower(x_spec).compile()
print_compiled_llo(kernel_name="tpu_kernel_test")
