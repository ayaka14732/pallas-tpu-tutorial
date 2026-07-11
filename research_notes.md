# Research Notes: JAX Pallas TPU

## Key Sources in JAX Repo

### Official Docs (docs/pallas/)
- `quickstart.md` - Hello world, Ref types, BlockSpec by example, matmul
- `grid_blockspec.md` - Detailed grid/BlockSpec semantics, index_map, OOB padding
- `pipelining.md` - General pipelining concepts, double-buffering derivation
- `tpu/details.rst` - TPU-specific: hardware overview, noteworthy properties, supported ops
- `tpu/pipelining.md` - TPU memory spaces (HBM/VMEM/SMEM), emit_pipeline, megacore
- `tpu/matmul.md` - Block matmul, tiling, performance analysis, bf16, fused ops
- `tpu/sparse.md` - Scalar prefetch, PrefetchScalarGridSpec, block-sparse matmul
- `tpu/distributed.md` - TPU topologies, RDMA, async remote copy, ppermute/all_gather/psum
- `tpu/core_map.md` - Per-core programming, core_map + shard_map
- `tpu/hardware.ipynb` - TPU hardware specs table

### Production Kernels (jax/experimental/pallas/ops/tpu/)
- `example_kernel.py` - Simplest: double kernel
- `flash_attention.py` - Full FlashAttention with BlockSizes, SegmentIds
- `matmul.py` - Production matmul
- `paged_attention/` - Paged attention kernel
- `ragged_paged_attention/kernel.py` - Complex: MultiPageAsyncCopyDescriptor, while_loop, double-buffering DMA, scalar prefetch
- `splash_attention/` - Splash attention
- `megablox/` - MoE grouped matmul

### Test Files (tests/pallas/)
- `tpu_pallas_test.py` - Broad test suite, scalar prefetch calling convention
- `tpu_pallas_pipeline_test.py` - Pipeline matmul patterns, emit_pipeline, megacore, buffering
- `tpu_pallas_memory_space_test.py` - Memory space constraints
- `tpu_pallas_distributed_test.py` - Distributed kernel tests

## TPU Architecture Key Facts
- TPU = sequential machine with very wide vector register (like CPU, not GPU)
- Async operations: HBM access (DMA), matrix multiply (MXU), transpose/permute (XLU)
- Memory hierarchy: HBM (DRAM, large) -> VMEM (SRAM, 16MB+) -> VREGs (8x128 for f32)
- SMEM: scalar memory, 32-bit random access, for control flow
- Grid executes sequentially in lexicographic order (not parallel like GPU)
- Block shape constraints: last 2 dims must be divisible by 8 and 128 respectively
- MXU always produces float32 results
- Megacore: 2 TensorCores per chip, dimension_semantics=["parallel", "arbitrary"]

## BlockSpec Key Concepts
- `BlockSpec(block_shape, index_map)` - maps grid indices to block indices
- Block indices * block_shape = element indices (start of slice)
- Consecutive grid indices writing same output slice: no race condition
- Reduction dimension must be last grid axis (output doesn't vary)
- `None` in block_shape = squeeze that dimension
- TPU: blocks rank >= 1, last 2 dims divisible by 8 and 128

## Pipelining Key Concepts
- Default: pallas_call copies HBM -> VMEM before kernel, VMEM -> HBM after
- Double-buffering: overlap copy_in(i+1) with compute(i)
- `pl.Buffered(buffer_count=N)` for N-buffering
- `pltpu.emit_pipeline` for nested pipelines
- Scratch buffers: persistent across iterations, for accumulators

## Scalar Prefetch (PrefetchScalarGridSpec)
- `num_scalar_prefetch=n`: first n args go to SMEM, no BlockSpec needed
- index_map receives: (*grid_indices, *prefetch_refs)
- kernel receives: (*prefetch_refs, *input_refs, *output_refs, *scratch_refs)
- Use case: dynamic/sparse block indexing

## Ragged Paged Attention Pattern
- MultiPageAsyncCopyDescriptor: manual DMA with double-buffering
- pltpu.make_async_copy for HBM->VMEM page copies
- lax.while_loop for dynamic iteration over sequences and KV blocks
- Online softmax with running max/sum (flash attention pattern)
- Grid: (num_heads_blks, num_q_blks), dimension_semantics=("arbitrary","arbitrary")
- Scratch: VMEM for kv_bufs, l, m, acc; DMA semaphores
