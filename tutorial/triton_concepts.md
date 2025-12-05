# Mastery Roadmap: OpenAI Triton

This document outlines the concepts, techniques, and "tricks" required to master Triton kernel development, progressing from basic implementation to advanced optimization and hardware-aware programming.

## 1. Core Concepts (The Foundation)

Before writing efficient kernels, you must understand how Triton abstracts the GPU hierarchy.

*   **Block-Based Programming (SPMD)**:
    *   Understanding that you write code for a *single block* of threads, not individual threads (unlike CUDA).
    *   **Program IDs (`pid`)**: How to map `tl.program_id(axis)` to specific chunks of your data.
    *   **Grids**: Defining the 1D, 2D, or 3D grid of blocks to launch.
*   **Pointers & Arithmetic**:
    *   **Pointer Arithmetic**: `ptr + offset`.
    *   **Broadcasting**: Creating grids of pointers using `offset[:, None] + offset[None, :]`.
    *   **Masking**: Using `mask=` in `tl.load` and `tl.store` to handle tensor dimensions that aren't powers of 2.
*   **Data Types**:
    *   `tl.float32`, `tl.float16`, `tl.bfloat16`.
    *   Type conversion (`.to()`) and when to cast (load in FP16 -> compute in FP32 -> store in FP16).

## 2. Memory Strategy (The #1 Performance Bottleneck)

GPU programming is primarily about memory management. Compute is cheap; data movement is expensive.

*   **Tiling (Block-wise Processing)**:
    *   **Concept**: Breaking large tensors into small "tiles" (e.g., 64x64) that fit into SRAM (L1/L2 Cache).
    *   **Implementation**: Using `tl.arange` to generate offsets for a tile and looping over dimensions (e.g., the K dimension in MatMul).
*   **Memory Coalescing**:
    *   **Concept**: Ensuring adjacent threads read adjacent memory addresses.
    *   **Goal**: Minimize the number of memory transactions.
    *   **Rule**: The fastest varying dimension of your `offsets` should match the contiguous dimension of your tensor in memory.
*   **Block Pointers (`tl.make_block_ptr`)**:
    *   **Concept**: High-level abstraction for 2D/3D pointer arithmetic.
    *   **Benefits**: Automatically handles strides, offsets, and boundary masking. Essential for complex MatMuls and FlashAttention.
*   **Strides**:
    *   Understanding `tensor.stride()` to navigate non-contiguous memory or slice multidimensional arrays manually.
    *   Row-major vs. Column-major layouts.

## 3. Kernel Fusion & Operational Logic

Triton's main advantage is fusing multiple PyTorch operations into a single kernel read/write cycle.

*   **Standard Fusion**:
    *   Loading data once, performing `MatMul -> Bias Add -> Activation (ReLU/GELU) -> Residual Add`, and storing once.
*   **Broadcasting Semantics**:
    *   How `x + y` works when `x` is shape `[BLOCK, 1]` and `y` is `[1, BLOCK]`.
*   **Reductions**:
    *   `tl.sum`, `tl.max`, `tl.min`, `tl.argmax`.
    *   Reducing along specific axes within a block.
*   **Atomic Operations**:
    *   `tl.atomic_add`, `tl.atomic_max`.
    *   **Use Case**: Scatter operations, histograms, or writing gradients to shared buffers where race conditions exist.

## 4. Advanced Optimization (Hardware Awareness)

*   **Pipeline Parallelism (`num_stages`)**:
    *   **Concept**: Prefetching data for the *next* loop iteration while computing the *current* one.
    *   **Usage**: Setting `num_stages=3` or `4` in `@triton.jit` or `triton.Config`.
    *   **Goal**: Hiding HBM (Global Memory) latency.
*   **Warp Divergence**:
    *   **Concept**: GPUs execute threads in groups of 32 (warps). If threads in a warp take different code paths (if/else), performance drops.
    *   **Fix**: Use predication/masking (`tl.where(condition, val_if_true, val_if_false)`) instead of Python `if/else` inside hot loops.
*   **L2 Cache Optimization**:
    *   **Swizzling**: Reordering block execution order (e.g., "Morton curve" or specialized group ordering) to maximize L2 cache hits.
    *   **Why**: Increases data reuse between adjacent blocks.

## 5. Autotuning (`@triton.autotune`)

Writing portable kernels that work on A100, H100, and 4090.

*   **Config Search Space**:
    *   Defining ranges for `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `num_warps`, and `num_stages`.
*   **Heuristics**:
    *   Using `prune_configs_by` to skip invalid configurations (e.g., Block Size > Tensor Size).
*   **Key Argument**:
    *   Using `key=['n_elements']` to trigger re-tuning only when problem size changes significantly.

## 6. Numerical Stability & Advanced Algorithms

*   **Online Softmax**:
    *   Computing Softmax without materializing the full N^2 matrix.
    *   Technique: Subtracting `max(row)` on the fly during accumulation to prevent exp() overflow.
*   **Accumulator Precision**:
    *   Always accumulating MatMul results in `fp32` (float) even if inputs are `fp16` (half) to prevent precision loss.
*   **FlashAttention Patterns**:
    *   Recomputing statistics during the backward pass to save memory (activation checkpointing within the kernel).

## 7. Debugging & Development Tools

*   **`tl.device_print`**:
    *   Printing values from inside the GPU kernel (use `if pid == 0:` to avoid flooding stdout).
*   **Interpreter Mode**:
    *   Running Triton kernels on CPU (slow but allows PDB/debugging) by setting environment variable `TRITON_INTERPRET=1`.
*   **Unit Testing**:
    *   Comparing Triton output against PyTorch reference implementation using `torch.allclose()`.

## 8. Study Checklist

To consider yourself "Advanced" in Triton, you should be able to implement:

1.  [ ] **Vector Add** (Hello World)
2.  [ ] **Softmax** (Row-wise reduction)
3.  [ ] **MatMul** (Tiled Matrix Multiplication with Block Pointers)
4.  [ ] **Fused MLP** (MatMul + Bias + GELU in one go)
5.  [ ] **FlashAttention** (Tiled Attention with Online Softmax)
6.  [ ] **Quantization** (Int8 loading dequantized to FP16 for compute)

## 9. Useful Resources

*   **Official Tutorials**: triton-lang.org/main/getting-started/tutorials
*   **Unsloth Kernels**: Excellent reference for readable, high-performance Triton.
*   **FlashAttention Paper/Code**: The gold standard for advanced tiling logic.

 The ONE Most Critical Concept: Tiled Matrix Multiplication

  This is the foundation of ALL GPU kernels. Here's why:

  1. Memory Hierarchy: GPUs have fast shared memory (10TB/s) vs slow global memory (1.5TB/s)
  2. Tiling Strategy: Load small blocks into fast memory, compute, repeat
  3. Pattern: Each program computes one output block by accumulating partial products

  The template:
  # Each program handles one output block
  pid_m, pid_n = tl.program_id(0), tl.program_id(1)
  accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

  for k_block in range(0, tl.cdiv(K, BLOCK_K)):
      # Load A[block_row, k_slice] and B[k_slice, block_col]
      a = tl.load(A_ptrs, mask=A_mask, other=0.0)
      b = tl.load(B_ptrs, mask=B_mask, other=0.0)
      # Accumulate partial product
      accumulator += tl.dot(a, b)

  # Store final result
  tl.store(C_ptrs, accumulator, mask=C_mask)

  Essential Prerequisites:

  1. GPU Memory Hierarchy - Registers → Shared → L2 → Global
  2. Program Model - One program = one thread block with unique ID
  3. Safe Memory Access - Always use masks with tl.load/store
  4. Pointer Arithmetic - Broadcasting for 2D indexing: ptr + row[:, None]*stride + col[None, :]
  5. Block Size Selection - Usually 64-128 for optimal shared memory usage

  Critical Functions:

  - tl.program_id(axis) - Get program ID
  - tl.arange(start, end) - Create indices
  - tl.load/store(ptr, mask, other) - Safe memory access
  - tl.dot(a, b) - Optimized matrix multiply
  - x.to(tl.float32) - Numerical stability


Summary Checklist for "Level 2" Triton:
[ ] Strides: Understanding tensor.stride(0) vs tensor.stride(1) to navigate N-D arrays manually.
[ ] Broadcasting: How offset[:, None] + offset[None, :] creates 2D grids.
[ ] Masking: Using mask= in every load/store to handle shapes that aren't powers of 2.
[ ] Reduction: Using tl.sum, tl.max along specific axes.
[ ] Compiler Hints: Using tl.constexpr to force the compiler to unroll loops