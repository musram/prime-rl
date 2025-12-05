"""
Matrix Multiplication with Tiling - The Foundation of GPU Computing
==================================================================

This is THE most important concept in GPU programming. Understanding this
will unlock everything else: attention, convolutions, linear layers, etc.

Why Tiling?
----------
GPU has a memory hierarchy:
- Global Memory (HBM): ~1TB/s bandwidth, high latency
- Shared Memory: ~10TB/s bandwidth, low latency  
- Registers: ~100TB/s bandwidth, immediate access

Tiling loads small blocks into fast memory, computes on them,
then moves to the next block. This maximizes bandwidth utilization.
"""

import torch
import triton
import triton.language as tl
import math


def visualize_tiling():
    """
    Understanding Tiling Conceptually
    ================================
    
    Matrix C = A @ B where:
    - A is (M, K) 
    - B is (K, N)
    - C is (M, N)
    
    Without tiling: Load entire A and B → compute → store C
    Problem: Doesn't fit in fast memory, poor memory reuse
    
    With tiling: 
    1. Break A into row blocks of size (BLOCK_M, BLOCK_K)
    2. Break B into column blocks of size (BLOCK_K, BLOCK_N)  
    3. For each output block C[i,j]:
       - Load A[i,:] and B[:,j] blocks
       - Compute partial products
       - Accumulate results
       - Store final C[i,j]
    """
    print("Tiling Visualization:")
    print("====================")
    print()
    
    # Example dimensions
    M, N, K = 128, 128, 128
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
    
    print(f"Matrix dimensions: A({M},{K}) @ B({K},{N}) = C({M},{N})")
    print(f"Block sizes: ({BLOCK_M},{BLOCK_K}) @ ({BLOCK_K},{BLOCK_N}) = ({BLOCK_M},{BLOCK_N})")
    print()
    
    # Calculate number of blocks
    num_blocks_m = M // BLOCK_M  # 4 blocks
    num_blocks_n = N // BLOCK_N  # 4 blocks  
    num_blocks_k = K // BLOCK_K  # 4 blocks
    
    print(f"Grid: {num_blocks_m} x {num_blocks_n} = {num_blocks_m * num_blocks_n} output blocks")
    print(f"Each output block needs {num_blocks_k} K-dimension blocks")
    print()
    
    # Visualize the computation for one output block
    print("For output block C[0,0] (rows 0-31, cols 0-31):")
    print("  1. Load A[0-31, 0-31]   @ B[0-31, 0-31]   → partial_sum_0")
    print("  2. Load A[0-31, 32-63]  @ B[32-63, 0-31]  → partial_sum_1") 
    print("  3. Load A[0-31, 64-95]  @ B[64-95, 0-31]  → partial_sum_2")
    print("  4. Load A[0-31, 96-127] @ B[96-127, 0-31] → partial_sum_3")
    print("  5. C[0,0] = partial_sum_0 + partial_sum_1 + partial_sum_2 + partial_sum_3")
    print()


@triton.jit
def tiled_matmul_kernel(
    # Matrix pointers
    A_ptr, B_ptr, C_ptr,
    # Matrix dimensions
    M, N, K,
    # Memory strides (how to move between rows/columns)
    stride_am, stride_ak,  # A: (M, K)
    stride_bk, stride_bn,  # B: (K, N)
    stride_cm, stride_cn,  # C: (M, N)
    # Block sizes - constexpr means compile-time constant
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    The Heart of GPU Programming: Tiled Matrix Multiplication
    
    Key Insight: Each "program" computes one output block C[i,j]
    
    Memory Access Pattern:
    =====================
    - Load A blocks: A[block_row_i, k_slice] 
    - Load B blocks: B[k_slice, block_col_j]
    - Compute: partial_product = A_block @ B_block
    - Accumulate: accumulator += partial_product
    - Store: C[block_row_i, block_col_j] = accumulator
    """
    
    # STEP 1: Identify which output block this program computes
    # ========================================================
    pid_m = tl.program_id(0)  # Which row block (0 to M//BLOCK_M - 1)
    pid_n = tl.program_id(1)  # Which col block (0 to N//BLOCK_N - 1)
    
    # Calculate the actual row/column indices for this block
    # Each program handles BLOCK_M rows and BLOCK_N columns
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [0,1,...,BLOCK_M-1] + offset
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [0,1,...,BLOCK_N-1] + offset
    
    # STEP 2: Initialize accumulator for this output block
    # ===================================================
    # We'll accumulate partial products here
    # Use float32 for numerical stability even with float16 inputs
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # STEP 3: Loop over K dimension in blocks
    # =======================================
    # This is the key insight: break the dot product into chunks
    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        # Calculate K indices for current block
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        
        # STEP 4: Load A block - shape (BLOCK_M, BLOCK_K)
        # ===============================================
        # We need A[offs_m, offs_k] 
        # Use broadcasting: offs_m[:, None] × offs_k[None, :]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # STEP 5: Load B block - shape (BLOCK_K, BLOCK_N)  
        # ===============================================
        # We need B[offs_k, offs_n]
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # STEP 6: Compute partial matrix multiplication
        # ============================================
        # tl.dot() is optimized matrix multiplication
        # a: (BLOCK_M, BLOCK_K) @ b: (BLOCK_K, BLOCK_N) → (BLOCK_M, BLOCK_N)
        accumulator += tl.dot(a, b)
        
        # Continue to next K block...
    
    # STEP 7: Store final result 
    # ==========================
    # Calculate output pointers
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # Convert back to original dtype and store
    tl.store(c_ptrs, accumulator.to(A_ptr.dtype.element_ty), mask=c_mask)


def triton_matmul(a, b):
    """
    High-level wrapper for tiled matrix multiplication
    
    This shows how to configure and launch the kernel
    """
    # Validate dimensions
    assert a.shape[1] == b.shape[0], f"Incompatible shapes: {a.shape} @ {b.shape}"
    assert a.is_contiguous() and b.is_contiguous(), "Tensors must be contiguous"
    
    M, K = a.shape
    K, N = b.shape
    
    # Create output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # BLOCK SIZE SELECTION - This is crucial for performance!
    # =====================================================
    # Factors to consider:
    # 1. GPU shared memory limits (typically ~48-164KB)
    # 2. Register pressure (too large blocks → spill to memory)
    # 3. Occupancy (too small blocks → underutilize GPU)
    # 4. Memory coalescing (prefer multiples of 32/64)
    
    BLOCK_M = 128  # Good balance for most cases
    BLOCK_N = 128  # Should be same as BLOCK_M for square efficiency  
    BLOCK_K = 32   # Smaller K reduces memory pressure
    
    # Verify blocks fit in shared memory
    # Each block needs BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N elements
    shared_memory_per_block = (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * 4  # 4 bytes for float32
    print(f"Shared memory per block: {shared_memory_per_block / 1024:.1f} KB")
    
    # GRID CONFIGURATION
    # ==================
    # We need one program for each output block
    grid = (
        triton.cdiv(M, BLOCK_M),  # Number of row blocks
        triton.cdiv(N, BLOCK_N),  # Number of column blocks  
    )
    
    print(f"Launching {grid[0]} x {grid[1]} = {grid[0] * grid[1]} programs")
    
    # KERNEL LAUNCH
    # =============
    tiled_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),  # How to navigate A
        b.stride(0), b.stride(1),  # How to navigate B  
        c.stride(0), c.stride(1),  # How to navigate C
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N, 
        BLOCK_K=BLOCK_K,
    )
    
    return c


def advanced_tiling_concepts():
    """
    Advanced Tiling Concepts for Performance
    """
    print("\nAdvanced Tiling Concepts:")
    print("=" * 30)
    
    print("\n1. MEMORY HIERARCHY OPTIMIZATION:")
    print("   - Global Memory (HBM): ~1.5 TB/s, high latency")
    print("   - Shared Memory: ~10+ TB/s, low latency")  
    print("   - Registers: ~100+ TB/s, immediate")
    print("   → Tiling maximizes use of faster memory levels")
    
    print("\n2. MEMORY COALESCING:")
    print("   - GPU loads memory in chunks (32-128 bytes)")
    print("   - Adjacent threads should load adjacent memory")
    print("   - Block sizes should be multiples of 32/64")
    
    print("\n3. BLOCK SIZE SELECTION:")
    print("   - Too small: Poor arithmetic intensity, underutilized GPU")
    print("   - Too large: Exceeds shared memory, register spilling")
    print("   - Sweet spot: Usually 64x64 to 128x128 for matmul")
    
    print("\n4. DOUBLE BUFFERING:")
    print("   - Load next block while computing current block")
    print("   - Hides memory latency behind computation")
    print("   - Advanced technique for maximum performance")
    
    print("\n5. WARP-LEVEL OPTIMIZATIONS:")
    print("   - 32 threads per warp execute in lockstep")
    print("   - Avoid warp divergence (different branches)")
    print("   - Use warp-level primitives when possible")


def performance_comparison():
    """
    Compare naive vs tiled implementations
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("CUDA not available, skipping performance test")
        return
        
    print("\nPerformance Comparison:")
    print("=" * 25)
    
    # Test matrices
    M, N, K = 1024, 1024, 1024
    a = torch.randn(M, K, device=device, dtype=torch.float16)
    b = torch.randn(K, N, device=device, dtype=torch.float16)
    
    # Triton implementation
    c_triton = triton_matmul(a, b)
    
    # PyTorch reference
    c_torch = torch.matmul(a, b)
    
    # Check correctness
    max_diff = torch.max(torch.abs(c_triton - c_torch)).item()
    rel_diff = max_diff / torch.max(torch.abs(c_torch)).item()
    
    print(f"Matrix size: {M} x {K} @ {K} x {N}")
    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Relative difference: {rel_diff:.2e}")
    print(f"Test {'PASSED' if rel_diff < 1e-2 else 'FAILED'}")


if __name__ == "__main__":
    print("Matrix Multiplication with Tiling - Deep Dive")
    print("=" * 50)
    
    # Conceptual explanation
    visualize_tiling()
    
    # Advanced concepts
    advanced_tiling_concepts()
    
    # Performance test
    performance_comparison()
    
    print("\n" + "=" * 50)
    print("KEY TAKEAWAYS:")
    print("1. Tiling breaks large matrices into GPU-friendly blocks")
    print("2. Each program computes one output block")  
    print("3. Loop over K dimension, accumulating partial products")
    print("4. Memory access pattern is critical for performance")
    print("5. Block size selection balances memory and compute")
    print("6. This pattern applies to ALL GPU kernels!")