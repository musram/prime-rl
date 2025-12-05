"""
Triton Kernel Tutorial - Step by Step Learning
=============================================

This tutorial teaches Triton kernels from basics to advanced concepts.
Triton is a language for writing GPU kernels that's easier than CUDA.

Prerequisites:
- pip install triton
- GPU with CUDA support (optional - can run on CPU for learning)

Note: This tutorial works on both GPU and CPU. GPU is recommended for
better performance, but CPU execution is useful for learning and testing.
"""

import torch
import triton
import triton.language as tl
import time
import numpy as np

# ============================================================================
# STEP 1: Basic Vector Addition Kernel
# ============================================================================

@triton.jit
def vector_add_kernel(
    x_ptr,  # Pointer to input vector x
    y_ptr,  # Pointer to input vector y  
    z_ptr,  # Pointer to output vector z
    n_elements,  # Number of elements
    BLOCK_SIZE: tl.constexpr,  # Block size (compile-time constant)
):
    """
    Simple vector addition: z = x + y
    
    Key Concepts:
    - @triton.jit: Decorator to compile function for GPU
    - tl.constexpr: Compile-time constants for optimization
    - Pointers: Memory addresses for input/output arrays
    """
    # Get the current program's block ID
    pid = tl.program_id(axis=0)
    
    # Calculate starting offset for this block
    block_start = pid * BLOCK_SIZE
    
    # Create array of offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle array bounds
    mask = offsets < n_elements
    
    # Load data from memory (with masking for safety)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform computation
    z = x + y
    
    # Store result back to memory
    tl.store(z_ptr + offsets, z, mask=mask)

def vector_add_triton(x, y):
    """Wrapper function to launch vector addition kernel"""
    # Allocate output tensor
    z = torch.empty_like(x)
    
    # Determine grid size and block size
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Choose power of 2
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)  # Number of blocks
    
    # Launch kernel
    vector_add_kernel[grid](
        x, y, z, 
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return z

# ============================================================================
# STEP 2: Matrix Multiplication Kernel  
# ============================================================================

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,  # Matrix dimensions: A(M,K) @ B(K,N) = C(M,N)
    stride_am, stride_ak,  # Strides for A
    stride_bk, stride_bn,  # Strides for B  
    stride_cm, stride_cn,  # Strides for C
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Matrix multiplication using tiling for better memory access.
    
    Key Concepts:
    - Tiling: Break large matrices into smaller blocks
    - Strides: How to navigate multi-dimensional arrays
    - Accumulation: Sum partial products
    """
    # Get program IDs for the current tile
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate offsets for the current tile
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in chunks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        
        # Calculate pointers for current tiles
        A_ptrs = A_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
        B_ptrs = B_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
        
        # Create masks for boundary checking
        mask_a = (rm[:, None] < M) & (rk[None, :] < K)
        mask_b = (rk[:, None] < K) & (rn[None, :] < N)
        
        # Load tiles from memory
        a = tl.load(A_ptrs, mask=mask_a, other=0.0)
        b = tl.load(B_ptrs, mask=mask_b, other=0.0)
        
        # Accumulate partial products
        acc += tl.dot(a, b)
    
    # Store result
    C_ptrs = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask_c = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C_ptrs, acc, mask=mask_c)

def matmul_triton(A, B):
    """Wrapper for matrix multiplication kernel"""
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    # Tile sizes (tune these for your hardware)
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64  
    BLOCK_SIZE_K = 32
    
    # Grid dimensions
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N)
    )
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return C

# ============================================================================
# STEP 3: Advanced Example - Softmax with Memory Optimization
# ============================================================================

@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Numerically stable softmax implementation.
    
    Key Concepts:
    - Numerical stability: subtract max to prevent overflow
    - Reduction operations: max and sum across a dimension
    - Memory coalescing: efficient memory access patterns
    """
    # Get row index
    row_idx = tl.program_id(0)
    
    # Calculate row pointers
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # Create column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load row data
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # Find max for numerical stability
    row_max = tl.max(row, axis=0)
    
    # Subtract max and compute exponential
    safe_row = row - row_max
    numerator = tl.exp(safe_row)
    
    # Compute sum for normalization
    denominator = tl.sum(numerator, axis=0)
    
    # Compute softmax
    softmax_output = numerator / denominator
    
    # Store result
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)

def softmax_triton(x):
    """Wrapper for softmax kernel"""
    n_rows, n_cols = x.shape
    
    # Determine block size (must be power of 2 and >= n_cols)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Allocate output
    y = torch.empty_like(x)
    
    # Launch kernel (one block per row)
    softmax_kernel[(n_rows,)](
        y, x,
        x.stride(0), y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y

# ============================================================================
# STEP 4: Benchmarking and Testing
# ============================================================================

def can_use_triton():
    """Check if Triton kernels can be used (requires GPU)"""
    try:
        # Try to create a small test kernel to see if Triton works
        @triton.jit
        def test_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            tl.store(x_ptr + offsets, x, mask=mask)
        
        # Try to run it on a small tensor
        if torch.cuda.is_available():
            x = torch.randn(10, device="cuda")
            test_kernel[(1,)](x, 10, BLOCK_SIZE=10)
            return True
        return False
    except:
        return False

def benchmark_kernels():
    """Compare Triton kernels with PyTorch implementations"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()
    use_triton = can_use_triton()
    
    print(f"=== Triton Kernel Benchmarks (Device: {device}) ===\n")
    
    if not use_triton:
        print("⚠️  Triton kernels require GPU support. Running PyTorch-only benchmarks on CPU.\n")
        print("Note: To use Triton kernels, you need:")
        print("  - NVIDIA GPU with CUDA support, OR")
        print("  - AMD GPU with ROCm support\n")
    
    # Vector Addition Benchmark
    print("1. Vector Addition (1M elements)")
    x = torch.randn(1000000, device=device)
    y = torch.randn(1000000, device=device)
    
    # Warmup
    for _ in range(10):
        if use_triton:
            _ = vector_add_triton(x, y)
        _ = x + y
    
    # Benchmark Triton (if available)
    if use_triton:
        if use_cuda:
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            z_triton = vector_add_triton(x, y)
        if use_cuda:
            torch.cuda.synchronize()
        triton_time = time.time() - start
    else:
        triton_time = None
        z_triton = None
    
    # Benchmark PyTorch
    if use_cuda:
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        z_torch = x + y
    if use_cuda:
        torch.cuda.synchronize()
    torch_time = time.time() - start
    
    if use_triton:
        print(f"  Triton:  {triton_time:.4f}s")
        print(f"  PyTorch: {torch_time:.4f}s")
        print(f"  Speedup: {torch_time/triton_time:.2f}x")
        print(f"  Correctness: {torch.allclose(z_triton, z_torch)}\n")
    else:
        print(f"  PyTorch: {torch_time:.4f}s")
        print(f"  (Triton not available on CPU)\n")
    
    # Matrix Multiplication Benchmark
    print("2. Matrix Multiplication (512x512)")
    A = torch.randn(512, 512, device=device)
    B = torch.randn(512, 512, device=device)
    
    # Warmup
    for _ in range(5):
        if use_triton:
            _ = matmul_triton(A, B)
        _ = torch.mm(A, B)
    
    # Benchmark
    if use_triton:
        if use_cuda:
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            C_triton = matmul_triton(A, B)
        if use_cuda:
            torch.cuda.synchronize()
        triton_time = time.time() - start
    else:
        triton_time = None
        C_triton = None
    
    if use_cuda:
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(20):
        C_torch = torch.mm(A, B)
    if use_cuda:
        torch.cuda.synchronize()
    torch_time = time.time() - start
    
    if use_triton:
        print(f"  Triton:  {triton_time:.4f}s")
        print(f"  PyTorch: {torch_time:.4f}s")
        print(f"  Speedup: {torch_time/triton_time:.2f}x")
        print(f"  Correctness: {torch.allclose(C_triton, C_torch, rtol=1e-3)}\n")
    else:
        print(f"  PyTorch: {torch_time:.4f}s")
        print(f"  (Triton not available on CPU)\n")
    
    # Softmax Benchmark
    print("3. Softmax (1000x1000)")
    x = torch.randn(1000, 1000, device=device)
    
    # Warmup
    for _ in range(5):
        if use_triton:
            _ = softmax_triton(x)
        _ = torch.softmax(x, dim=1)
    
    # Benchmark
    if use_triton:
        if use_cuda:
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(50):
            y_triton = softmax_triton(x)
        if use_cuda:
            torch.cuda.synchronize()
        triton_time = time.time() - start
    else:
        triton_time = None
        y_triton = None
    
    if use_cuda:
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(50):
        y_torch = torch.softmax(x, dim=1)
    if use_cuda:
        torch.cuda.synchronize()
    torch_time = time.time() - start
    
    if use_triton:
        print(f"  Triton:  {triton_time:.4f}s")
        print(f"  PyTorch: {torch_time:.4f}s")
        print(f"  Speedup: {torch_time/triton_time:.2f}x")
        print(f"  Correctness: {torch.allclose(y_triton, y_torch, rtol=1e-3)}")
    else:
        print(f"  PyTorch: {torch_time:.4f}s")
        print(f"  (Triton not available on CPU)")

# ============================================================================
# STEP 5: Learning Exercises
# ============================================================================

def learning_exercises():
    """
    Practice exercises to deepen understanding:
    
    Exercise 1: Element-wise operations
    - Implement ReLU kernel: relu(x) = max(0, x)
    - Implement GELU kernel: gelu(x) = x * Φ(x) where Φ is CDF of standard normal
    
    Exercise 2: Reductions  
    - Implement sum reduction along a dimension
    - Implement mean and variance calculation
    
    Exercise 3: Memory patterns
    - Implement transpose kernel with coalesced memory access
    - Implement convolution kernel (1D or 2D)
    
    Exercise 4: Advanced features
    - Use tl.atomic_add for race-condition-free updates
    - Implement sparse operations using masks
    """
    pass

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if torch.cuda.is_available():
        print("CUDA available! Running benchmarks on GPU...")
    else:
        print("CUDA not available. Running benchmarks on CPU...")
        print("Note: Triton kernels require GPU support. PyTorch operations will be used on CPU.\n")
    
    benchmark_kernels()
        
    print("\n=== Quick Test ===")
    
    # Test vector addition
    x = torch.tensor([1., 2., 3., 4.], device=device)
    y = torch.tensor([5., 6., 7., 8.], device=device)
    z_torch = x + y
    print(f"Vector add (PyTorch): {x} + {y} = {z_torch}")
    
    # Try Triton if available
    if torch.cuda.is_available():
        try:
            z_triton = vector_add_triton(x, y)
            print(f"Vector add (Triton): {x} + {y} = {z_triton}")
            print(f"Results match: {torch.allclose(z_triton, z_torch)}")
        except Exception as e:
            print(f"Triton test failed: {e}")
    else:
        print("(Triton kernels require GPU - skipping Triton test)")