"""
The 5 Essential Triton Concepts - Deep Dive
==========================================

These 5 concepts are the foundation of ALL Triton kernels.
Master these and you can implement any GPU operation.

1. Strides: Manual tensor navigation
2. Broadcasting: Creating 2D index grids  
3. Masking: Safe memory access for any shape
4. Reduction: Aggregating data along axes
5. Compiler Hints: Optimization with constexpr
"""

import torch
import triton
import triton.language as tl
import numpy as np


# =============================================================================
# 1. STRIDES: Understanding Tensor Memory Layout
# =============================================================================

def understand_strides():
    """
    CRITICAL: Strides define how to navigate multi-dimensional tensors in memory
    
    Memory is 1D, but tensors are N-D. Strides tell us how to convert
    tensor[i,j,k] coordinates into memory addresses.
    """
    print("1. TENSOR STRIDES - The Key to Memory Navigation")
    print("=" * 55)
    print()
    
    # Example tensor
    tensor = torch.randn(3, 4, 5)  # 3D tensor: (batch, height, width)
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor strides: {tensor.stride()}")
    print()
    
    # Understanding what strides mean
    print("Stride Explanation:")
    print(f"- stride(0) = {tensor.stride(0)}: To move 1 step in dim 0 (batch), jump {tensor.stride(0)} elements")
    print(f"- stride(1) = {tensor.stride(1)}: To move 1 step in dim 1 (height), jump {tensor.stride(1)} elements") 
    print(f"- stride(2) = {tensor.stride(2)}: To move 1 step in dim 2 (width), jump {tensor.stride(2)} elements")
    print()
    
    # Manual address calculation
    print("Manual Address Calculation:")
    print("tensor[i,j,k] is at memory address:")
    print("base_ptr + i*stride(0) + j*stride(1) + k*stride(2)")
    print()
    
    # Examples
    examples = [
        (0, 0, 0, 0*tensor.stride(0) + 0*tensor.stride(1) + 0*tensor.stride(2)),
        (1, 0, 0, 1*tensor.stride(0) + 0*tensor.stride(1) + 0*tensor.stride(2)),
        (0, 1, 0, 0*tensor.stride(0) + 1*tensor.stride(1) + 0*tensor.stride(2)),
        (0, 0, 1, 0*tensor.stride(0) + 0*tensor.stride(1) + 1*tensor.stride(2)),
        (1, 2, 3, 1*tensor.stride(0) + 2*tensor.stride(1) + 3*tensor.stride(2)),
    ]
    
    for i, j, k, offset in examples:
        print(f"tensor[{i},{j},{k}] → offset {offset}")
        
    print()
    print("💡 Key Insight: In Triton, we manually calculate these offsets!")
    print("   ptr + row_idx*stride(0) + col_idx*stride(1)")


@triton.jit
def stride_demo_kernel(
    input_ptr, output_ptr,
    M, N,
    input_stride_0, input_stride_1,
    output_stride_0, output_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    """Demonstrate stride usage in Triton kernels"""
    
    # Get current position
    row = tl.program_id(0)
    
    # Create column offsets
    cols = tl.arange(0, BLOCK_SIZE)
    col_mask = cols < N
    
    # CRITICAL: Manual pointer arithmetic using strides
    # This is equivalent to input_ptr[row, cols] 
    input_ptrs = input_ptr + row * input_stride_0 + cols * input_stride_1
    output_ptrs = output_ptr + row * output_stride_0 + cols * output_stride_1
    
    # Load, process, store
    data = tl.load(input_ptrs, mask=col_mask, other=0.0)
    result = data * 2.0  # Simple operation
    tl.store(output_ptrs, result, mask=col_mask)


# =============================================================================
# 2. BROADCASTING: Creating 2D Index Grids
# =============================================================================

def understand_broadcasting():
    """
    Broadcasting creates 2D grids from 1D arrays.
    This is ESSENTIAL for 2D operations like matrix multiplication.
    """
    print("\n2. BROADCASTING - Creating 2D Index Grids")
    print("=" * 45)
    print()
    
    # 1D arrays
    rows = np.array([0, 1, 2])  # Shape: (3,)
    cols = np.array([0, 1, 2, 3])  # Shape: (4,)
    
    print(f"rows: {rows} (shape: {rows.shape})")
    print(f"cols: {cols} (shape: {cols.shape})")
    print()
    
    # Broadcasting to 2D
    print("Broadcasting Magic:")
    print("rows[:, None] adds a new axis → shape becomes (3, 1)")
    print("cols[None, :] adds a new axis → shape becomes (1, 4)")
    print()
    
    rows_2d = rows[:, None]  # Shape: (3, 1)
    cols_2d = cols[None, :]  # Shape: (1, 4)
    
    print(f"rows[:, None]:\n{rows_2d}")
    print(f"cols[None, :]:\n{cols_2d}")
    print()
    
    # The magic: broadcasting creates 2D grid
    print("When combined, broadcasting creates full 2D grids:")
    row_grid = rows_2d + cols_2d * 0  # Broadcast rows to (3, 4)
    col_grid = rows_2d * 0 + cols_2d  # Broadcast cols to (3, 4)
    
    print(f"Row indices grid:\n{row_grid}")
    print(f"Col indices grid:\n{col_grid}")
    print()
    
    print("💡 In Triton: offset[:, None] + offset[None, :] creates 2D pointer arrays!")


@triton.jit
def broadcasting_demo_kernel(
    input_ptr, output_ptr,
    M, N,
    stride_m, stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Demonstrate broadcasting in Triton"""
    
    # Get block position
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 1D offsets for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # Shape: (BLOCK_M,)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # Shape: (BLOCK_N,)
    
    # BROADCASTING MAGIC: Create 2D pointer grid
    # offs_m[:, None] → shape (BLOCK_M, 1)
    # offs_n[None, :] → shape (1, BLOCK_N)  
    # Result → shape (BLOCK_M, BLOCK_N)
    ptrs = input_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    
    # 2D mask (also uses broadcasting!)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # Load/store 2D blocks
    data = tl.load(ptrs, mask=mask, other=0.0)
    result = data + 1.0
    
    output_ptrs = output_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    tl.store(output_ptrs, result, mask=mask)


# =============================================================================
# 3. MASKING: Safe Memory Access for Any Shape
# =============================================================================

def understand_masking():
    """
    Masking prevents out-of-bounds memory access.
    ALWAYS use masks - this prevents crashes and wrong results.
    """
    print("\n3. MASKING - Safe Memory Access")
    print("=" * 35)
    print()
    
    print("The Problem:")
    print("- GPU kernels work with fixed block sizes (powers of 2)")
    print("- Real tensors have arbitrary sizes") 
    print("- Without masks: read/write garbage memory!")
    print()
    
    # Example scenario
    size = 100  # Actual tensor size
    BLOCK_SIZE = 64  # GPU block size
    
    print(f"Example: tensor size = {size}, BLOCK_SIZE = {BLOCK_SIZE}")
    print()
    
    # Calculate blocks needed
    num_blocks = triton.cdiv(size, BLOCK_SIZE)  # Ceiling division
    print(f"Blocks needed: {num_blocks}")
    print()
    
    # Show what each block processes
    for block_id in range(num_blocks):
        start = block_id * BLOCK_SIZE
        offsets = np.arange(start, start + BLOCK_SIZE)
        valid = offsets < size
        
        print(f"Block {block_id}:")
        print(f"  Offsets: {offsets}")
        print(f"  Valid:   {valid}")
        print(f"  Valid indices: {offsets[valid]}")
        print()
    
    print("💡 Without masking: Block 1 would access indices 64-99 (out of bounds!)")
    print("💡 With masking: Invalid accesses return 'other' value or are skipped")


@triton.jit  
def masking_demo_kernel(
    input_ptr, output_ptr,
    size,
    BLOCK_SIZE: tl.constexpr,
):
    """Demonstrate proper masking"""
    
    pid = tl.program_id(0)
    
    # Create offsets for this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # CRITICAL: Create mask for valid indices
    mask = offsets < size
    
    # Safe load: invalid indices return 0.0
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Process data
    result = data * 2.0
    
    # Safe store: invalid indices are not written  
    tl.store(output_ptr + offsets, result, mask=mask)
    
    # Advanced: Different masks for different operations
    # Load mask: protect against reading garbage
    load_mask = offsets < size
    
    # Store mask: might be different (e.g., only store positive values)
    store_mask = load_mask & (result > 0)
    
    # data = tl.load(input_ptr + offsets, mask=load_mask, other=0.0)
    # tl.store(output_ptr + offsets, result, mask=store_mask)


# =============================================================================
# 4. REDUCTION: Aggregating Data Along Axes
# =============================================================================

def understand_reductions():
    """
    Reductions aggregate data: sum, max, mean, etc.
    Understanding axis parameter is crucial.
    """
    print("\n4. REDUCTIONS - Aggregating Data")
    print("=" * 35)
    print()
    
    # Example 2D tensor
    data = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8], 
        [9, 10, 11, 12]
    ])
    
    print(f"Example tensor (3x4):\n{data}")
    print()
    
    print("Reduction Operations:")
    print(f"- Sum all elements: {np.sum(data)} (no axis)")
    print(f"- Sum along axis=0: {np.sum(data, axis=0)} (collapse rows → shape (4,))")
    print(f"- Sum along axis=1: {np.sum(data, axis=1)} (collapse cols → shape (3,))")
    print()
    
    print("Axis Understanding:")
    print("- axis=0: Operate along dimension 0 (rows)")
    print("- axis=1: Operate along dimension 1 (columns)")
    print("- No axis: Reduce everything to scalar")
    print()
    
    print("💡 In Triton:")
    print("   tl.sum(x, axis=0) → sum across rows")
    print("   tl.sum(x, axis=1) → sum across columns")
    print("   tl.max(x, axis=0) → max across rows")


@triton.jit
def reduction_demo_kernel(
    input_ptr, 
    row_sum_ptr,    # Output: sum along columns (axis=1)
    col_sum_ptr,    # Output: sum along rows (axis=0)
    M, N,
    stride_m, stride_n,
    BLOCK_SIZE: tl.constexpr,
):
    """Demonstrate reduction operations"""
    
    row_idx = tl.program_id(0)
    
    # Load entire row
    col_offsets = tl.arange(0, BLOCK_SIZE) 
    col_mask = col_offsets < N
    
    ptrs = input_ptr + row_idx * stride_m + col_offsets * stride_n
    row_data = tl.load(ptrs, mask=col_mask, other=0.0)
    
    # REDUCTION 1: Sum along columns (axis=1 equivalent)
    # This gives us one value per row
    row_sum = tl.sum(row_data, axis=0)  # Sum all elements in the row
    
    # Store row sum
    tl.store(row_sum_ptr + row_idx, row_sum)
    
    # REDUCTION 2: For column sums, we need different approach
    # (This would typically be done in a separate kernel)
    
    # Other reductions:
    row_max = tl.max(row_data, axis=0)    # Maximum in row
    row_mean = tl.sum(row_data, axis=0) / N  # Mean of row
    
    # Advanced: Reduction with conditions
    positive_data = tl.where(row_data > 0, row_data, 0.0)
    positive_sum = tl.sum(positive_data, axis=0)


# =============================================================================
# 5. COMPILER HINTS: Optimization with constexpr
# =============================================================================

def understand_compiler_hints():
    """
    constexpr tells compiler values are known at compile time.
    This enables aggressive optimization and loop unrolling.
    """
    print("\n5. COMPILER HINTS - constexpr Optimization")
    print("=" * 45)
    print()
    
    print("What is constexpr?")
    print("- Tells compiler: 'This value is constant at compile time'")
    print("- Enables loop unrolling and other optimizations")
    print("- Required for array dimensions in Triton")
    print()
    
    print("When to use constexpr:")
    print("✓ Block sizes: BLOCK_SIZE: tl.constexpr")
    print("✓ Loop bounds that are powers of 2")
    print("✓ Array dimensions: tl.zeros((BLOCK_SIZE, BLOCK_SIZE))")
    print("✓ Compile-time flags: HAS_BIAS: tl.constexpr")
    print()
    
    print("When NOT to use constexpr:")
    print("❌ Tensor dimensions (M, N, K) - these vary at runtime")
    print("❌ Tensor data - values change")
    print("❌ Dynamic indices")
    print()
    
    print("Example optimizations enabled by constexpr:")
    print("- Loop unrolling: for i in range(BLOCK_SIZE) → unrolled")
    print("- Memory layout: Arrays sized at compile time")
    print("- Dead code elimination: if HAS_BIAS → branches removed")


@triton.jit
def compiler_hints_demo(
    input_ptr, output_ptr,
    M, N,  # Runtime values - NOT constexpr
    stride_m, stride_n,
    
    # Compile-time constants - MUST be constexpr
    BLOCK_SIZE: tl.constexpr,
    VECTOR_SIZE: tl.constexpr, 
    HAS_BIAS: tl.constexpr,
    EPS: tl.constexpr,
):
    """Demonstrate constexpr usage"""
    
    pid = tl.program_id(0)
    
    # GOOD: Using constexpr for array sizing
    # Compiler knows exact size, can optimize aggressively
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    temp_buffer = tl.zeros((VECTOR_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # GOOD: constexpr enables loop unrolling
    # Compiler will unroll this loop completely
    for i in tl.static_range(VECTOR_SIZE):  # static_range uses constexpr
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        
        data = tl.load(input_ptr + i * stride_m + offsets * stride_n, mask=mask, other=0.0)
        temp_buffer[i, :] = data
    
    # GOOD: Compile-time branching
    # Compiler will eliminate unused branch
    if HAS_BIAS:
        # This branch completely removed if HAS_BIAS=False
        bias_data = tl.load(input_ptr + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
        result = temp_buffer[0, :] + bias_data
    else:
        result = temp_buffer[0, :]
    
    # GOOD: constexpr enables compile-time math
    normalized = result / EPS  # EPS is compile-time constant
    
    # Store result
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    tl.store(output_ptr + offsets, normalized, mask=mask)


# BAD EXAMPLES - What NOT to do
@triton.jit
def bad_examples_kernel(
    input_ptr, output_ptr,
    M, N,
    # WRONG: Making runtime values constexpr
    # M: tl.constexpr,  # BAD - M varies with input tensor size
    # N: tl.constexpr,  # BAD - N varies with input tensor size
):
    """Examples of what NOT to do"""
    
    pid = tl.program_id(0)
    
    # BAD: Using runtime values for array sizing
    # accumulator = tl.zeros((M,), dtype=tl.float32)  # ERROR!
    
    # BAD: Runtime loop bounds prevent unrolling  
    # for i in range(M):  # Can't optimize - M unknown at compile time
    #     pass
    
    # GOOD: Use constexpr properly
    BLOCK_SIZE = 64  # This could be constexpr
    accumulator = tl.zeros((64,), dtype=tl.float32)  # Literal is fine
    
    # Process data...
    offsets = pid * 64 + tl.arange(0, 64)
    mask = offsets < N
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, data, mask=mask)


# =============================================================================
# PRACTICAL EXAMPLE: All 5 Concepts Working Together
# =============================================================================

@triton.jit
def all_concepts_kernel(
    input_ptr, output_ptr, weight_ptr,
    M, N,
    # CONCEPT 1: STRIDES - Manual tensor navigation
    input_stride_0, input_stride_1,
    output_stride_0, output_stride_1,
    # CONCEPT 5: CONSTEXPR - Compile-time optimization
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    SCALE: tl.constexpr,
):
    """
    Practical example showing ALL 5 concepts in one kernel
    Implements: normalized weighted sum per row
    """
    
    # Get current row (one program per row)
    row_idx = tl.program_id(0)
    
    # CONCEPT 2: BROADCASTING - Create 1D column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)  # Shape: (BLOCK_SIZE,)
    
    # CONCEPT 3: MASKING - Safe boundary checking
    col_mask = col_offsets < N
    
    # CONCEPT 1: STRIDES - Manual pointer arithmetic
    # Equivalent to: input_ptr[row_idx, col_offsets]
    input_ptrs = input_ptr + row_idx * input_stride_0 + col_offsets * input_stride_1
    
    # CONCEPT 3: MASKING - Safe load with boundary protection
    row_data = tl.load(input_ptrs, mask=col_mask, other=0.0)
    
    # Convert to float32 for numerical stability
    row_data = row_data.to(tl.float32)
    
    # CONCEPT 4: REDUCTION - Compute row statistics
    row_sum = tl.sum(row_data, axis=0)  # Sum along the row
    row_max = tl.max(row_data, axis=0)  # Max along the row
    row_mean = row_sum / N  # Average
    
    # Normalize: (x - mean) / (max - mean + eps)
    normalized = (row_data - row_mean) / (row_max - row_mean + 1e-8)
    
    # CONCEPT 5: CONSTEXPR - Conditional compilation
    if HAS_WEIGHT:
        # Apply per-element weights
        weight_ptrs = weight_ptr + col_offsets
        weights = tl.load(weight_ptrs, mask=col_mask, other=1.0)
        normalized = normalized * weights.to(tl.float32)
    
    # CONCEPT 5: CONSTEXPR - Compile-time scaling
    output = normalized * SCALE
    
    # CONCEPT 1: STRIDES + CONCEPT 3: MASKING - Safe output
    output_ptrs = output_ptr + row_idx * output_stride_0 + col_offsets * output_stride_1
    tl.store(output_ptrs, output.to(input_ptr.dtype.element_ty), mask=col_mask)


@triton.jit
def matrix_concepts_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    # CONCEPT 1: STRIDES for all matrices
    stride_am, stride_ak,
    stride_bk, stride_bn, 
    stride_cm, stride_cn,
    # CONCEPT 5: CONSTEXPR
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Matrix multiplication showing all concepts in 2D setting
    C = A @ B with all 5 concepts demonstrated
    """
    
    # CONCEPT 2: BROADCASTING - 2D program grid
    pid_m = tl.program_id(0)  # Row block
    pid_n = tl.program_id(1)  # Column block
    
    # Calculate block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # (BLOCK_N,)
    
    # CONCEPT 4: REDUCTION - Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        
        # CONCEPT 2: BROADCASTING + CONCEPT 1: STRIDES
        # A block: (BLOCK_M, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        # B block: (BLOCK_K, BLOCK_N) 
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        
        # CONCEPT 3: MASKING - 2D masks with broadcasting
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        
        # Load blocks
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # CONCEPT 4: REDUCTION - Accumulate partial products
        accumulator += tl.dot(a, b)
    
    # CONCEPT 2: BROADCASTING + CONCEPT 1: STRIDES + CONCEPT 3: MASKING
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator.to(A_ptr.dtype.element_ty), mask=c_mask)


def test_all_concepts():
    """Test kernels demonstrating all 5 concepts"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTesting All 5 Concepts Together")
    print("=" * 40)
    
    if device.type == "cpu":
        print("CUDA not available - showing conceptual examples")
        demonstrate_stride_calculation()
        return
    
    # Test 1: Row-wise normalization (all 5 concepts)
    print("Test 1: Row-wise Normalization")
    M, N = 64, 100  # Non-power-of-2 for masking demo
    input_tensor = torch.randn(M, N, device=device, dtype=torch.float16)
    weight_tensor = torch.randn(N, device=device, dtype=torch.float16)
    output_tensor = torch.empty_like(input_tensor)
    
    BLOCK_SIZE = triton.next_power_of_2(N)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    
    grid = (M,)
    all_concepts_kernel[grid](
        input_tensor, output_tensor, weight_tensor,
        M, N,
        input_tensor.stride(0), input_tensor.stride(1),
        output_tensor.stride(0), output_tensor.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_WEIGHT=True,
        SCALE=2.0,
    )
    
    print(f"✓ Input shape: {input_tensor.shape}")
    print(f"✓ Output shape: {output_tensor.shape}")
    print(f"✓ Used BLOCK_SIZE: {BLOCK_SIZE}")
    print("✓ All 5 concepts working together!")
    
    # Test 2: Small matrix multiplication
    print("\nTest 2: Matrix Multiplication (2D)")
    M, N, K = 64, 64, 32
    A = torch.randn(M, K, device=device, dtype=torch.float16)
    B = torch.randn(K, N, device=device, dtype=torch.float16) 
    C = torch.empty(M, N, device=device, dtype=torch.float16)
    
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 16
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    matrix_concepts_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    # Verify correctness
    C_torch = torch.matmul(A, B)
    max_diff = torch.max(torch.abs(C - C_torch)).item()
    
    print(f"✓ Matrix shapes: {A.shape} @ {B.shape} = {C.shape}")
    print(f"✓ Max difference vs PyTorch: {max_diff:.2e}")
    print(f"✓ Test {'PASSED' if max_diff < 1e-2 else 'FAILED'}")


def demonstrate_stride_calculation():
    """Show stride calculations step by step"""
    print("\nStride Calculation Demo:")
    print("=" * 25)
    
    # Create example tensor
    tensor = torch.randn(3, 4)
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor strides: {tensor.stride()}")
    print(f"Tensor data:\n{tensor}")
    print()
    
    # Show manual indexing
    print("Manual stride calculations:")
    for i in range(3):
        for j in range(4):
            offset = i * tensor.stride(0) + j * tensor.stride(1)
            value = tensor[i, j].item()
            flat_value = tensor.flatten()[offset].item()
            print(f"tensor[{i},{j}] → offset {offset} → value {value:.3f} (flat: {flat_value:.3f})")
    print()
    
    # Show broadcasting example
    print("Broadcasting Example:")
    rows = torch.arange(3)[:, None]  # (3, 1)
    cols = torch.arange(4)[None, :]  # (1, 4)
    print(f"rows[:, None] shape: {rows.shape}")
    print(f"cols[None, :] shape: {cols.shape}")
    print(f"rows[:, None]:\n{rows}")
    print(f"cols[None, :]:\n{cols}")
    
    # Create 2D offset grid
    offset_grid = rows * tensor.stride(0) + cols * tensor.stride(1)
    print(f"2D offset grid:\n{offset_grid}")
    
    print("\n💡 This is exactly what happens in Triton kernels!")
    print("   ptr + offs_m[:, None] * stride_0 + offs_n[None, :] * stride_1")


if __name__ == "__main__":
    print("The 5 Essential Triton Concepts")
    print("=" * 35)
    
    understand_strides()
    understand_broadcasting() 
    understand_masking()
    understand_reductions()
    understand_compiler_hints()
    
    test_all_concepts()
    
    print("\n" + "=" * 50)
    print("MASTERY CHECKLIST:")
    print("✓ I understand how strides map tensor[i,j] to memory")
    print("✓ I can create 2D grids with broadcasting") 
    print("✓ I always use masks for safe memory access")
    print("✓ I know how reductions work along different axes")
    print("✓ I use constexpr for compile-time optimization")
    print("\n🎉 You're ready to write advanced Triton kernels!")