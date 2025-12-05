"""
Practical Triton Kernel Examples
================================

Additional examples to practice and understand Triton kernels.
"""

import torch
import triton
import triton.language as tl
import math

# ============================================================================
# Example 1: Element-wise Operations
# ============================================================================

@triton.jit
def relu_kernel(
    x_ptr,
    y_ptr, 
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """ReLU activation: y = max(0, x)"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0.0)  # ReLU operation
    tl.store(y_ptr + offsets, y, mask=mask)

def relu_triton(x):
    y = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    relu_kernel[grid](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return y

@triton.jit
def gelu_kernel(
    x_ptr,
    y_ptr,
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    """GELU activation: y = x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # GELU approximation
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/π)
    coeff = 0.044715
    
    inner = sqrt_2_over_pi * (x + coeff * x * x * x)
    y = 0.5 * x * (1.0 + tl.libdevice.tanh(inner))
    
    tl.store(y_ptr + offsets, y, mask=mask)

def gelu_triton(x):
    y = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    gelu_kernel[grid](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return y

# ============================================================================
# Example 2: Reduction Operations
# ============================================================================

@triton.jit
def sum_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Sum along rows (reduce columns)"""
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
        
    row_start = row_idx * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    acc = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + i + col_offsets
        mask = (i + col_offsets) < n_cols
        
        vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        acc += tl.sum(vals)
    
    tl.store(output_ptr + row_idx, acc)

def sum_rows_triton(x):
    """Sum each row independently"""
    n_rows, n_cols = x.shape
    y = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE = triton.next_power_of_2(min(n_cols, 1024))
    
    sum_kernel[(n_rows,)](
        x, y, n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y

# ============================================================================
# Example 3: Matrix Transpose
# ============================================================================

@triton.jit 
def transpose_kernel(
    input_ptr,
    output_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Efficient matrix transpose with tiling"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate tile coordinates
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Input coordinates: (rm, rn)
    input_ptrs = input_ptr + rm[:, None] * N + rn[None, :]
    mask_in = (rm[:, None] < M) & (rn[None, :] < N)
    
    # Load tile
    tile = tl.load(input_ptrs, mask=mask_in)
    
    # Output coordinates: (rn, rm) - transposed
    output_ptrs = output_ptr + rn[:, None] * M + rm[None, :]  
    mask_out = (rn[:, None] < N) & (rm[None, :] < M)
    
    # Store transposed tile
    tl.store(output_ptrs, tl.trans(tile), mask=mask_out)

def transpose_triton(x):
    M, N = x.shape
    y = torch.empty(N, M, device=x.device, dtype=x.dtype)
    
    BLOCK_M = 32
    BLOCK_N = 32
    
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N)
    )
    
    transpose_kernel[grid](
        x, y, M, N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N
    )
    return y

# ============================================================================
# Example 4: Layer Normalization
# ============================================================================

@triton.jit
def layernorm_kernel(
    output_ptr,
    input_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rstd_ptr,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Layer normalization kernel"""
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
        
    # Calculate row start
    row_start = row_idx * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    # Load input row
    input_ptrs = input_ptr + row_start + offsets
    x = tl.load(input_ptrs, mask=mask, other=0.0)
    
    # Calculate mean
    mean = tl.sum(x) / n_cols
    
    # Calculate variance
    centered = x - mean
    variance = tl.sum(centered * centered) / n_cols
    rstd = 1.0 / tl.sqrt(variance + eps)
    
    # Normalize
    normalized = centered * rstd
    
    # Load weight and bias
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    # Apply affine transformation
    output = normalized * weight + bias
    
    # Store outputs
    output_ptrs = output_ptr + row_start + offsets
    tl.store(output_ptrs, output, mask=mask)
    tl.store(mean_ptr + row_idx, mean)
    tl.store(rstd_ptr + row_idx, rstd)

def layernorm_triton(x, weight, bias, eps=1e-5):
    n_rows, n_cols = x.shape
    
    # Allocate outputs
    y = torch.empty_like(x)
    mean = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    rstd = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    layernorm_kernel[(n_rows,)](
        y, x, weight, bias, mean, rstd,
        n_rows, n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y, mean, rstd

# ============================================================================
# Example 5: Fused Linear + Bias + Activation  
# ============================================================================

@triton.jit
def fused_linear_relu_kernel(
    output_ptr,
    input_ptr, 
    weight_ptr,
    bias_ptr,
    M, N, K,  # input: (M, K), weight: (K, N), bias: (N,), output: (M, N)
    stride_im, stride_ik,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr,
):
    """Fused linear layer with bias and ReLU activation"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Matrix multiplication
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        rk = k * BLOCK_K + tl.arange(0, BLOCK_K)
        
        # Load input and weight tiles
        input_ptrs = input_ptr + (rm[:, None] * stride_im + rk[None, :] * stride_ik)
        weight_ptrs = weight_ptr + (rk[:, None] * stride_wk + rn[None, :] * stride_wn)
        
        mask_input = (rm[:, None] < M) & (rk[None, :] < K)
        mask_weight = (rk[:, None] < K) & (rn[None, :] < N)
        
        input_tile = tl.load(input_ptrs, mask=mask_input, other=0.0)
        weight_tile = tl.load(weight_ptrs, mask=mask_weight, other=0.0)
        
        acc += tl.dot(input_tile, weight_tile)
    
    # Add bias
    bias_ptrs = bias_ptr + rn
    mask_bias = rn < N
    bias = tl.load(bias_ptrs, mask=mask_bias, other=0.0)
    acc += bias[None, :]
    
    # Apply ReLU
    acc = tl.maximum(acc, 0.0)
    
    # Store result
    output_ptrs = output_ptr + (rm[:, None] * stride_om + rn[None, :] * stride_on)
    mask_output = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(output_ptrs, acc, mask=mask_output)

def fused_linear_relu_triton(input, weight, bias):
    """Fused linear + bias + ReLU"""
    M, K = input.shape
    K, N = weight.shape
    
    output = torch.empty((M, N), device=input.device, dtype=input.dtype)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N)
    )
    
    fused_linear_relu_kernel[grid](
        output, input, weight, bias,
        M, N, K,
        input.stride(0), input.stride(1),
        weight.stride(0), weight.stride(1), 
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K
    )
    
    return output

# ============================================================================
# Testing and Demo
# ============================================================================

def test_examples():
    """Test all example kernels"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}")
    
    if device == "cpu":
        print("Triton kernels require CUDA. Testing PyTorch equivalents.")
        return
        
    # Test ReLU
    x = torch.randn(1000, device=device)
    y_triton = relu_triton(x)
    y_torch = torch.relu(x)
    print(f"ReLU correctness: {torch.allclose(y_triton, y_torch)}")
    
    # Test GELU  
    y_triton = gelu_triton(x)
    y_torch = torch.nn.functional.gelu(x)
    print(f"GELU correctness: {torch.allclose(y_triton, y_torch, rtol=1e-3)}")
    
    # Test sum reduction
    x = torch.randn(100, 50, device=device)
    y_triton = sum_rows_triton(x) 
    y_torch = torch.sum(x, dim=1)
    print(f"Sum reduction correctness: {torch.allclose(y_triton, y_torch)}")
    
    # Test transpose
    x = torch.randn(64, 128, device=device)
    y_triton = transpose_triton(x)
    y_torch = x.t()
    print(f"Transpose correctness: {torch.allclose(y_triton, y_torch)}")
    
    # Test layer norm
    x = torch.randn(32, 256, device=device)
    weight = torch.randn(256, device=device)
    bias = torch.randn(256, device=device)
    
    y_triton, _, _ = layernorm_triton(x, weight, bias)
    y_torch = torch.nn.functional.layer_norm(x, (256,), weight, bias)
    print(f"LayerNorm correctness: {torch.allclose(y_triton, y_torch, rtol=1e-3)}")
    
    # Test fused linear + ReLU
    input = torch.randn(32, 64, device=device)
    weight = torch.randn(64, 128, device=device)  
    bias = torch.randn(128, device=device)
    
    y_triton = fused_linear_relu_triton(input, weight, bias)
    y_torch = torch.relu(torch.addmm(bias, input, weight))
    print(f"Fused Linear+ReLU correctness: {torch.allclose(y_triton, y_torch, rtol=1e-3)}")

if __name__ == "__main__":
    test_examples()