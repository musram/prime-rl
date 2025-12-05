"""
Unsloth-Style Triton Kernel Tutorial
===================================

Learn to write Triton kernels using patterns from Unsloth.
We'll implement 3 examples with increasing complexity:

1. Vector Addition (Basic Pattern)
2. RMS LayerNorm (Reduction Pattern) 
3. Simple Attention (Matrix Pattern)
"""

import torch
import triton
import triton.language as tl
import math


# =============================================================================
# Example 1: Vector Addition (Basic Pattern)
# =============================================================================

def calculate_settings_1d(n_elements):
    """Unsloth pattern: Calculate optimal block size for 1D operations"""
    BLOCK_SIZE = triton.next_power_of_2(n_elements)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)  # Cap at 1024 for this example
    
    if BLOCK_SIZE >= 512:
        num_warps = 8
    elif BLOCK_SIZE >= 256:
        num_warps = 4
    else:
        num_warps = 2
        
    return BLOCK_SIZE, num_warps


@triton.jit
def vector_add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Unsloth Pattern 1: Basic element-wise operation
    
    Key concepts:
    - tl.program_id(0): Get current program ID  
    - tl.arange(): Create index range
    - Masking for boundary safety
    """
    
    # Pattern: Get program ID and calculate offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Pattern: Create mask for safe memory access
    mask = offsets < n_elements
    
    # Pattern: Load with masking
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Pattern: Compute and store
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_vector_add(x, y):
    """Unsloth-style wrapper for vector addition"""
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Unsloth pattern: Dynamic configuration
    BLOCK_SIZE, num_warps = calculate_settings_1d(n_elements)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Unsloth pattern: Kernel launch
    vector_add_kernel[grid](
        x, y, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    return output


# =============================================================================
# Example 2: RMS LayerNorm (Reduction Pattern)
# =============================================================================

def calculate_settings_2d(n_cols):
    """Unsloth pattern: 2D block size calculation"""
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 2048)
    
    if BLOCK_SIZE >= 2048:
        num_warps = 16
    elif BLOCK_SIZE >= 1024:
        num_warps = 8
    elif BLOCK_SIZE >= 512:
        num_warps = 4
    else:
        num_warps = 2
        
    return BLOCK_SIZE, num_warps


@triton.jit
def rms_layernorm_kernel(
    input_ptr, weight_ptr, output_ptr,
    n_rows, n_cols,
    input_row_stride, weight_stride, output_row_stride,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Unsloth Pattern 2: Row-wise reduction operation
    
    Key concepts:
    - Row-level parallelism with program_id(0)
    - Reduction operations (sum, mean)
    - Float32 for numerical stability
    """
    
    # Pattern: One program per row
    row_idx = tl.program_id(0)
    
    # Pattern: Column offsets and masking
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Pattern: Calculate pointers for current row
    input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
    weight_ptrs = weight_ptr + col_offsets * weight_stride
    output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
    
    # Pattern: Load and convert to float32
    input_data = tl.load(input_ptrs, mask=mask, other=0.0)
    weight = tl.load(weight_ptrs, mask=mask, other=0.0)
    input_data = input_data.to(tl.float32)
    weight = weight.to(tl.float32)
    
    # Pattern: Reduction computation
    variance = tl.sum(input_data * input_data, axis=0) / n_cols
    inv_var = tl.rsqrt(variance + eps)
    
    # Pattern: Normalize and apply weight
    normalized = input_data * inv_var
    output = normalized * weight
    
    # Pattern: Store with dtype conversion
    tl.store(output_ptrs, output.to(input_ptr.dtype.element_ty), mask=mask)


class RMSLayerNorm(torch.autograd.Function):
    """Unsloth pattern: Autograd integration"""
    
    @staticmethod
    def forward(ctx, input_tensor, weight, eps=1e-6):
        input_tensor = input_tensor.contiguous()
        output = torch.empty_like(input_tensor)
        
        n_rows, n_cols = input_tensor.shape
        BLOCK_SIZE, num_warps = calculate_settings_2d(n_cols)
        
        grid = (n_rows,)
        
        rms_layernorm_kernel[grid](
            input_tensor, weight, output,
            n_rows, n_cols,
            input_tensor.stride(0), weight.stride(0), output.stride(0),
            eps, BLOCK_SIZE,
            num_warps=num_warps,
        )
        
        return output


def triton_rms_layernorm(x, weight, eps=1e-6):
    """Wrapper function"""
    return RMSLayerNorm.apply(x, weight, eps)


# =============================================================================
# Example 3: Simple Attention QK^T (Matrix Pattern)
# =============================================================================

@triton.jit
def attention_qk_kernel(
    q_ptr, k_ptr, output_ptr,
    seq_len, head_dim,
    q_row_stride, q_col_stride,
    k_row_stride, k_col_stride, 
    output_row_stride, output_col_stride,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Unsloth Pattern 3: Matrix multiplication pattern
    
    Key concepts:
    - 2D program grid for matrix blocks
    - Tiled matrix multiplication
    - Accumulator pattern
    """
    
    # Pattern: 2D block indices
    pid_m = tl.program_id(0)  # Row block
    pid_n = tl.program_id(1)  # Column block
    
    # Pattern: Block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pattern: Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Pattern: Tiled multiplication loop
    for k in range(0, tl.cdiv(head_dim, BLOCK_K)):
        # Load Q block: [BLOCK_M, BLOCK_K]
        q_ptrs = q_ptr + (offs_m[:, None] * q_row_stride + offs_k[None, :] * q_col_stride)
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len) & (offs_k[None, :] < head_dim), other=0.0)
        
        # Load K block: [BLOCK_K, BLOCK_N] (transposed)
        k_ptrs = k_ptr + (offs_k[:, None] * k_row_stride + offs_n[None, :] * k_col_stride)
        k = tl.load(k_ptrs, mask=(offs_k[:, None] < head_dim) & (offs_n[None, :] < seq_len), other=0.0)
        
        # Pattern: Matrix multiply and accumulate
        accumulator += tl.dot(q, k)
        
        # Move to next block
        offs_k += BLOCK_K
    
    # Pattern: Apply scaling and store
    output = accumulator * scale
    
    output_ptrs = output_ptr + (offs_m[:, None] * output_row_stride + offs_n[None, :] * output_col_stride)
    output_mask = (offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len)
    tl.store(output_ptrs, output.to(q_ptr.dtype.element_ty), mask=output_mask)


def triton_attention_qk(q, k):
    """Compute Q @ K^T with scaling"""
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Reshape for matrix multiplication
    q_2d = q.view(-1, head_dim)  # [batch*heads*seq, head_dim]
    k_2d = k.view(-1, head_dim)  # [batch*heads*seq, head_dim]
    
    output = torch.empty(batch_size * num_heads, seq_len, seq_len, 
                        device=q.device, dtype=q.dtype)
    
    scale = 1.0 / math.sqrt(head_dim)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    
    grid = (
        triton.cdiv(seq_len, BLOCK_M),
        triton.cdiv(seq_len, BLOCK_N), 
        batch_size * num_heads,
    )
    
    attention_qk_kernel[grid](
        q_2d, k_2d, output,
        seq_len, head_dim,
        q_2d.stride(0), q_2d.stride(1),
        k_2d.stride(0), k_2d.stride(1),
        output.stride(1), output.stride(2),
        scale,
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    
    return output.view(batch_size, num_heads, seq_len, seq_len)


# =============================================================================
# Testing and Examples
# =============================================================================

def test_all_kernels():
    """Test all three kernel examples"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")
    
    # Test 1: Vector Addition
    print("\n1. Testing Vector Addition...")
    x = torch.randn(1000, device=device, dtype=torch.float32)
    y = torch.randn(1000, device=device, dtype=torch.float32)
    
    result_triton = triton_vector_add(x, y)
    result_torch = x + y
    
    max_diff = torch.max(torch.abs(result_triton - result_torch)).item()
    print(f"   Max difference: {max_diff:.6f}")
    print(f"   Test {'PASSED' if max_diff < 1e-5 else 'FAILED'}")
    
    # Test 2: RMS LayerNorm
    print("\n2. Testing RMS LayerNorm...")
    x = torch.randn(128, 512, device=device, dtype=torch.float16)
    weight = torch.randn(512, device=device, dtype=torch.float16)
    
    result_triton = triton_rms_layernorm(x, weight)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {result_triton.shape}")
    print("   ✓ RMS LayerNorm completed")
    
    # Test 3: Attention QK^T  
    print("\n3. Testing Attention QK^T...")
    batch, heads, seq_len, head_dim = 2, 8, 256, 64
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
    
    result_triton = triton_attention_qk(q, k)
    print(f"   Q shape: {q.shape}")
    print(f"   K shape: {k.shape}")
    print(f"   Output shape: {result_triton.shape}")
    print("   ✓ Attention QK^T completed")


if __name__ == "__main__":
    print("Unsloth-Style Triton Tutorial")
    print("=" * 40)
    
    test_all_kernels()
    
    print("\n" + "=" * 40)
    print("Key Takeaways:")
    print("1. Use calculate_settings() for optimal block sizing")
    print("2. Always mask memory operations for safety") 
    print("3. Convert to float32 for numerical stability")
    print("4. One program per row for element-wise ops")
    print("5. 2D grid for matrix operations")
    print("6. Accumulator pattern for reductions")
    print("7. Autograd integration for differentiable ops")