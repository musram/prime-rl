"""
Advanced Triton Concepts & Tricks
=================================

This file demonstrates advanced Triton features beyond the basics:
1. Autotuning (@triton.autotune): Automatically finding the best config.
2. Atomics (tl.atomic_add): Handling race conditions in scatter operations.
3. Advanced Pointer Arithmetic: Handling 2D block pointers and strides manually.
4. Debugging Tricks: Using device_print.
"""

import torch
import triton
import triton.language as tl

# ============================================================================
# 1. Autotuning: The "Free Lunch" of Performance
# ============================================================================
# Instead of guessing BLOCK_SIZE or num_warps, let Triton find it.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'num_warps': 4}, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256, 'num_warps': 8}, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512, 'num_warps': 8}, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024, 'num_warps': 8}, num_stages=4),
    ],
    key=['n_elements'], # Re-tune when this argument changes significantly
)
@triton.jit
def autotuned_relu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0.0) # ReLU
    tl.store(y_ptr + offsets, y, mask=mask)

def test_autotuning(x):
    y = torch.empty_like(x)
    n_elements = x.numel()
    # We don't pass BLOCK_SIZE here; autotuner handles it
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)
    autotuned_relu_kernel[grid](x, y, n_elements)
    return y

# ============================================================================
# 2. Atomic Operations: Scatter / Histogram
# ============================================================================
# Useful when multiple threads might write to the same memory location.

@triton.jit
def histogram_kernel(
    x_ptr,          # Input data
    bins_ptr,       # Output bins
    n_elements,
    n_bins,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data
    val = tl.load(x_ptr + offsets, mask=mask, other=-1.0)
    
    # Filter valid indices (assume data is 0..n_bins-1)
    # We verify bounds effectively by masking the atomic update
    valid_mask = mask & (val >= 0) & (val < n_bins)
    
    # Cast to integer for indexing
    bin_idx = val.to(tl.int32)
    
    # Atomic Add: Safe concurrent update
    # memory location: bins_ptr + bin_idx
    # value to add: 1
    if valid_mask.any(): # Optimization: skip if whole block is invalid
        tl.atomic_add(bins_ptr + bin_idx, 1, mask=valid_mask)

def compute_histogram(x, n_bins):
    bins = torch.zeros(n_bins, device=x.device, dtype=torch.int32)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    histogram_kernel[grid](x, bins, n_elements, n_bins, BLOCK_SIZE=BLOCK_SIZE)
    return bins

# ============================================================================
# 3. 2D Pointer Arithmetic & Tiling (Fused Softmax Example)
# ============================================================================
# Advanced trick: Keep data in SRAM (registers) as long as possible.
# This kernel computes Softmax per row completely in-register (fused).

@triton.jit
def online_softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # Row index
    row_idx = tl.program_id(0)
    
    # Pointers to the start of the row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # Load the entire row into registers (if it fits in BLOCK_SIZE)
    # TRICK: For very large rows, you'd need a loop, but for <4k-8k cols, 
    # loading it all at once is fastest.
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load with 'other=-inf' for correct max calculation with padding
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # 1. Find Max (Reduction)
    row_max = tl.max(row, axis=0)
    
    # 2. Subtract Max & Exp
    numerator = tl.exp(row - row_max)
    
    # 3. Sum (Reduction)
    denominator = tl.sum(numerator, axis=0)
    
    # 4. Divide
    softmax_out = numerator / denominator
    
    # Store
    output_row_start = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start + col_offsets, softmax_out, mask=mask)

# ============================================================================
# 4. Debugging Trick: device_print
# ============================================================================

@triton.jit
def debug_kernel(x_ptr):
    pid = tl.program_id(0)
    if pid == 0: # Only print from one thread/block to avoid spam
        val = tl.load(x_ptr)
        tl.device_print("Value at index 0:", val)

# ============================================================================
# Main Execution for Testing
# ============================================================================

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests.")
        exit()
        
    torch.manual_seed(0)
    device = "cuda"
    
    print("--- Testing Autotuning ---")
    x = torch.randn(4096, device=device)
    y = test_autotuning(x)
    print("Autotuned ReLU success")
    
    print("\n--- Testing Atomics (Histogram) ---")
    data = torch.randint(0, 10, (1000,), device=device, dtype=torch.float32)
    hist = compute_histogram(data, 10)
    print(f"Histogram sum (should be 1000): {hist.sum().item()}")
    
    print("\n--- Testing Fused Softmax ---")
    rows, cols = 16, 1024
    x_mat = torch.randn(rows, cols, device=device)
    y_mat = torch.empty_like(x_mat)
    
    # Next power of 2 for block size
    BLOCK_SIZE = triton.next_power_of_2(cols)
    
    online_softmax_kernel[(rows,)](
        y_mat, x_mat,
        x_mat.stride(0), y_mat.stride(0),
        cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Verify
    y_torch = torch.softmax(x_mat, dim=1)
    print(f"Softmax Max Diff: {(y_mat - y_torch).abs().max().item()}")
    
    print("\nAll Advanced Examples Ran Successfully.")

