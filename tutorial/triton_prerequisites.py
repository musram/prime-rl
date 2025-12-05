"""
Complete Triton Prerequisites and Key Concepts
==============================================

Before diving into complex kernels, you need to understand these fundamentals.
This is your complete checklist for Triton mastery.
"""

import torch
import triton
import triton.language as tl
import math


def gpu_memory_hierarchy():
    """
    CRITICAL: Understanding GPU Memory Hierarchy
    ===========================================
    
    This is THE most important concept for GPU programming.
    Everything else follows from this.
    """
    print("GPU Memory Hierarchy (from fastest to slowest):")
    print("=" * 50)
    print()
    
    print("1. REGISTERS (per thread)")
    print("   - Speed: ~100,000 GB/s") 
    print("   - Size: ~64KB per SM")
    print("   - Access: Immediate (0 cycles)")
    print("   - Usage: Local variables, small arrays")
    print()
    
    print("2. SHARED MEMORY (per SM)")
    print("   - Speed: ~10,000 GB/s")
    print("   - Size: ~48-164KB per SM") 
    print("   - Access: ~1-5 cycles")
    print("   - Usage: Tiles, temporary results")
    print("   - KEY: This is where tiling magic happens!")
    print()
    
    print("3. L2 CACHE (global)")
    print("   - Speed: ~5,000 GB/s")
    print("   - Size: ~40MB")
    print("   - Access: ~200 cycles")
    print("   - Usage: Automatic caching")
    print()
    
    print("4. GLOBAL MEMORY (HBM)")
    print("   - Speed: ~1,500 GB/s") 
    print("   - Size: ~80GB")
    print("   - Access: ~400-600 cycles")
    print("   - Usage: Main tensor storage")
    print("   - PROBLEM: High latency, limited bandwidth")
    print()
    
    print("GOLDEN RULE: Minimize global memory access!")
    print("STRATEGY: Load to shared memory → compute → store back")


def triton_execution_model():
    """
    Understanding Triton's Execution Model
    ====================================
    """
    print("\nTriton Execution Model:")
    print("=" * 25)
    print()
    
    print("1. PROGRAM = THREAD BLOCK")
    print("   - One program = one CUDA thread block")
    print("   - Each program has unique tl.program_id()")
    print("   - Programs run independently and in parallel")
    print()
    
    print("2. GRID = PROGRAM LAYOUT")
    print("   - Grid defines how many programs to launch") 
    print("   - 1D: (num_programs,)")
    print("   - 2D: (programs_x, programs_y)")
    print("   - 3D: (programs_x, programs_y, programs_z)")
    print()
    
    print("3. BLOCK = DATA CHUNK")
    print("   - Each program processes a block of data")
    print("   - Block size determines memory usage")
    print("   - Larger blocks = better efficiency (usually)")
    print()
    
    print("Example: Matrix multiplication")
    print("- Grid: (M//BLOCK_M, N//BLOCK_N)")
    print("- Each program computes one output block")
    print("- program_id(0) = row block, program_id(1) = col block")


def essential_triton_functions():
    """
    Essential Triton Functions You Must Know
    =======================================
    """
    print("\nEssential Triton Functions:")
    print("=" * 30)
    
    functions = [
        ("tl.program_id(axis)", "Get current program ID", "tl.program_id(0)"),
        ("tl.arange(start, end)", "Create index array", "tl.arange(0, 64)"),
        ("tl.load(ptr, mask, other)", "Load with boundary check", "tl.load(ptr, mask=valid, other=0.0)"),
        ("tl.store(ptr, data, mask)", "Store with boundary check", "tl.store(ptr, result, mask=valid)"),
        ("tl.dot(a, b)", "Optimized matrix multiply", "tl.dot(A_block, B_block)"),
        ("tl.sum(x, axis)", "Reduction sum", "tl.sum(x, axis=0)"),
        ("tl.max(x, axis)", "Reduction max", "tl.max(x, axis=1)"),
        ("tl.exp(x)", "Element-wise exp", "tl.exp(logits)"),
        ("tl.sqrt(x)", "Element-wise sqrt", "tl.sqrt(variance)"),
        ("tl.rsqrt(x)", "1/sqrt(x)", "tl.rsqrt(variance + eps)"),
        ("tl.where(cond, a, b)", "Conditional select", "tl.where(mask, x, 0.0)"),
        ("tl.zeros(shape, dtype)", "Create zero tensor", "tl.zeros((64, 64), tl.float32)"),
        ("tl.full(shape, val, dtype)", "Create filled tensor", "tl.full((64,), -float('inf'), tl.float32)"),
        ("x.to(dtype)", "Type conversion", "x.to(tl.float32)"),
        ("tl.cdiv(a, b)", "Ceiling division", "tl.cdiv(1000, 64) = 16"),
        ("tl.multiple_of(x, n)", "Optimization hint", "tl.multiple_of(offset, 16)"),
    ]
    
    for func, desc, example in functions:
        print(f"{func:25} - {desc:25} - {example}")


@triton.jit 
def demonstrate_key_patterns(
    input_ptr, output_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Demonstrate the 5 most important Triton patterns
    """
    # PATTERN 1: Program ID and indexing
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE
    
    # PATTERN 2: Safe indexing with arange + mask
    offsets = row_start + tl.arange(0, BLOCK_SIZE) 
    row_mask = offsets < M
    
    col_offsets = tl.arange(0, N)
    col_mask = col_offsets < N
    
    # PATTERN 3: Pointer arithmetic (broadcasting)
    ptrs = input_ptr + offsets[:, None] * N + col_offsets[None, :]
    mask = row_mask[:, None] & col_mask[None, :]
    
    # PATTERN 4: Masked load/store
    data = tl.load(ptrs, mask=mask, other=0.0)
    
    # PATTERN 5: Type safety
    data = data.to(tl.float32)  # Compute in float32
    result = data * 2.0  # Some computation
    
    output_ptrs = output_ptr + offsets[:, None] * N + col_offsets[None, :]
    tl.store(output_ptrs, result.to(input_ptr.dtype.element_ty), mask=mask)


def common_pitfalls():
    """
    Common Pitfalls and How to Avoid Them
    ====================================
    """
    print("\nCommon Pitfalls:")
    print("=" * 18)
    print()
    
    pitfalls = [
        ("Out-of-bounds access", "Always use masks with tl.load/tl.store", "mask = offsets < size"),
        ("Wrong pointer arithmetic", "Use broadcasting: ptr[i,j] = base + i*stride0 + j*stride1", "ptr + row[:, None]*stride + col[None, :]"),
        ("Type mismatches", "Convert to consistent types", "x.to(tl.float32)"),
        ("Block size too large", "Exceeds shared memory limit", "Keep blocks ≤ 128x128 usually"),
        ("Uncoalesced memory", "Adjacent threads access adjacent memory", "Use proper stride patterns"),
        ("Forgetting constexpr", "Runtime values can't determine array sizes", "BLOCK_SIZE: tl.constexpr"),
        ("Grid size errors", "Wrong number of programs launched", "grid = (triton.cdiv(size, block),)"),
        ("Warp divergence", "Different threads take different branches", "Minimize if/else in kernels"),
    ]
    
    for pitfall, solution, example in pitfalls:
        print(f"❌ {pitfall}")
        print(f"   ✅ {solution}")
        print(f"   📝 {example}")
        print()


def performance_optimization_checklist():
    """
    Performance Optimization Checklist
    =================================
    """
    print("Performance Optimization Checklist:")
    print("=" * 40)
    
    checklist = [
        "✓ Use appropriate block sizes (64-128 usually optimal)",
        "✓ Maximize arithmetic intensity (compute/memory ratio)", 
        "✓ Ensure memory coalescing (adjacent threads → adjacent memory)",
        "✓ Minimize global memory accesses",
        "✓ Use shared memory for data reuse",
        "✓ Convert to float32 for numerical stability",
        "✓ Use constexpr for compile-time optimization",
        "✓ Profile with nsight-compute for bottlenecks",
        "✓ Consider double buffering for advanced cases",
        "✓ Validate correctness before optimizing",
    ]
    
    for item in checklist:
        print(f"  {item}")


def debugging_tips():
    """
    Debugging Triton Kernels
    =======================
    """
    print("\nDebugging Tips:")
    print("=" * 16)
    print()
    
    tips = [
        "Start small: Debug with tiny matrices first",
        "Print shapes: Add print statements in Python wrapper",
        "Check masks: Ensure all memory accesses are masked",
        "Verify strides: Print tensor.stride() to understand layout", 
        "Use torch.allclose(): Compare against PyTorch reference",
        "Single program: Test with grid=(1,) first",
        "Add assertions: Check tensor.is_contiguous()",
        "Gradual complexity: Start simple, add features incrementally",
    ]
    
    for tip in tips:
        print(f"  💡 {tip}")


def what_to_learn_next():
    """
    Learning Path After Mastering Basics
    ===================================
    """
    print("\nLearning Path:")
    print("=" * 15)
    print()
    
    path = [
        ("1. Master tiled matrix multiplication", "Foundation for everything"),
        ("2. Implement element-wise operations", "Broadcasting, masking"),
        ("3. Build reduction kernels", "Sum, max, softmax"),
        ("4. Create normalization layers", "LayerNorm, RMSNorm"),
        ("5. Implement attention mechanisms", "Scaled dot-product attention"),
        ("6. Advanced: Flash Attention", "Memory-efficient attention"),
        ("7. Custom activations", "GELU, SwiGLU, etc."),
        ("8. Quantization kernels", "INT8, FP8 operations"),
        ("9. Advanced optimizations", "Double buffering, async"),
        ("10. Real-world integration", "PyTorch modules, autograd"),
    ]
    
    for step, description in path:
        print(f"{step:35} - {description}")


if __name__ == "__main__":
    print("Triton Prerequisites and Key Concepts")
    print("=" * 40)
    
    # Core concepts
    gpu_memory_hierarchy()
    triton_execution_model() 
    essential_triton_functions()
    
    print()
    common_pitfalls()
    performance_optimization_checklist()
    debugging_tips()
    what_to_learn_next()
    
    print("\n" + "=" * 40)
    print("YOU'RE READY WHEN YOU UNDERSTAND:")
    print("1. GPU memory hierarchy and why tiling matters")
    print("2. How programs map to thread blocks")
    print("3. Safe memory access with masks")
    print("4. Pointer arithmetic with broadcasting")
    print("5. The tiled matrix multiplication pattern")
    print("\nOnce you master these, you can implement any kernel!")