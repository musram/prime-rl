import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides for each dimension
    stride_am, stride_ak,  # A is (M, K)
    stride_bk, stride_bn,  # B is (K, N) 
    stride_cm, stride_cn,  # C is (M, N)
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Basic matrix multiplication kernel: C = A @ B
    
    Key Triton concepts:
    1. tl.program_id() - gets the current "program" (thread block) ID
    2. tl.arange() - creates a range of indices
    3. tl.load() / tl.store() - memory operations with masking
    4. tl.dot() - matrix multiplication
    """
    
    # Get the program IDs (which block of the output we're computing)
    pid_m = tl.program_id(0)  # Row block index
    pid_n = tl.program_id(1)  # Column block index
    
    # Compute the row and column indices this program will handle
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # [0, 1, ..., BLOCK_SIZE_M-1] + offset
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # [0, 1, ..., BLOCK_SIZE_N-1] + offset
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator for this block
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Calculate pointers for current A and B blocks
        # A[offs_m, offs_k] - shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        # B[offs_k, offs_n] - shape (BLOCK_SIZE_K, BLOCK_SIZE_N)  
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        
        # Load blocks with boundary checking
        # mask ensures we don't read out-of-bounds memory
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        
        # Compute partial matrix multiplication and accumulate
        accumulator += tl.dot(a, b)
        
        # Move to next K block
        offs_k += BLOCK_SIZE_K
    
    # Store result
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def triton_matmul(a, b):
    """
    Wrapper function for Triton matrix multiplication
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    
    M, K = a.shape
    K, N = b.shape
    
    # Create output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Define block sizes (tunable hyperparameters)
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64  
    BLOCK_SIZE_K = 32
    
    # Calculate grid size (how many programs/blocks we need)
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),  # Number of row blocks
        triton.cdiv(N, BLOCK_SIZE_N),  # Number of column blocks
    )
    
    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),  # A strides
        b.stride(0), b.stride(1),  # B strides  
        c.stride(0), c.stride(1),  # C strides
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    )
    
    return c


if __name__ == "__main__":
    # Test our implementation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create test matrices
    M, N, K = 128, 128, 128
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float16)
    
    # Triton implementation
    c_triton = triton_matmul(a, b)
    
    # PyTorch reference
    c_torch = torch.matmul(a, b)
    
    # Check correctness
    max_diff = torch.max(torch.abs(c_triton - c_torch)).item()
    print(f"Max difference: {max_diff}")
    print(f"Test {'PASSED' if max_diff < 1e-2 else 'FAILED'}")
    
    # Print shapes for understanding
    print(f"\nMatrix shapes:")
    print(f"A: {a.shape}")
    print(f"B: {b.shape}") 
    print(f"C: {c_triton.shape}")