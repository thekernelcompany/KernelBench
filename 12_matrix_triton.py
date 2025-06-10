import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_M': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_M': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_M': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_M': 256}, num_warps=8),
    ],
    key=['N_dim', 'M_dim'],
)
@triton.jit
def diag_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    N_dim, M_dim,
    stride_a_n,  # Stride for A (1D vector)
    stride_b_n, stride_b_m, # Strides for B (2D matrix)
    stride_c_n, stride_c_m, # Strides for C (2D matrix)
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr
):
    """
    Triton kernel for C[i, j] = A[i] * B[i, j].
    A is a 1D vector (diagonal), B and C are 2D matrices.
    """
    pid = tl.program_id(axis=0)

    # Calculate 2D block indices from 1D program ID
    num_m_blocks = tl.cdiv(M_dim, BLOCK_SIZE_M)
    block_idx_n = pid // num_m_blocks
    block_idx_m = pid % num_m_blocks

    # Starting offsets for the current block
    start_n = block_idx_n * BLOCK_SIZE_N
    start_m = block_idx_m * BLOCK_SIZE_M

    # Offsets for elements within the block
    offs_n = start_n + tl.arange(0, BLOCK_SIZE_N) # Row offsets
    offs_m = start_m + tl.arange(0, BLOCK_SIZE_M) # Column offsets

    # Pointers to the current block in B
    # b_ptrs shape: (BLOCK_SIZE_N, BLOCK_SIZE_M)
    b_ptrs = b_ptr + (offs_n[:, None] * stride_b_n + offs_m[None, :] * stride_b_m)
    
    # Pointers to the relevant part of A (vector A[offs_n])
    # a_val_ptrs shape: (BLOCK_SIZE_N,)
    a_val_ptrs = a_ptr + offs_n * stride_a_n

    # Create masks for bounds checking
    # Mask for rows (N dimension)
    mask_n = offs_n < N_dim
    # Mask for columns (M dimension)
    mask_m = offs_m < M_dim

    # Load A values for the current rows, shape (BLOCK_SIZE_N,)
    # Mask with mask_n to avoid out-of-bounds reads if N_dim is not a multiple of BLOCK_SIZE_N
    a_vals = tl.load(a_val_ptrs, mask=mask_n, other=0.0) 

    # Load B values for the current block, shape (BLOCK_SIZE_N, BLOCK_SIZE_M)
    # Mask with combined mask_n and mask_m
    b_vals = tl.load(b_ptrs, mask=mask_n[:, None] & mask_m[None, :], other=0.0)

    # Perform element-wise multiplication: A[i] * B[i, j]
    # a_vals needs to be broadcasted from (BLOCK_SIZE_N,) to (BLOCK_SIZE_N, 1)
    output_block = a_vals[:, None] * b_vals

    # Pointers to the current block in C
    # c_ptrs shape: (BLOCK_SIZE_N, BLOCK_SIZE_M)
    c_ptrs = c_ptr + (offs_n[:, None] * stride_c_n + offs_m[None, :] * stride_c_m)
    
    # Store the result, masked for bounds
    tl.store(c_ptrs, output_block, mask=mask_n[:, None] & mask_m[None, :])

def triton_diag_matmul(A_vec: torch.Tensor, B_mat: torch.Tensor) -> torch.Tensor:
    """
    Python wrapper for the Triton diag_matmul_kernel.
    A_vec: 1D tensor representing the diagonal, shape (N,)
    B_mat: 2D matrix, shape (N, M)
    Output: 2D matrix C, shape (N, M), where C[i,j] = A_vec[i] * B_mat[i,j]
    """
    assert A_vec.is_cuda and B_mat.is_cuda, "Input tensors must be on CUDA device"
    assert A_vec.ndim == 1, "A_vec must be 1D"
    assert B_mat.ndim == 2, "B_mat must be 2D"
    assert A_vec.shape[0] == B_mat.shape[0], "Dimension N of A_vec and B_mat must match"
    assert A_vec.dtype == torch.float32 and B_mat.dtype == torch.float32, "Input tensors must be float32"

    A_vec = A_vec.contiguous()
    B_mat = B_mat.contiguous()

    N_dim = A_vec.shape[0]
    M_dim = B_mat.shape[1]

    C_mat = torch.empty((N_dim, M_dim), device=A_vec.device, dtype=A_vec.dtype)

    grid = lambda meta: (
        triton.cdiv(N_dim, meta['BLOCK_SIZE_N']) * triton.cdiv(M_dim, meta['BLOCK_SIZE_M']),
    )
    
    diag_matmul_kernel[grid](
        A_vec, B_mat, C_mat,
        N_dim, M_dim,
        A_vec.stride(0),
        B_mat.stride(0), B_mat.stride(1),
        C_mat.stride(0), C_mat.stride(1)
        # Autotuner passes BLOCK_SIZE_N, BLOCK_SIZE_M
    )
    return C_mat

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_diag_matmul(A, B) 