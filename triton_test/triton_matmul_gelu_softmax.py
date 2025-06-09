import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

# --- Kernel 1: Matmul + Bias + GELU ---
@triton.jit
def matmul_bias_gelu_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_x_m, stride_x_k,
    stride_w_k, stride_w_n, # weight_ptr is to W_transposed (K,N)
    stride_o_m, stride_o_n,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GELU_CONST_INV_SQRT_2: tl.constexpr # New constant for erf-GELU
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = (pid // num_pid_n) % num_pid_m
    pid_n = pid % num_pid_n

    offs_m_block = tl.arange(0, BLOCK_SIZE_M)
    offs_n_block = tl.arange(0, BLOCK_SIZE_N)
    offs_k_block = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the current block in X and W.T
    # offs_m are absolute row indices for the X block
    offs_x_m = pid_m * BLOCK_SIZE_M + offs_m_block
    # offs_w_n are absolute col indices for the W.T block
    offs_w_n = pid_n * BLOCK_SIZE_N + offs_n_block

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        # Current K offset for this iteration
        current_k_offs = k_start + offs_k_block
        
        # Boundary checks for K
        k_mask = current_k_offs < K

        # Load block of X: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # x_ptrs = x_ptr + (offs_x_m[:, None] * stride_x_m + current_k_offs[None, :] * stride_x_k)
        # Mask for X loading
        x_load_mask = (offs_x_m[:, None] < M) & (k_mask[None, :])
        x_block_ptrs = x_ptr + offs_x_m[:, None] * stride_x_m + current_k_offs[None, :] * stride_x_k
        x_block = tl.load(x_block_ptrs, mask=x_load_mask, other=0.0)

        # Load block of W.T: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        # w_ptrs = weight_ptr + (current_k_offs[:, None] * stride_w_k + offs_w_n[None, :] * stride_w_n)
        # Mask for W.T loading
        w_load_mask = (k_mask[:, None]) & (offs_w_n[None, :] < N)
        w_block_ptrs = weight_ptr + current_k_offs[:, None] * stride_w_k + offs_w_n[None,:] * stride_w_n
        w_block = tl.load(w_block_ptrs, mask=w_load_mask, other=0.0)
        
        accumulator += tl.dot(x_block, w_block)

    # Add bias
    # Bias is 1D tensor of size N. offs_w_n are effective column indices for output block
    bias_load_mask = offs_w_n < N
    bias_vals_ptr = bias_ptr + offs_w_n
    bias_block = tl.load(bias_vals_ptr, mask=bias_load_mask, other=0.0)
    accumulator += bias_block[None, :]

    # GELU Activation (erf based)
    # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    # accumulator is 'x' in the formula above
    x_div_sqrt2 = accumulator * GELU_CONST_INV_SQRT_2
    erf_val = tl.math.erf(x_div_sqrt2) # Use tl.math.erf
    gelu_output = 0.5 * accumulator * (1.0 + erf_val)
    
    # Write output
    output_offs_m = pid_m * BLOCK_SIZE_M + offs_m_block
    output_offs_n = pid_n * BLOCK_SIZE_N + offs_n_block
    
    output_ptrs = output_ptr + output_offs_m[:, None] * stride_o_m + output_offs_n[None, :] * stride_o_n
    output_mask = (output_offs_m[:, None] < M) & (output_offs_n[None, :] < N)
    tl.store(output_ptrs, gelu_output, mask=output_mask)

# --- Kernel 2: Softmax ---
@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    num_rows, num_cols,
    stride_row_in, stride_col_in,
    stride_row_out, stride_col_out,
    BLOCK_SIZE_COLS: tl.constexpr # Block size for processing columns
):
    row_idx = tl.program_id(axis=0)

    row_start_ptr_in = input_ptr + row_idx * stride_row_in
    row_start_ptr_out = output_ptr + row_idx * stride_row_out
    
    # Pass 1: Find max value in the row
    max_val = -float('inf')
    col_arange = tl.arange(0, BLOCK_SIZE_COLS)
    for col_block_start in range(0, tl.cdiv(num_cols, BLOCK_SIZE_COLS)):
        actual_cols = col_block_start * BLOCK_SIZE_COLS + col_arange
        col_mask = actual_cols < num_cols
        
        x_block = tl.load(row_start_ptr_in + actual_cols * stride_col_in, mask=col_mask, other=-float('inf'))
        block_max = tl.max(x_block, axis=0) # tl.max on 1D tensor gives scalar
        max_val = tl.maximum(max_val, block_max)
        
    # Pass 2: Calculate exp(x - max_val) and sum them up (denominator)
    # Also store exp(x - max_val) into output_ptr temporarily
    sum_exp_val = 0.0
    for col_block_start in range(0, tl.cdiv(num_cols, BLOCK_SIZE_COLS)):
        actual_cols = col_block_start * BLOCK_SIZE_COLS + col_arange
        col_mask = actual_cols < num_cols

        x_block = tl.load(row_start_ptr_in + actual_cols * stride_col_in, mask=col_mask, other=0.0) # other=0 won't affect sum if masked
        shifted_x = x_block - max_val
        exp_block = tl.exp(shifted_x)
        
        # Store temporary exp values
        tl.store(row_start_ptr_out + actual_cols * stride_col_out, exp_block, mask=col_mask)
        
        # Accumulate sum for denominator, only for valid elements
        sum_exp_val += tl.sum(tl.where(col_mask, exp_block, 0.0), axis=0)

    # Pass 3: Divide stored exp(x - max_val) by sum_exp_val
    # Add epsilon to denominator for numerical stability, though sum_exp_val should be >0 if any input is not -inf
    epsilon = 1e-6 
    for col_block_start in range(0, tl.cdiv(num_cols, BLOCK_SIZE_COLS)):
        actual_cols = col_block_start * BLOCK_SIZE_COLS + col_arange
        col_mask = actual_cols < num_cols
        
        exp_val_block = tl.load(row_start_ptr_out + actual_cols * stride_col_out, mask=col_mask, other=0.0)
        norm_block = exp_val_block / (sum_exp_val + epsilon)
        tl.store(row_start_ptr_out + actual_cols * stride_col_out, norm_block, mask=col_mask)

# --- Python Wrappers ---
def triton_matmul_bias_gelu_wrapper(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All input tensors must be on CUDA"
    # PyTorch nn.Linear: output = x @ weight.T + bias
    # x: (M, K), weight: (N, K), bias: (N,)
    # We need to pass weight.T to Triton kernel, which is (K, N)
    M, K_x = x.shape
    N_w, K_w = weight.shape 
    assert K_x == K_w, f"Weight K dim {K_w} must match X K dim {K_x}"
    K = K_x
    N = N_w
    
    weight_t = weight.T.contiguous() # Shape (K, N)

    output = torch.empty((M, N), device=x.device, dtype=x.dtype)

    GELU_CONST_INV_SQRT_2 = 1.0 / math.sqrt(2.0)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    
    matmul_bias_gelu_kernel[grid](
        x, weight_t, bias, output,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32,
        GELU_CONST_INV_SQRT_2=GELU_CONST_INV_SQRT_2
    )
    return output

def triton_softmax_wrapper(input_tensor: torch.Tensor):
    assert input_tensor.is_cuda, "Input tensor must be on CUDA"
    num_rows, num_cols = input_tensor.shape
    output_tensor = torch.empty_like(input_tensor)
    
    grid = (num_rows,) # One program per row
    
    # Determine a reasonable BLOCK_SIZE_COLS for softmax
    # For small num_cols, make it at least num_cols, power of 2 if possible
    if num_cols <= 16: BLOCK_SIZE_COLS = 16
    elif num_cols <= 32: BLOCK_SIZE_COLS = 32
    elif num_cols <= 64: BLOCK_SIZE_COLS = 64
    elif num_cols <= 128: BLOCK_SIZE_COLS = 128
    elif num_cols <= 256: BLOCK_SIZE_COLS = 256
    elif num_cols <= 512: BLOCK_SIZE_COLS = 512
    else: BLOCK_SIZE_COLS = 1024

    softmax_kernel[grid](
        input_tensor, output_tensor,
        num_rows, num_cols,
        input_tensor.stride(0), input_tensor.stride(1),
        output_tensor.stride(0), output_tensor.stride(1),
        BLOCK_SIZE_COLS=BLOCK_SIZE_COLS
    )
    return output_tensor

# --- ModelNew Class ---
class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        # These parameters will be synchronized by KernelBench with the reference model's nn.Linear layer
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
        self._initialize_parameters(in_features) # Custom init, KernelBench should override state_dict

    def _initialize_parameters(self, in_features):
        # Initialize to something, though KernelBench usually sets state_dict from reference
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            # fan_in for bias is in_features based on weight (out_features, in_features)
            bound = 1 / math.sqrt(in_features) if in_features > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input x is compatible with weight dtype (float32)
        x_casted = x.to(dtype=self.weight.dtype, device=self.weight.device)
        
        intermediate = triton_matmul_bias_gelu_wrapper(x_casted, self.weight, self.bias)
        output = triton_softmax_wrapper(intermediate)
        return output 