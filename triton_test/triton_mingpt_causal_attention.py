import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

# --- Triton Kernel 1: Linear Projection (Matmul + Bias) ---
@triton.jit
def linear_projection_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K, # Input X (M, K), Weight W (N, K), Output O (M, N)
    stride_x_m, stride_x_k,
    stride_w_n, stride_w_k, # weight_ptr is to W (N,K) (out_features, in_features)
    stride_o_m, stride_o_n,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    ADD_BIAS: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = (pid // num_pid_n) % num_pid_m
    pid_n = pid % num_pid_n

    offs_m_block = tl.arange(0, BLOCK_SIZE_M)
    offs_n_block = tl.arange(0, BLOCK_SIZE_N)
    offs_k_block = tl.arange(0, BLOCK_SIZE_K)

    # offs for the current block of X
    offs_x_m = pid_m * BLOCK_SIZE_M + offs_m_block
    # offs for the current block of W (N-dim) and Bias
    offs_w_n_bias = pid_n * BLOCK_SIZE_N + offs_n_block

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        current_k_offs = k_start + offs_k_block # K-dim offsets for this iteration
        k_mask_iter = current_k_offs < K

        # Load X block: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        x_load_mask = (offs_x_m[:, None] < M) & k_mask_iter[None, :]
        x_block_ptrs = x_ptr + offs_x_m[:, None] * stride_x_m + current_k_offs[None, :] * stride_x_k
        x_block = tl.load(x_block_ptrs, mask=x_load_mask, other=0.0)

        # Load W block, effectively W.T: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        # W is (N,K). offs_w_n_bias are for rows of W (N-dim), current_k_offs are for cols of W (K-dim).
        # We want to load a (K_BLK, N_BLK) tile from W, to perform X_BLK @ W_BLK_T
        w_load_mask = k_mask_iter[:, None] & (offs_w_n_bias[None, :] < N)
        w_block_ptrs = weight_ptr + current_k_offs[:, None] * stride_w_k + offs_w_n_bias[None, :] * stride_w_n
        w_block_transposed_tile = tl.load(w_block_ptrs, mask=w_load_mask, other=0.0)
        
        accumulator += tl.dot(x_block, w_block_transposed_tile)

    if ADD_BIAS:
        bias_load_mask = offs_w_n_bias < N
        bias_vals_ptr = bias_ptr + offs_w_n_bias
        bias_block = tl.load(bias_vals_ptr, mask=bias_load_mask, other=0.0)
        accumulator += bias_block[None, :] # Add to each row of accumulator block
    
    output_offs_m = pid_m * BLOCK_SIZE_M + offs_m_block
    output_offs_n = pid_n * BLOCK_SIZE_N + offs_n_block
    output_ptrs = output_ptr + output_offs_m[:, None] * stride_o_m + output_offs_n[None, :] * stride_o_n
    output_mask = (output_offs_m[:, None] < M) & (output_offs_n[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=output_mask)

# --- Triton Kernel 2: Batched Matrix Multiplication (BMM) ---
@triton.jit
def bmm_kernel(
    a_ptr, b_ptr, output_ptr,
    B, H, M, N, K, # A: (B,H,M,K), B_orig: (B,H,T_k,H_k) (shape of K or V)
                   # if TRANSPOSE_B=True for QK.T: B_orig is K (B,H,T,hs), effectively B.T (B,H,hs,T)
                   # Op is A(M,K) @ B_eff(K,N). N is T for QK.T, N is hs for Att@V.
    stride_a_b, stride_a_h, stride_a_m, stride_a_k,
    stride_b_orig_b, stride_b_orig_h, stride_b_orig_dim1, stride_b_orig_dim2, # Strides for original B tensor
    stride_o_b, stride_o_h, stride_o_m, stride_o_n,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    TRANSPOSE_B: tl.constexpr
):
    pid_bh = tl.program_id(axis=0) # Batch and Head dimensions combined
    pid_mn_flat = tl.program_id(axis=1) # M and N dimensions combined for output block

    batch_idx = pid_bh // H
    head_idx = pid_bh % H

    num_m_blocks = tl.cdiv(M, BLOCK_SIZE_M)
    num_n_blocks = tl.cdiv(N, BLOCK_SIZE_N)
    
    block_m_idx = pid_mn_flat // num_n_blocks
    block_n_idx = pid_mn_flat % num_n_blocks
    
    offs_m_block = tl.arange(0, BLOCK_SIZE_M)
    offs_n_block = tl.arange(0, BLOCK_SIZE_N)
    offs_k_block = tl.arange(0, BLOCK_SIZE_K)

    a_start_ptr = a_ptr + batch_idx * stride_a_b + head_idx * stride_a_h
    b_orig_start_ptr = b_ptr + batch_idx * stride_b_orig_b + head_idx * stride_b_orig_h
    output_start_ptr = output_ptr + batch_idx * stride_o_b + head_idx * stride_o_h

    current_m_offs = block_m_idx * BLOCK_SIZE_M + offs_m_block
    current_n_offs = block_n_idx * BLOCK_SIZE_N + offs_n_block

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start_iter in range(0, K, BLOCK_SIZE_K):
        current_k_iter_offs = k_start_iter + offs_k_block # These are offsets for the K_op dimension
        k_mask_op = current_k_iter_offs < K

        # Load A block (M_BLK, K_BLK)
        a_load_mask = (current_m_offs[:, None] < M) & k_mask_op[None, :]
        a_block_ptrs = a_start_ptr + current_m_offs[:, None] * stride_a_m + current_k_iter_offs[None, :] * stride_a_k
        a_block = tl.load(a_block_ptrs, mask=a_load_mask, other=0.0)

        # Load B block (effectively K_BLK, N_BLK for the matmul)
        if TRANSPOSE_B:
            # B_orig is (B,H,N_orig,K_orig) = (B,H,T,hs) for K tensor.
            # We want B_eff(K_op, N_op) = B_eff(hs, T)
            # current_k_iter_offs for K_op=hs, current_n_offs for N_op=T
            # So, current_k_iter_offs map to K_orig dim of B_orig (stride_b_orig_dim2)
            # And current_n_offs map to N_orig dim of B_orig (stride_b_orig_dim1)
            b_load_mask = (current_k_iter_offs[:, None] < K) & (current_n_offs[None, :] < N) # K is K_op, N is N_op
            b_block_ptrs = b_orig_start_ptr + current_n_offs[None, :] * stride_b_orig_dim1 + current_k_iter_offs[:, None] * stride_b_orig_dim2
        else:
            # B_orig is (B,H,K_op,N_op) = (B,H,T,hs) for V tensor, or (B,H,K_op,N_op) for regular B in A@B
            # We want B_eff(K_op, N_op)
            # current_k_iter_offs for K_op, current_n_offs for N_op
            # So, current_k_iter_offs map to K_op dim of B_orig (stride_b_orig_dim1)
            # And current_n_offs map to N_op dim of B_orig (stride_b_orig_dim2)
            b_load_mask = (current_k_iter_offs[:, None] < K) & (current_n_offs[None, :] < N)
            b_block_ptrs = b_orig_start_ptr + current_k_iter_offs[:, None] * stride_b_orig_dim1 + current_n_offs[None, :] * stride_b_orig_dim2
        
        b_eff_block = tl.load(b_block_ptrs, mask=b_load_mask, other=0.0)

        acc += tl.dot(a_block, b_eff_block)

    output_block_ptrs = output_start_ptr + current_m_offs[:, None] * stride_o_m + current_n_offs[None, :] * stride_o_n
    output_mask = (current_m_offs[:, None] < M) & (current_n_offs[None, :] < N)
    tl.store(output_block_ptrs, acc, mask=output_mask)


# --- Triton Kernel 3: Attention Core (Scale, Causal Mask, Softmax) ---
@triton.jit
def attention_core_kernel(
    input_ptr, # Raw attention scores (B, H, T_actual, T_actual)
    causal_mask_ptr, # Precomputed causal mask (T_actual, T_actual), 1 for allow, 0 for mask
    output_ptr, # Softmaxed attention scores (B, H, T_actual, T_actual)
    B, H, T_actual,
    scale_factor,
    stride_in_b, stride_in_h, stride_in_t_row, stride_in_t_col,
    stride_mask_t_row, stride_mask_t_col,
    stride_out_b, stride_out_h, stride_out_t_row, stride_out_t_col,
    BLOCK_SIZE_T_COLS: tl.constexpr
):
    pid_bh = tl.program_id(axis=0)
    pid_t_row_idx = tl.program_id(axis=1)

    batch_idx = pid_bh // H
    head_idx = pid_bh % H

    row_start_ptr_in = input_ptr + batch_idx * stride_in_b + head_idx * stride_in_h + pid_t_row_idx * stride_in_t_row
    row_start_ptr_out = output_ptr + batch_idx * stride_out_b + head_idx * stride_out_h + pid_t_row_idx * stride_out_t_row
    mask_row_start_ptr = causal_mask_ptr + pid_t_row_idx * stride_mask_t_row

    max_val = -float('inf')
    col_arange = tl.arange(0, BLOCK_SIZE_T_COLS)

    for col_block_offset in range(0, T_actual, BLOCK_SIZE_T_COLS):
        current_cols = col_block_offset + col_arange
        col_boundary_mask = current_cols < T_actual
        
        # Load raw scores for the current block of columns in the row
        scores_block = tl.load(row_start_ptr_in + current_cols * stride_in_t_col, mask=col_boundary_mask, other=-float('inf'))
        scaled_scores = scores_block * scale_factor

        # Load causal mask values for this block of columns
        # Mask is 1 if allowed, 0 if masked. We need to fill with -inf if mask is 0.
        causal_values = tl.load(mask_row_start_ptr + current_cols * stride_mask_t_col, mask=col_boundary_mask, other=0.0)
        final_scores = tl.where(causal_values == 0, -float('inf'), scaled_scores)
        
        current_max_in_block = tl.max(tl.where(col_boundary_mask, final_scores, -float('inf')), axis=0)
        max_val = tl.maximum(max_val, current_max_in_block)

    sum_exp_val = 0.0
    for col_block_offset in range(0, T_actual, BLOCK_SIZE_T_COLS):
        current_cols = col_block_offset + col_arange
        col_boundary_mask = current_cols < T_actual

        scores_block = tl.load(row_start_ptr_in + current_cols * stride_in_t_col, mask=col_boundary_mask, other=0.0) # other doesn't matter as much here due to mask
        scaled_scores = scores_block * scale_factor
        
        causal_values = tl.load(mask_row_start_ptr + current_cols * stride_mask_t_col, mask=col_boundary_mask, other=0.0)
        final_scores = tl.where(causal_values == 0, -float('inf'), scaled_scores)

        shifted_scores = final_scores - max_val
        exp_scores = tl.exp(shifted_scores)
        
        # Store temp exp values, masked to ensure 0 for out-of-bounds cols before sum and for final division pass
        exp_to_store = tl.where(col_boundary_mask, exp_scores, 0.0)
        tl.store(row_start_ptr_out + current_cols * stride_out_t_col, exp_to_store, mask=col_boundary_mask) # only store valid cols
        
        sum_exp_val += tl.sum(exp_to_store, axis=0)

    epsilon = 1e-9 # Adjusted epsilon
    for col_block_offset in range(0, T_actual, BLOCK_SIZE_T_COLS):
        current_cols = col_block_offset + col_arange
        col_boundary_mask = current_cols < T_actual
        
        exp_val_block = tl.load(row_start_ptr_out + current_cols * stride_out_t_col, mask=col_boundary_mask, other=0.0)
        norm_block = exp_val_block / (sum_exp_val + epsilon)
        tl.store(row_start_ptr_out + current_cols * stride_out_t_col, norm_block, mask=col_boundary_mask)

# --- Python Wrappers ---
def triton_linear_projection_wrapper(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
    assert x.is_cuda and weight.is_cuda
    if bias is not None: assert bias.is_cuda
    assert x.dtype == weight.dtype and (bias is None or bias.dtype == x.dtype), "Dtype mismatch"

    M, K_x = x.shape
    N_w, K_w = weight.shape # W is (out_features, in_features)
    assert K_x == K_w, f"Weight K dim {K_w} must match X K dim {K_x}"
    K = K_x
    N = N_w

    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Heuristic for block sizes, can be tuned
    BLOCK_SIZE_M = 64 if M > 512 else (32 if M > 128 else 16)
    BLOCK_SIZE_N = 64 if N > 512 else (32 if N > 128 else 16)
    BLOCK_SIZE_K = 32 if K > 128 else (16 if K > 32 else K) # K can be small
    if BLOCK_SIZE_K == 0: BLOCK_SIZE_K = 16 # Avoid 0

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    linear_projection_kernel[grid](
        x, weight, bias, output,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        ADD_BIAS=(bias is not None)
    )
    return output

def triton_bmm_wrapper(a: torch.Tensor, b_orig: torch.Tensor, transpose_b_effective: bool):
    assert a.is_cuda and b_orig.is_cuda
    assert a.dtype == b_orig.dtype, "Dtype mismatch for BMM inputs"

    B_a, H_a, M_a, K_a = a.shape # A is (B, H, M_op, K_op)
    B_b, H_b, D1_b, D2_b = b_orig.shape # b_orig is the original tensor K or V

    assert B_a == B_b and H_a == H_b, "Batch and Head dims must match for A and B_orig"
    B, H = B_a, H_a
    M_op = M_a
    K_op = K_a

    if transpose_b_effective:
        # A(M_op, K_op) @ B_orig.T(K_op, N_op) effectively.
        # B_orig is (B,H,N_op,K_op) e.g., K tensor (B,H,T,hs) -> N_op=T, K_op=hs
        assert K_op == D2_b, f"Inner K_op dim mismatch for A@B.T: A_K_op({K_op}) vs B_orig_D2({D2_b})"
        N_op = D1_b 
    else:
        # A(M_op, K_op) @ B_orig(K_op, N_op) effectively.
        # B_orig is (B,H,K_op,N_op) e.g., V tensor (B,H,T,hs) -> K_op=T, N_op=hs
        assert K_op == D1_b, f"Inner K_op dim mismatch for A@B: A_K_op({K_op}) vs B_orig_D1({D1_b})"
        N_op = D2_b
        
    output = torch.empty((B, H, M_op, N_op), device=a.device, dtype=a.dtype)

    # Heuristic block sizes
    BLOCK_SIZE_M = 32 # M_op (e.g. T for QK.T, T for Att@V)
    BLOCK_SIZE_N = 32 # N_op (e.g. T for QK.T, hs for Att@V)
    BLOCK_SIZE_K = 32 # K_op (e.g. hs for QK.T, T for Att@V)
    if K_op <= 16: BLOCK_SIZE_K = K_op if K_op > 0 else 16
    elif K_op <= 32: BLOCK_SIZE_K = 32
    else: BLOCK_SIZE_K = 64
    if BLOCK_SIZE_K == 0: BLOCK_SIZE_K = 16

    grid = (B * H, triton.cdiv(M_op, BLOCK_SIZE_M) * triton.cdiv(N_op, BLOCK_SIZE_N))
    
    bmm_kernel[grid](
        a, b_orig, output,
        B, H, M_op, N_op, K_op,
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        b_orig.stride(0), b_orig.stride(1), b_orig.stride(2), b_orig.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        TRANSPOSE_B=transpose_b_effective
    )
    return output

# Renamed and modified for in-place operation
def triton_attention_core_inplace_wrapper(att_tensor_to_modify_inplace: torch.Tensor, causal_mask_full_2d: torch.Tensor, scale_factor: float, current_T: int):
    assert att_tensor_to_modify_inplace.is_cuda and causal_mask_full_2d.is_cuda
    assert att_tensor_to_modify_inplace.dtype == torch.float32 # Kernels typically use float32
    B, H, T_att_row, T_att_col = att_tensor_to_modify_inplace.shape
    assert T_att_row == current_T and T_att_col == current_T, "Attention tensor T dim must match current_T"
    assert causal_mask_full_2d.shape[0] >= current_T and causal_mask_full_2d.shape[1] >= current_T

    causal_mask_sliced = causal_mask_full_2d[:current_T, :current_T].contiguous()

    # No longer creating a new output tensor. Will operate in-place.
    # output = torch.empty_like(att_tensor_to_modify_inplace)
    
    grid = (B * H, current_T)

    if current_T == 0: BLOCK_SIZE_T_COLS = 16
    elif current_T <= 16: BLOCK_SIZE_T_COLS = 16
    elif current_T <= 32: BLOCK_SIZE_T_COLS = 32
    elif current_T <= 64: BLOCK_SIZE_T_COLS = 64
    elif current_T <= 128: BLOCK_SIZE_T_COLS = 128
    elif current_T <= 256: BLOCK_SIZE_T_COLS = 256
    elif current_T <= 512: BLOCK_SIZE_T_COLS = 512
    else: BLOCK_SIZE_T_COLS = 1024 
    if BLOCK_SIZE_T_COLS > current_T and current_T > 0 : BLOCK_SIZE_T_COLS = triton.next_power_of_2(current_T)
    
    attention_core_kernel[grid](
        att_tensor_to_modify_inplace, # Pass as input_ptr
        causal_mask_sliced, 
        att_tensor_to_modify_inplace, # Pass SAME TENSOR as output_ptr for in-place
        B, H, current_T,
        scale_factor,
        att_tensor_to_modify_inplace.stride(0), att_tensor_to_modify_inplace.stride(1), att_tensor_to_modify_inplace.stride(2), att_tensor_to_modify_inplace.stride(3),
        causal_mask_sliced.stride(0), causal_mask_sliced.stride(1),
        att_tensor_to_modify_inplace.stride(0), att_tensor_to_modify_inplace.stride(1), att_tensor_to_modify_inplace.stride(2), att_tensor_to_modify_inplace.stride(3),
        BLOCK_SIZE_T_COLS=BLOCK_SIZE_T_COLS
    )
    return att_tensor_to_modify_inplace # Return the modified tensor

# --- ModelNew Class ---
class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head

        # Parameters for c_attn (QKV projection) - nn.Linear(n_embd, 3 * n_embd)
        # Weight shape: (3 * n_embd, n_embd)
        self.c_attn_weight = nn.Parameter(torch.empty(3 * n_embd, n_embd, dtype=torch.float32))
        self.c_attn_bias = nn.Parameter(torch.empty(3 * n_embd, dtype=torch.float32))
        
        # Parameters for c_proj (output projection) - nn.Linear(n_embd, n_embd)
        # Weight shape: (n_embd, n_embd)
        self.c_proj_weight = nn.Parameter(torch.empty(n_embd, n_embd, dtype=torch.float32))
        self.c_proj_bias = nn.Parameter(torch.empty(n_embd, dtype=torch.float32))

        self.register_buffer("causal_mask", torch.tril(torch.ones(max_seqlen, max_seqlen, dtype=torch.float32)))
        
        self._initialize_weights() # KernelBench should override this with loaded state_dict

    def _initialize_weights(self):
        # Standard init for nn.Linear like layers
        nn.init.kaiming_uniform_(self.c_attn_weight, a=math.sqrt(5))
        fan_in_attn, _ = nn.init._calculate_fan_in_and_fan_out(self.c_attn_weight)
        if fan_in_attn > 0:
            bound_attn = 1 / math.sqrt(fan_in_attn)
            nn.init.uniform_(self.c_attn_bias, -bound_attn, bound_attn)
        else:
            nn.init.zeros_(self.c_attn_bias)

        nn.init.kaiming_uniform_(self.c_proj_weight, a=math.sqrt(5))
        fan_in_proj, _ = nn.init._calculate_fan_in_and_fan_out(self.c_proj_weight)
        if fan_in_proj > 0:
            bound_proj = 1 / math.sqrt(fan_in_proj)
            nn.init.uniform_(self.c_proj_bias, -bound_proj, bound_proj)
        else:
            nn.init.zeros_(self.c_proj_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        assert C == self.n_embd, f"Input embedding dim {C} doesn't match model n_embd {self.n_embd}"
        x = x.to(torch.float32) # Ensure float32 for Triton kernels

        # 1. QKV Projection (c_attn)
        x_reshaped = x.reshape(B * T, C)
        qkv_combined_flat = triton_linear_projection_wrapper(x_reshaped, self.c_attn_weight, self.c_attn_bias)
        qkv_combined = qkv_combined_flat.reshape(B, T, 3 * self.n_embd)

        q, k, v = qkv_combined.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)

        # 2. Compute Q @ K.T
        att_raw = triton_bmm_wrapper(q, k, transpose_b_effective=True)
        del q, k # Free up q, k after use, qkv_combined still holds data
        # Potentially add torch.cuda.empty_cache() here if memory is extremely tight, but GC should handle q,k

        # 3. Scale, Causal Mask, Softmax (IN-PLACE)
        scale = 1.0 / math.sqrt(self.head_size)
        # att_raw will be modified in-place to become att_softmaxed
        att_softmaxed = triton_attention_core_inplace_wrapper(att_raw, self.causal_mask, scale, T)
        # Now att_softmaxed (which is the modified att_raw tensor) contains the softmaxed attention scores.

        # 4. Compute Attention_scores @ V
        y_heads = triton_bmm_wrapper(att_softmaxed, v, transpose_b_effective=False)
        del att_softmaxed, v # Free up att_softmaxed (modified att_raw) and v
        # torch.cuda.empty_cache()

        # 5. Reshape output heads
        y_reshaped = y_heads.transpose(1, 2).contiguous().view(B, T, C)
        del y_heads

        # 6. Final Projection (c_proj)
        y_for_proj = y_reshaped.reshape(B * T, C)
        del y_reshaped
        final_output_flat = triton_linear_projection_wrapper(y_for_proj, self.c_proj_weight, self.c_proj_bias)
        final_output = final_output_flat.reshape(B, T, C)

        return final_output 