import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

# --- Triton Kernel: Linear Projection + Optional ReLU ---
@triton.jit
def linear_projection_relu_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K, # Input X (M, K), Weight W (N, K), Output O (M, N)
    stride_x_m, stride_x_k,
    stride_w_n, stride_w_k, # weight_ptr is to W (N,K) (out_features, in_features)
    stride_o_m, stride_o_n,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    APPLY_RELU: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = (pid // num_pid_n) % num_pid_m
    pid_n = pid % num_pid_n

    offs_m_block = tl.arange(0, BLOCK_SIZE_M)
    offs_n_block = tl.arange(0, BLOCK_SIZE_N)
    offs_k_block = tl.arange(0, BLOCK_SIZE_K)

    offs_x_m = pid_m * BLOCK_SIZE_M + offs_m_block
    offs_w_n_bias = pid_n * BLOCK_SIZE_N + offs_n_block

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        current_k_offs = k_start + offs_k_block
        k_mask_iter = current_k_offs < K

        x_load_mask = (offs_x_m[:, None] < M) & k_mask_iter[None, :]
        x_block_ptrs = x_ptr + offs_x_m[:, None] * stride_x_m + current_k_offs[None, :] * stride_x_k
        x_block = tl.load(x_block_ptrs, mask=x_load_mask, other=0.0)

        w_load_mask = k_mask_iter[:, None] & (offs_w_n_bias[None, :] < N)
        w_block_ptrs = weight_ptr + current_k_offs[:, None] * stride_w_k + offs_w_n_bias[None, :] * stride_w_n
        w_block_transposed_tile = tl.load(w_block_ptrs, mask=w_load_mask, other=0.0)
        
        accumulator += tl.dot(x_block, w_block_transposed_tile)

    if ADD_BIAS:
        bias_load_mask = offs_w_n_bias < N
        bias_vals_ptr = bias_ptr + offs_w_n_bias
        bias_block = tl.load(bias_vals_ptr, mask=bias_load_mask, other=0.0)
        accumulator += bias_block[None, :]
    
    if APPLY_RELU:
        accumulator = tl.maximum(accumulator, 0.0)

    output_offs_m = pid_m * BLOCK_SIZE_M + offs_m_block
    output_offs_n = pid_n * BLOCK_SIZE_N + offs_n_block
    output_ptrs = output_ptr + output_offs_m[:, None] * stride_o_m + output_offs_n[None, :] * stride_o_n
    output_mask = (output_offs_m[:, None] < M) & (output_offs_n[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=output_mask)

# --- Python Wrapper for Linear + Optional ReLU ---
def triton_linear_relu_wrapper(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, apply_relu: bool = False):
    assert x.is_cuda and weight.is_cuda
    if bias is not None: assert bias.is_cuda
    assert x.dtype == weight.dtype and (bias is None or bias.dtype == x.dtype), "Dtype mismatch"

    M, K_x = x.shape
    N_w, K_w = weight.shape # W is (out_features, in_features)
    assert K_x == K_w, f"Weight K dim {K_w} must match X K dim {K_x}"
    K = K_x
    N = N_w

    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE_M = 64 if M > 512 else (32 if M > 128 else 16)
    BLOCK_SIZE_N = 64 if N > 512 else (32 if N > 128 else 16)
    BLOCK_SIZE_K = 32 if K > 256 else (16 if K > 64 else K) 
    if BLOCK_SIZE_K == 0: BLOCK_SIZE_K = 16 # Handle K being too small
    if K < BLOCK_SIZE_K : BLOCK_SIZE_K = triton.next_power_of_2(K) if K>0 else 16

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    linear_projection_relu_kernel[grid](
        x, weight, bias, output,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        ADD_BIAS=(bias is not None),
        APPLY_RELU=apply_relu
    )
    return output

# --- ModelNew Class (Hybrid AlexNet) ---
class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        # Convolutional layers (using PyTorch's nn.Conv2d and nn.MaxPool2d)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        # self.relu1 = nn.ReLU(inplace=True) # Will be applied after conv in forward if not fused, or part of conv kernel if fused
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Parameters for Triton-based fully connected layers
        # fc1: in_features=256 * 6 * 6 = 9216, out_features=4096
        fc1_in_features = 256 * 6 * 6
        fc1_out_features = 4096
        self.fc1_weight = nn.Parameter(torch.empty(fc1_out_features, fc1_in_features, dtype=torch.float32))
        self.fc1_bias = nn.Parameter(torch.empty(fc1_out_features, dtype=torch.float32))

        # fc2: in_features=4096, out_features=4096
        fc2_in_features = 4096
        fc2_out_features = 4096
        self.fc2_weight = nn.Parameter(torch.empty(fc2_out_features, fc2_in_features, dtype=torch.float32))
        self.fc2_bias = nn.Parameter(torch.empty(fc2_out_features, dtype=torch.float32))

        # fc3: in_features=4096, out_features=num_classes
        fc3_in_features = 4096
        self.fc3_weight = nn.Parameter(torch.empty(num_classes, fc3_in_features, dtype=torch.float32))
        self.fc3_bias = nn.Parameter(torch.empty(num_classes, dtype=torch.float32))

        self._initialize_fc_weights() # KernelBench should override via state_dict from ref model

    def _initialize_fc_weights(self):
        # Mimic nn.Linear initialization for consistency if state_dict isn't loaded
        for name, param in self.named_parameters():
            if 'conv' in name: continue # PyTorch layers handle their own init
            if 'weight' in name:
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            elif 'bias' in name:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.get_parameter(name.replace('_bias', '_weight')))
                if fan_in > 0:
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(param, -bound, bound)
                else:
                    nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32) # Ensure float32

        # Convolutional part (using PyTorch layers)
        x = F.relu(self.conv1(x)) # Apply ReLU after each conv
        x = self.maxpool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool3(x)
        
        x = torch.flatten(x, 1)
        
        # Fully connected part (using Triton kernel)
        # fc1 + relu
        x = triton_linear_relu_wrapper(x, self.fc1_weight, self.fc1_bias, apply_relu=True)
        # dropout1 is identity (p=0.0)
        
        # fc2 + relu
        x = triton_linear_relu_wrapper(x, self.fc2_weight, self.fc2_bias, apply_relu=True)
        # dropout2 is identity (p=0.0)
        
        # fc3 (no relu)
        x = triton_linear_relu_wrapper(x, self.fc3_weight, self.fc3_bias, apply_relu=False)
        
        return x 