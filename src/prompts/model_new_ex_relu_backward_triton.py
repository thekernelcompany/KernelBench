import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def relu_forward_kernel(
    input_ptr,      # Pointer to input tensor
    output_ptr,     # Pointer to output tensor  
    n_elements,     # Number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """Forward pass: ReLU(x) = max(0, x)"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    
    # Apply ReLU: max(0, x)
    output_vals = tl.maximum(input_vals, 0.0)
    
    # Store output values
    tl.store(output_ptr + offsets, output_vals, mask=mask)


@triton.jit
def relu_backward_kernel(
    grad_output_ptr,  # Pointer to gradient of output
    input_ptr,        # Pointer to original input (saved from forward)
    grad_input_ptr,   # Pointer to gradient of input (output)
    n_elements,       # Number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """Backward pass: d/dx ReLU(x) = 1 if x > 0 else 0"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load gradient of output and original input
    grad_output_vals = tl.load(grad_output_ptr + offsets, mask=mask)
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute gradient: 1 if input > 0, else 0
    grad_input_vals = tl.where(input_vals > 0.0, grad_output_vals, 0.0)
    
    # Store gradient of input
    tl.store(grad_input_ptr + offsets, grad_input_vals, mask=mask)


def triton_relu_forward(input: torch.Tensor) -> torch.Tensor:
    """Triton ReLU forward pass"""
    output = torch.empty_like(input)
    n_elements = input.numel()
    
    # Launch kernel with auto-tuned block size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    relu_forward_kernel[grid](input, output, n_elements, BLOCK_SIZE=1024)
    
    return output


def triton_relu_backward(grad_output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    """Triton ReLU backward pass"""
    grad_input = torch.empty_like(input)
    n_elements = input.numel()
    
    # Launch kernel with auto-tuned block size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    relu_backward_kernel[grid](grad_output, input, grad_input, n_elements, BLOCK_SIZE=1024)
    
    return grad_input


class TritonReLUFunction(torch.autograd.Function):
    """
    Custom autograd function for Triton ReLU with forward and backward passes
    """
    
    @staticmethod
    def forward(ctx, input):
        # Save input for backward pass
        ctx.save_for_backward(input)
        
        # Compute forward pass using Triton kernel
        output = triton_relu_forward(input)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved input from forward pass
        input, = ctx.saved_tensors
        
        # Compute backward pass using Triton kernel
        grad_input = triton_relu_backward(grad_output, input)
        return grad_input


class ModelNew(nn.Module):
    """
    ReLU model with Triton kernels for both forward and backward passes
    """
    
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ReLU activation using custom Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ReLU applied, same shape as input.
        """
        return TritonReLUFunction.apply(x) 