import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # We are processing a 1D array of elements
    
    # Calculate offsets for the current block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle elements at the end of the tensor, if n_elements is not a multiple of BLOCK_SIZE
    mask = offsets < n_elements
    
    # Load data from input_ptr
    # Apply mask to avoid out-of-bounds reads, 'other=0.0' specifies a default value for masked-out elements (won't affect ReLU)
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # ReLU computation: output = max(0, x)
    output_val = tl.maximum(x, 0.0) # tl.maximum is element-wise max
    
    # Store the result to output_ptr
    # Apply mask to avoid out-of-bounds writes
    tl.store(output_ptr + offsets, output_val, mask=mask)

def triton_relu_function(input_tensor: torch.Tensor) -> torch.Tensor:
    # Ensure input is on CUDA
    assert input_tensor.is_cuda, "Input tensor must be on a CUDA device"
    
    # Allocate output tensor with the same shape and device as input
    output_tensor = torch.empty_like(input_tensor)
    
    # Get total number of elements in the tensor
    n_elements = input_tensor.numel()
    
    # Define the grid for kernel launch.
    # The grid is 1D, and its size is the number of blocks needed to cover all elements.
    # triton.cdiv(a, b) computes ceil(a / b)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch the kernel
    # Default BLOCK_SIZE, can be tuned. 1024 is a common starting point for element-wise ops.
    relu_kernel[grid](
        input_tensor,
        output_tensor,
        n_elements,
        BLOCK_SIZE=1024
    )
    
    return output_tensor

class ModelNew(nn.Module):
    """
    Required class name for KernelBench.
    This model wraps the Triton ReLU kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward method must match the signature of the reference PyTorch model's forward method.
        For 19_ReLU.py, it's forward(self, x: torch.Tensor).
        """
        return triton_relu_function(x) 