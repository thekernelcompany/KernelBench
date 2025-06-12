# Triton Support for KernelBench

This document describes the Triton kernel evaluation support that has been added to KernelBench, allowing you to evaluate Triton kernels against PyTorch reference implementations using the same rigorous correctness and performance testing framework.

## ✨ Features

- **Auto-detection**: Automatically detects whether a kernel is CUDA or Triton based on imports and syntax
- **Unified evaluation**: Same correctness and performance metrics for both CUDA and Triton kernels
- **Triton-optimized caching**: Proper handling of Triton's JIT compilation and cache management
- **Drop-in replacement**: Works with existing KernelBench problems and evaluation pipelines

## 🚀 Quick Start

### 1. Install Triton

```bash
pip install triton
```

### 2. Run Triton Kernel Evaluation

```bash
# Auto-detect kernel type and evaluate
python3 scripts/run_and_check_triton.py \
    ref_origin=local \
    ref_arch_src_path=src/prompts/model_ex_add.py \
    kernel_src_path=src/prompts/model_new_ex_add_triton.py

# Force Triton evaluation
python3 scripts/run_and_check_triton.py \
    ref_origin=kernelbench \
    level=1 \
    problem_id=1 \
    kernel_src_path=path/to/your/triton_kernel.py \
    force_triton=True
```

### 3. Evaluate Against KernelBench Problems

```bash
# Evaluate a Triton matrix multiplication against level 1, problem 1
python3 scripts/run_and_check_triton.py \
    ref_origin=kernelbench \
    level=1 \
    problem_id=1 \
    kernel_src_path=src/prompts/model_new_ex_matmul_triton.py
```

## 📝 Triton Kernel Format

Triton kernels should follow the same structure as CUDA kernels in KernelBench:

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def your_forward_kernel(...):
    # Triton forward kernel implementation
    pass

# If implementing a backward pass:
@triton.jit
def your_backward_kernel(...):
    # Triton backward kernel implementation
    pass

def your_triton_forward_function(inputs):
    # Function that calls the Triton forward kernel
    # Allocate output tensor, define grid, launch kernel
    return result

# If implementing a backward pass:
def your_triton_backward_function(grad_output, saved_tensors):
    # Function that calls the Triton backward kernel
    # Retrieve saved tensors, allocate grad_input, define grid, launch kernel
    return grad_input

# To enable backward pass, use torch.autograd.Function
class YourCustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, other_args...):
        # Save tensors for backward pass if needed
        # ctx.save_for_backward(input_tensor, ...)
        # Call your Triton forward function
        output = your_triton_forward_function(input_tensor, other_args...)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        # input_tensor, ... = ctx.saved_tensors
        # Call your Triton backward function
        grad_input = your_triton_backward_function(grad_output, saved_tensors)
        # Ensure you return gradients for all inputs to forward, e.g., None for non-tensor inputs
        return grad_input, None # Assuming other_args... did not require gradients

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *inputs):
        # If using torch.autograd.Function for backward pass:
        # return YourCustomFunction.apply(*inputs)
        # Otherwise, for forward-only:
        return your_triton_forward_function(*inputs)
```

**Key considerations for backward pass:**
- Implement both a forward Triton kernel (`your_forward_kernel`) and a backward Triton kernel (`your_backward_kernel`).
- Create corresponding Python wrapper functions (`your_triton_forward_function`, `your_triton_backward_function`).
- Subclass `torch.autograd.Function` to integrate your custom forward and backward logic with PyTorch's autograd engine.
- In the `forward` method of your custom `Function`, use `ctx.save_for_backward(...)` to store any tensors needed for the gradient computation.
- In the `backward` method, retrieve these tensors using `ctx.saved_tensors`.
- The `backward` method must return a gradient for each input of the `forward` method. If an input did not require a gradient or was not a tensor, return `None` for that position.

## 📁 Example Kernels

### Element-wise Addition
- **Reference**: `src/prompts/model_ex_add.py`
- **Triton**: `src/prompts/model_new_ex_add_triton.py`

### Matrix Multiplication
- **Reference**: Available in KernelBench Level 1
- **Triton**: `src/prompts/model_new_ex_matmul_triton.py`
- **Triton with Backward Pass (Example - ReLU)**: `src/prompts/model_new_ex_relu_backward_triton.py` (Illustrates `torch.autograd.Function` structure)

## 🔧 API Reference

### New Functions in `src/eval.py`

#### `detect_triton_kernel(model_src: str) -> bool`
Automatically detects if source code contains Triton kernels.

#### `eval_triton_kernel_against_ref(...)`
Evaluates Triton kernels with specialized handling for JIT compilation.

#### `eval_kernel_against_ref_auto(...)`
Automatically chooses the appropriate evaluation function based on kernel type.

#### `build_compile_cache_triton(...)`
Pre-warms Triton kernels and manages Triton-specific caching.

#### `eval_triton_backward_pass(...)`
Evaluates Triton kernels with backward pass, including gradient correctness and performance. (Used by `eval_kernel_backward_pass_auto`)

#### `eval_kernel_backward_pass_auto(...)`
Automatically detects kernel type and evaluates the backward pass. Currently supports Triton.

#### `test_gradient_correctness_triton(...)`
Tests gradient correctness for Triton kernels using `torch.autograd.gradcheck` with memory optimizations.

### Scripts

#### `scripts/run_and_check_triton.py`
Enhanced version of `run_and_check.py` with Triton support and auto-detection.

**Key Parameters:**
- `auto_detect=True`: Automatically detect kernel type
- `force_triton=True`: Force Triton evaluation
- `verbose=True`: Enable detailed logging
- `test_backward_pass=True`: Enable full backward pass evaluation (gradient correctness and performance).
- `num_gradient_trials=3`: Number of trials for `torch.autograd.gradcheck`.
- `gradient_tolerance=1e-4`: Tolerance for gradient checking.
- `measure_backward_performance=True`: Measure performance of the backward pass for custom and reference kernels.

## 🏎️ Performance Considerations

### Triton vs CUDA
- **Triton**: JIT compiled, potentially slower first run but optimized for productivity
- **CUDA**: Pre-compiled, consistent performance but requires more low-level programming

### Compilation Differences
- **CUDA**: Compilation happens at load time via `torch.utils.cpp_extension`
- **Triton**: JIT compilation happens on first kernel execution
- **Caching**: Both use different caching mechanisms that are properly handled

### Backward Pass Performance
- When `test_backward_pass=True` and `measure_backward_performance=True`, the script will benchmark:
    - Your custom Triton kernel's backward pass.
    - PyTorch eager mode's backward pass for the reference model.
    - `torch.compile`'s backward pass for the reference model.
- Speedups are reported for your custom kernel's backward pass against these two baselines.

## 🔍 Evaluation Metrics

The same metrics apply to both CUDA and Triton kernels:

- **Correctness**:
    - **Forward Pass**: Multiple trials with randomized inputs, tolerance-based comparison.
    - **Backward Pass (if `test_backward_pass=True`)**:
        - Forward pass correctness is checked first.
        - Gradient correctness is verified using `torch.autograd.gradcheck` over `num_gradient_trials`.
- **Performance**:
    - **Forward Pass**: Wall-clock timing using CUDA events.
    - **Backward Pass (if measured)**: Wall-clock timing of the backward computation.
- **Speedup**:
    - **Forward Pass**: Comparison against PyTorch eager and `torch.compile` baselines.
    - **Backward Pass (if measured)**: Comparison against PyTorch eager backward and `torch.compile` backward baselines.

Example output:
```
========================================
[FORWARD PASS RESULTS]
========================================
[Eval] triton kernel eval result: compiled=True, correctness=True, runtime=0.245ms, metadata={...}
----------------------------------------
[Timing] PyTorch Reference Eager exec time: 0.892 ms
[Timing] PyTorch Reference torch.compile time: 0.156 ms
[Timing] Custom triton Kernel exec time: 0.245 ms
----------------------------------------
[Speedup] Forward Speedup over eager: 3.64x
[Speedup] Forward Speedup over torch.compile: 0.64x
========================================

========================================
[BACKWARD PASS RESULTS]
========================================
[Eval] triton backward pass result: compiled=True, correctness=True, runtime=0.294ms, metadata={..., 'gradient_correctness': '(3 / 3)', 'backward_pass_correctness': True, ...}
----------------------------------------
[Correctness] Forward trials: (5 / 5)
[Correctness] Gradient trials: (3 / 3)
[Correctness] Overall backward pass: ✅ PASS
----------------------------------------
[Timing] PyTorch Reference Backward Eager time: 0.271 ms
[Timing] PyTorch Reference Backward torch.compile time: 0.450 ms
[Timing] Custom triton Backward Kernel time: 0.294 ms
----------------------------------------
[Speedup] Backward Speedup over eager: 0.92x
[Speedup] Backward Speedup over torch.compile: 1.53x
========================================
```

## 🏗️ Integration with Existing Tools

### With existing scripts:
The Triton support integrates seamlessly with existing KernelBench evaluation tools:

```bash
# Generate samples (will work with Triton if detected)
python3 scripts/generate_samples.py run_name=triton_test level=1

# Evaluate samples (auto-detects kernel type, can test backward pass if specified in generation or if kernels are structured for it)
python3 scripts/eval_from_generations.py run_name=triton_test level=1

# Analyze results (same metrics for CUDA and Triton, includes backward pass if present in logs)
python3 scripts/benchmark_eval_analysis.py run_name=triton_test level=1
```

## 🧪 Testing

Test the Triton evaluation with the provided examples:

```bash
# Test element-wise addition (Forward only example)
python3 scripts/run_and_check_triton.py \
    ref_origin=local \
    ref_arch_src_path=src/prompts/model_ex_add.py \
    kernel_src_path=src/prompts/model_new_ex_add_triton.py \
    verbose=True

# Test ReLU with Backward Pass (Triton)
python3 scripts/run_and_check_triton.py \
    ref_origin=kernelbench \
    level=1 \
    problem_id=19 \
    kernel_src_path=src/prompts/model_new_ex_relu_backward_triton.py \
    test_backward_pass=True \
    verbose=True

# Test matrix multiplication against KernelBench (Forward only example)
python3 scripts/run_and_check_triton.py \
    ref_origin=kernelbench \
    level=1 \
    problem_id=1 \
    kernel_src_path=src/prompts/model_new_ex_matmul_triton.py \
    verbose=True
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Error**: Make sure Triton is installed (`pip install triton`)
2. **CUDA Device**: Triton requires CUDA-capable GPU
3. **Cache Issues**: Clear Triton cache if needed (handled automatically)
4. **JIT Compilation**: First run may be slower due to JIT compilation

### Debug Mode

Enable verbose logging for detailed information:
```bash
python3 scripts/run_and_check_triton.py \
    ... \
    verbose=True
```

## 🛣️ Future Enhancements

- [ ] Triton-specific optimization hints and auto-tuning
- [ ] Full CUDA backward pass evaluation support
- [ ] Integration with Triton's built-in benchmarking tools
- [ ] Support for Triton language extensions
- [ ] Triton kernel template generation for forward and backward passes
- [ ] Performance comparison dashboards (including backward pass metrics)

## 📚 References

- [Triton Documentation](https://triton-lang.org/)
- [KernelBench Paper](https://arxiv.org/abs/2502.10517)
- [Original KernelBench Repository](https://github.com/ScalingIntelligence/KernelBench)

---

For questions or issues related to Triton support, please refer to the main KernelBench documentation or open an issue on the repository. 