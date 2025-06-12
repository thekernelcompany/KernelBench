# 🚀 Modal Triton Automation - Usage Guide

This document explains how to use the Modal-based automation for running Triton kernel evaluations on cloud GPUs. The Modal scripts now produce **exactly the same terminal output** as running locally, including **forward and backward pass testing** with identical formatting.

## 📋 Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Modal CLI**: Install and authenticate
   ```bash
   pip install modal
   modal token new
   ```
> Note: Python 3.10 might have issues with SSL
## 🎯 Quick Start

### 1. Against KernelBench Problem (Recommended)

```bash
# Evaluate your Triton kernel against KernelBench Level 1, Problem 3 on H100
python run_triton_modal.py --kernel my_triton_kernel.py --level 1 --problem_id 3

# Quick test (faster, fewer trials)
python run_triton_modal.py --kernel my_triton_kernel.py --level 1 --problem_id 3 --quick

# On different GPU
python run_triton_modal.py --kernel my_triton_kernel.py --level 1 --problem_id 3 --gpu L40S
```

### 2. Against Local Reference

```bash
# Evaluate against your own reference implementation
python run_triton_modal.py --kernel my_triton_kernel.py --reference my_reference.py
```

### 3. Backward Pass Evaluation

```bash
# Evaluate forward and backward pass
python run_triton_modal.py --kernel my_triton_kernel.py --level 1 --problem_id 3 --test_backward_pass

# Backward pass with custom gradient settings
python run_triton_modal.py --kernel my_triton_kernel.py --level 1 --problem_id 3 \
    --test_backward_pass \
    --num_gradient_trials 5 \
    --gradient_tolerance 1e-5
```

### 4. Advanced Options

```bash
# Full production evaluation with custom settings
python run_triton_modal.py \
    --kernel my_triton_kernel.py \
    --level 2 \
    --problem_id 5 \
    --num_correct_trials 10 \
    --num_perf_trials 200 \
    --gpu H100 \
    --verbose \
    --test_backward_pass
```

## 🔧 Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--kernel` | Path to your Triton kernel file | Required |
| `--level` | KernelBench level (1-4) | - |
| `--problem_id` | Problem ID within the level | - |
| `--reference` | Path to local reference file | - |
| `--gpu` | GPU type (H100, L40S, A100, L4, T4, A10G) | H100 |
| `--quick` | Quick test mode (1 correctness, 10 perf trials) | False |
| `--num_correct_trials` | Number of correctness trials | 5 |
| `--num_perf_trials` | Number of performance trials | 100 |
| `--verbose` | Enable verbose output | False |
| `--test_backward_pass` | Enable backward pass testing | False |
| `--num_gradient_trials` | Number of gradient correctness trials | 3 |
| `--gradient_tolerance` | Tolerance for gradient checking | 1e-4 |
| `--measure_backward_performance` | Enable backward pass performance measurement | True |

## 📁 File Structure

Your Triton kernel should follow this structure:

**Forward Pass Only:**
```python
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def my_kernel(...):
    # Triton kernel implementation
    pass

def my_function(inputs):
    # Function that calls the Triton kernel
    return result

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *inputs):
        return my_function(*inputs)
```

**Forward + Backward Pass:**
```python
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def forward_kernel(...):
    # Forward Triton kernel implementation
    pass

@triton.jit
def backward_kernel(...):
    # Backward Triton kernel implementation
    pass

class CustomTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return triton_forward_function(input)
    
    @staticmethod 
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return triton_backward_function(grad_output, input)

class ModelNew(nn.Module):
    def forward(self, *inputs):
        return CustomTritonFunction.apply(*inputs)
```

## 🏁 Example Workflow

### Step 1: Write Your Triton Kernel

```python
# my_matmul_kernel.py
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    # Your Triton matrix multiplication implementation
    pass

def triton_matmul(a, b):
    # Wrapper function
    pass

class ModelNew(nn.Module):
    def forward(self, a, b):
        return triton_matmul(a, b)
```

### Step 2: Test Against KernelBench

```bash
# Quick test first (forward pass only)
python run_triton_modal.py --kernel my_matmul_kernel.py --level 1 --problem_id 1 --quick

# If it passes, run full evaluation
python run_triton_modal.py --kernel my_matmul_kernel.py --level 1 --problem_id 1 --verbose

# Test backward pass (if your kernel supports it)
python run_triton_modal.py --kernel my_matmul_kernel.py --level 1 --problem_id 1 --test_backward_pass --verbose
```

### Step 3: Check Results

The Modal scripts now produce **exactly the same terminal output** as running locally! You'll see:

**Forward Pass Only:**
```
🚀 Starting Triton evaluation on H100
➡️ Mode: Forward Pass Only
📄 Kernel: my_matmul_kernel.py
📚 Reference: KernelBench Level 1, Problem 1

Running with config ScriptConfig({'ref_origin': 'kernelbench', 'level': 1, 'problem_id': 1, ...})
Filter: 100%|████████████████████████| 100/100 [00:00<00:00, 31455.71 examples/s]
Fetched problem 1 from KernelBench level 1: 1_Square_matrix_multiplication_
[INFO] Auto-detected kernel type: Triton
[INFO] Evaluating kernel against reference code
[Eval] Detected Triton kernel, using Triton evaluation
[Eval] Start Triton Evaluation! on device: cuda:0
...
============================================================
[FORWARD PASS RESULTS]
============================================================
[Eval] triton kernel eval result: compiled=True correctness=True runtime=0.245ms
------------------------------------------------------------
[Timing] PyTorch Reference Eager exec time: 0.189 ms
[Timing] PyTorch Reference torch.compile time: 0.213 ms
[Timing] Custom triton Kernel exec time: 0.245 ms
------------------------------------------------------------
[Speedup] Forward Speedup over eager: 1.23x
[Speedup] Forward Speedup over torch.compile: 1.15x
============================================================
```

**Forward + Backward Pass:**
```
🚀 Starting Triton evaluation on H100
🔄 Mode: Forward + Backward Pass Evaluation

Running with config ScriptConfig({'ref_origin': 'kernelbench', 'test_backward_pass': True, ...})
Filter: 100%|████████████████████████| 100/100 [00:00<00:00, 31455.71 examples/s]
Fetched problem 19 from KernelBench level 1: 19_ReLU
[INFO] Auto-detected kernel type: Triton
[INFO] Evaluating kernel against reference code
...
============================================================
[FORWARD PASS RESULTS]
============================================================
[Eval] triton kernel eval result: compiled=True correctness=True runtime=0.0687ms
------------------------------------------------------------
[Timing] PyTorch Reference Eager exec time: 0.0273 ms
[Timing] PyTorch Reference torch.compile time: 0.065 ms
[Timing] Custom triton Kernel exec time: 0.0687 ms
------------------------------------------------------------
[Speedup] Forward Speedup over eager: 0.40x
[Speedup] Forward Speedup over torch.compile: 0.95x

============================================================
[BACKWARD PASS RESULTS]
============================================================
[Eval] triton backward pass result: compiled=True correctness=True runtime=0.276ms
[Correctness] Gradient trials: (5 / 5)
[Correctness] Overall backward pass: ✅ PASS
------------------------------------------------------------
[Timing] PyTorch Reference Backward Eager time: 0.233 ms
[Timing] PyTorch Reference Backward torch.compile time: 0.515 ms
[Timing] Custom triton Backward Kernel time: 0.276 ms
------------------------------------------------------------
[Speedup] Backward Speedup over eager: 0.85x
[Speedup] Backward Speedup over torch.compile: 1.87x
============================================================
```

## 🎛️ GPU Selection

Choose the appropriate GPU for your workload:

- **H100**: Latest, fastest, most expensive
- **L40S**: Great performance/cost ratio
- **A100**: Reliable workhorse
- **L4**: Budget-friendly for smaller workloads

## 🔍 Debugging

### Common Issues

1. **Module not found errors**: Make sure you're running from the KernelBench root directory
2. **GPU memory errors**: Try a smaller problem or reduce batch sizes
3. **Compilation errors**: Check your Triton kernel syntax

### Enable Debugging

```bash
# Run with verbose output for detailed logs
python run_triton_modal.py --kernel my_kernel.py --level 1 --problem_id 1 --verbose
```

## 📊 Understanding Results

The Modal scripts now output **exactly the same format** as running locally. Here's what to look for:

### Configuration and Setup
```
Running with config ScriptConfig({'ref_origin': 'kernelbench', ...})
Filter: 100%|████████████████████████| 100/100 [00:00<00:00, 31455.71 examples/s]
Fetched problem 19 from KernelBench level 1: 19_ReLU
[INFO] Auto-detected kernel type: Triton
```

### Forward Pass Results Section
```
============================================================
[FORWARD PASS RESULTS]
============================================================
[Eval] triton kernel eval result: compiled=True correctness=True runtime=0.0687ms
------------------------------------------------------------
[Timing] PyTorch Reference Eager exec time: 0.0273 ms
[Timing] PyTorch Reference torch.compile time: 0.065 ms
[Timing] Custom triton Kernel exec time: 0.0687 ms
------------------------------------------------------------
[Speedup] Forward Speedup over eager: 0.40x
[Speedup] Forward Speedup over torch.compile: 0.95x
```

### Backward Pass Results Section (if enabled)
```
============================================================
[BACKWARD PASS RESULTS]  
============================================================
[Eval] triton backward pass result: compiled=True correctness=True runtime=0.276ms
[Correctness] Gradient trials: (5 / 5)
[Correctness] Overall backward pass: ✅ PASS
------------------------------------------------------------
[Timing] PyTorch Reference Backward Eager time: 0.233 ms
[Timing] PyTorch Reference Backward torch.compile time: 0.515 ms
[Timing] Custom triton Backward Kernel time: 0.276 ms
------------------------------------------------------------
[Speedup] Backward Speedup over eager: 0.85x
[Speedup] Backward Speedup over torch.compile: 1.87x
============================================================
```

### Performance Comparison
The system automatically compares against:
- PyTorch eager execution (forward and backward)
- `torch.compile` baseline (forward and backward)
- Reference implementation

### Key Metrics to Watch
- **Compilation**: Should show `compiled=True`
- **Correctness**: Should show `correctness=True` 
- **Gradient Trials**: Should show `(X / X)` where all trials pass
- **Speedups**: Look for values > 1.0x for performance improvements
- **Memory Usage**: GPU memory info is displayed for backward pass testing

### Error Categorization
Errors are automatically categorized:
- `triton_jit_compilation_error`: JIT compilation issues
- `cuda_illegal_memory_access`: Memory access violations
- `tensor_dimension_error`: Shape mismatches

## 🚀 Advanced Usage

### Direct Modal Script Usage

```bash
# Use the Modal script directly with full control
modal run --mount .:/workspace modal_run_and_check_triton.py \
    --kernel_src_path=/workspace/my_kernel.py \
    --ref_origin=kernelbench \
    --level=1 \
    --problem_id=3 \
    --gpu=H100 \
    --verbose
```

### Batch Evaluation

For evaluating multiple kernels, use the comprehensive automation:

```bash
# Use the full automation script
modal run --mount .:/workspace modal_triton_automation.py \
    mode=batch_eval \
    kernel_paths=['kernel1.py','kernel2.py'] \
    gpu=H100
```

## 💡 Tips for Success

### General Tips
1. **Start Small**: Use `--quick` for initial testing
2. **Check Examples**: Look at `TRITON_INTEGRATION_GUIDE.md` for working examples
3. **GPU Choice**: H100 for performance, L40S for cost-effectiveness
4. **Verbose Mode**: Use `--verbose` for detailed debugging information
5. **Incremental Testing**: Test simple operations before complex kernels
6. **Terminal Output**: Modal now shows identical output to local execution - all the same progress bars, timing details, and result formatting

### Backward Pass Tips
6. **Test Forward First**: Always verify forward pass works before adding backward pass
7. **Use torch.autograd.Function**: Required for proper gradient integration
8. **Save Required Tensors**: Use `ctx.save_for_backward()` for tensors needed in backward pass
9. **Gradient Tolerance**: Start with default `1e-4`, tighten to `1e-5` or `1e-6` for higher precision
10. **Memory Management**: Backward pass doubles memory usage, consider smaller batch sizes for testing

## 🔗 Related Documentation

- [TRITON_INTEGRATION_GUIDE.md](./TRITON_INTEGRATION_GUIDE.md) - Complete technical guide
- [TRITON_README.md](./TRITON_README.md) - Local Triton support overview
- [Modal Documentation](https://modal.com/docs) - Modal platform docs

---

*This automation system provides a production-ready way to evaluate Triton kernels against KernelBench with **full forward and backward pass support**, minimal setup and maximum flexibility.* 