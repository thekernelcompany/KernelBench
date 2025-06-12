# 🚀 Modal Triton Automation - Usage Guide

This document explains how to use the Modal-based automation for running Triton kernel evaluations on cloud GPUs.

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

### 3. Advanced Options

```bash
# Full production evaluation with custom settings
python run_triton_modal.py \
    --kernel my_triton_kernel.py \
    --level 2 \
    --problem_id 5 \
    --num_correct_trials 10 \
    --num_perf_trials 200 \
    --gpu H100 \
    --verbose
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

## 📁 File Structure

Your Triton kernel should follow this structure:

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
# Quick test first
python run_triton_modal.py --kernel my_matmul_kernel.py --level 1 --problem_id 1 --quick

# If it passes, run full evaluation
python run_triton_modal.py --kernel my_matmul_kernel.py --level 1 --problem_id 1 --verbose
```

### Step 3: Check Results

You'll see output like:
```
🚀 Starting Triton evaluation on H100
📄 Kernel: my_matmul_kernel.py
📚 Reference: KernelBench Level 1, Problem 1
🔧 Trials: 5 correctness, 100 performance

========================================
📊 EVALUATION RESULTS
========================================
Problem: 1_Square_matrix_multiplication_.py
Kernel Type: Triton
✅ Compiled: True
✅ Correctness: True
⚡ Runtime: 0.245 ms
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

### Success Output
```
✅ Compiled: True     # Kernel compiled successfully
✅ Correctness: True  # All correctness trials passed
⚡ Runtime: 0.245 ms  # Average runtime
```

### Performance Comparison
The system automatically compares against:
- PyTorch eager execution
- `torch.compile` baseline
- Reference implementation

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

1. **Start Small**: Use `--quick` for initial testing
2. **Check Examples**: Look at `TRITON_INTEGRATION_GUIDE.md` for working examples
3. **GPU Choice**: H100 for performance, L40S for cost-effectiveness
4. **Verbose Mode**: Use `--verbose` for detailed debugging information
5. **Incremental Testing**: Test simple operations before complex kernels

## 🔗 Related Documentation

- [TRITON_INTEGRATION_GUIDE.md](./TRITON_INTEGRATION_GUIDE.md) - Complete technical guide
- [TRITON_README.md](./TRITON_README.md) - Local Triton support overview
- [Modal Documentation](https://modal.com/docs) - Modal platform docs

---

*This automation system provides a production-ready way to evaluate Triton kernels against KernelBench with minimal setup and maximum flexibility.* 