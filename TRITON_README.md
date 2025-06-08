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
def your_kernel(...):
    # Triton kernel implementation
    pass

def your_function(inputs):
    # Function that calls the Triton kernel
    return result

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *inputs):
        return your_function(*inputs)
```

## 📁 Example Kernels

### Element-wise Addition
- **Reference**: `src/prompts/model_ex_add.py`
- **Triton**: `src/prompts/model_new_ex_add_triton.py`

### Matrix Multiplication
- **Reference**: Available in KernelBench Level 1
- **Triton**: `src/prompts/model_new_ex_matmul_triton.py`

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

### Scripts

#### `scripts/run_and_check_triton.py`
Enhanced version of `run_and_check.py` with Triton support and auto-detection.

**Key Parameters:**
- `auto_detect=True`: Automatically detect kernel type
- `force_triton=True`: Force Triton evaluation
- `verbose=True`: Enable detailed logging

## 🏎️ Performance Considerations

### Triton vs CUDA
- **Triton**: JIT compiled, potentially slower first run but optimized for productivity
- **CUDA**: Pre-compiled, consistent performance but requires more low-level programming

### Compilation Differences
- **CUDA**: Compilation happens at load time via `torch.utils.cpp_extension`
- **Triton**: JIT compilation happens on first kernel execution
- **Caching**: Both use different caching mechanisms that are properly handled

## 🔍 Evaluation Metrics

The same metrics apply to both CUDA and Triton kernels:

- **Correctness**: Multiple trials with randomized inputs, tolerance-based comparison
- **Performance**: Wall-clock timing using CUDA events
- **Speedup**: Comparison against PyTorch eager and `torch.compile` baselines

Example output:
```
========================================
[Eval] triton kernel eval result: compiled=True, correctness=True, runtime=0.245ms
----------------------------------------
[Timing] PyTorch Reference Eager exec time: 0.892 ms
[Timing] PyTorch Reference torch.compile time: 0.156 ms
[Timing] Custom triton Kernel exec time: 0.245 ms
----------------------------------------
[Speedup] Speedup over eager: 3.64x
[Speedup] Speedup over torch.compile: 0.64x
========================================
```

## 🏗️ Integration with Existing Tools

### With existing scripts:
The Triton support integrates seamlessly with existing KernelBench evaluation tools:

```bash
# Generate samples (will work with Triton if detected)
python3 scripts/generate_samples.py run_name=triton_test level=1

# Evaluate samples (auto-detects kernel type)
python3 scripts/eval_from_generations.py run_name=triton_test level=1

# Analyze results (same metrics for CUDA and Triton)
python3 scripts/benchmark_eval_analysis.py run_name=triton_test level=1
```

## 🧪 Testing

Test the Triton evaluation with the provided examples:

```bash
# Test element-wise addition
python3 scripts/run_and_check_triton.py \
    ref_origin=local \
    ref_arch_src_path=src/prompts/model_ex_add.py \
    kernel_src_path=src/prompts/model_new_ex_add_triton.py \
    verbose=True

# Test matrix multiplication against KernelBench
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
- [ ] Integration with Triton's built-in benchmarking tools
- [ ] Support for Triton language extensions
- [ ] Triton kernel template generation
- [ ] Performance comparison dashboards

## 📚 References

- [Triton Documentation](https://triton-lang.org/)
- [KernelBench Paper](https://arxiv.org/abs/2502.10517)
- [Original KernelBench Repository](https://github.com/ScalingIntelligence/KernelBench)

---

For questions or issues related to Triton support, please refer to the main KernelBench documentation or open an issue on the repository. 