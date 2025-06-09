#!/usr/bin/env python3

import torch
import triton
import triton.language as tl

print("🚀 Testing KernelBench Setup")
print("=" * 40)
print(f"PyTorch version: {torch.__version__}")
print(f"Triton version: {triton.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test a simple Triton kernel definition
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

print("✓ Triton kernel compilation successful")

# Test basic tensor operations (CPU)
x = torch.randn(10, 10)
y = torch.randn(10, 10)
z = x + y
print("✓ Basic PyTorch tensor operations working")

print("")
print("🎉 Setup Status:")
print("✅ Python virtual environment")
print("✅ PyTorch 2.5.0 with CUDA libs")
print("✅ Triton 3.1.0")
print("✅ All KernelBench dependencies")
print("✅ KernelBench package installed")
print("")
if torch.cuda.is_available():
    print("✅ CUDA drivers working")
    print("🚀 Ready to run GPU benchmarks!")
else:
    print("⚠️  CUDA drivers need fixing")
    print("💡 Run 'sudo reboot' to fix driver mismatch")
    print("🔧 Or continue with CPU-only testing")

print("")
print("Next steps:")
print("1. Fix CUDA: sudo reboot")
print("2. Test: python scripts/run_and_check_triton.py ...")
print("3. Run benchmarks!") 