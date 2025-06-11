# KernelBench AMD/ROCm Setup Guide

This guide helps you set up KernelBench to work with **AMD GPUs using ROCm**.

## Prerequisites

### 1. ROCm Installation
- **ROCm 5.7+** installed on your system
- Supported AMD GPUs (e.g., MI300X, MI250X, RX 7900 XTX, etc.)
- Linux system (ROCm is primarily Linux-based)

Install ROCm following the [official guide](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html).

### 2. Verify ROCm Setup
```bash
# Check ROCm installation
rocminfo --version

# Check for AMD GPUs
rocminfo | grep "Marketing Name"
```

## Quick Setup (Recommended)

### Option 1: Automated Script
```bash
# Clone the repository
git clone <repository-url>
cd KernelBench

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Run the automated setup script
./install-amd-rocm.sh
```

### Option 2: Manual Installation
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (using uv for speed, or pip)
uv pip install -r requirements-amd-rocm.txt --index-url https://download.pytorch.org/whl/rocm6.1
# OR with pip:
# pip install -r requirements-amd-rocm.txt --index-url https://download.pytorch.org/whl/rocm6.1
```

## Testing Your Setup

### Basic Platform Detection
```bash
python3 -c "
import sys
sys.path.append('src')
from device_utils import get_platform, get_platform_summary, validate_gpu_setup

platform = get_platform()
summary = get_platform_summary()
validation = validate_gpu_setup()

print('Platform:', platform)
print('Summary:', summary)
print('Status:', validation['message'])
"
```

### Run KernelBench Test
```bash
python3 scripts/run_and_check_triton.py \
    ref_origin=local \
    ref_arch_src_path=src/prompts/model_ex_add.py \
    kernel_src_path=src/prompts/model_new_ex_add_triton.py
```

Expected output should show:
- ✅ AMD GPU detection
- ✅ Triton kernel compilation
- ✅ Correctness tests passing
- ✅ Performance benchmarks

## ROCm Version Compatibility

| ROCm Version | PyTorch Wheel | Status |
|--------------|---------------|---------|
| 6.1.x        | rocm6.1       | ✅ Recommended |
| 6.0.x        | rocm6.0       | ✅ Supported |
| 5.7.x        | rocm5.7       | ✅ Supported |
| < 5.7        | -             | ❌ Not supported |

## Supported AMD GPUs

### Data Center GPUs
- **MI300X** (CDNA3) - Fully supported ✅
- **MI250X** (CDNA2) - Fully supported ✅
- **MI210** (CDNA2) - Supported ✅
- **MI100** (CDNA1) - Supported ✅

### Consumer GPUs (with ROCm)
- **RX 7900 XTX** (RDNA3) - Supported ✅
- **RX 7900 XT** (RDNA3) - Supported ✅
- **RX 6900 XT** (RDNA2) - Limited support ⚠️

## Troubleshooting

### Common Issues

#### 1. "ROCm not found"
```bash
# Install ROCm following AMD's official guide
# Ubuntu/Debian:
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.1.60103-1_all.deb
sudo dpkg -i amdgpu-install_6.1.60103-1_all.deb
sudo amdgpu-install --usecase=rocm
```

#### 2. "No AMD GPU detected"
```bash
# Check if GPU is visible to ROCm
rocminfo
ls /dev/kfd  # Should exist
```

#### 3. "PyTorch CUDA version installed"
```bash
# Uninstall CUDA version and reinstall ROCm version
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
```

#### 4. "Triton compilation errors"
- Ensure you have the ROCm version of PyTorch installed
- Check that your GPU is supported by ROCm
- Verify ROCm environment variables are set

### Environment Variables
```bash
# Add to your ~/.bashrc or ~/.profile
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0  # Use first GPU
```

## Performance Notes

- **ROCm Performance**: Generally excellent on data center GPUs (MI series)
- **Consumer GPU**: Performance varies, data center GPUs recommended for production
- **Memory**: AMD GPUs often have large VRAM (e.g., MI300X has 192GB HBM3)
- **Triton**: AMD backend is actively developed and improving

## Differences from CUDA Version

### What's the Same
- ✅ All KernelBench functionality
- ✅ Triton kernel development
- ✅ Performance benchmarking
- ✅ API compatibility (uses `torch.cuda.*`)

### What's Different
- 🔄 PyTorch installation source (ROCm wheels)
- 🔄 Some vendor-specific optimizations
- 🔄 GPU architecture naming (gfx942 vs sm_90)

## Contributing

When contributing code that should work on both NVIDIA and AMD:
- Use the `device_utils.py` abstraction layer
- Test on both CUDA and ROCm when possible
- Avoid vendor-specific optimizations in core code

## Getting Help

- **ROCm Issues**: [ROCm GitHub](https://github.com/RadeonOpenCompute/ROCm)
- **PyTorch ROCm**: [PyTorch Forum](https://discuss.pytorch.org)
- **Triton AMD**: [Triton GitHub](https://github.com/triton-lang/triton)
- **KernelBench**: Create an issue in this repository 