#!/bin/bash
# KernelBench AMD/ROCm Installation Script
# 
# This script sets up KernelBench to work with AMD GPUs using ROCm
# 
# Prerequisites:
# - ROCm 6.0+ installed on the system
# - AMD GPU with ROCm support
# - Python 3.8+ available

set -e  # Exit on any error

echo "🚀 KernelBench AMD/ROCm Setup Script"
echo "======================================"

# Check if ROCm is available
if ! command -v rocminfo &> /dev/null; then
    echo "❌ ERROR: rocminfo not found. Please install ROCm first."
    echo "   Visit: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
    exit 1
fi

# Check ROCm version
echo "🔍 Checking ROCm installation..."
ROCM_VERSION=$(rocminfo --version | head -1 | grep -oP 'version \K[0-9]+\.[0-9]+')
echo "   ROCm version: $ROCM_VERSION"

# Check for AMD GPU
GPU_COUNT=$(rocminfo | grep -c "Device Type:.*GPU" || echo "0")
if [ "$GPU_COUNT" -eq 0 ]; then
    echo "⚠️  WARNING: No AMD GPUs detected by ROCm"
    echo "   This might still work if you're setting up for later use"
else
    echo "✅ Found $GPU_COUNT AMD GPU(s)"
    rocminfo | grep "Marketing Name:" | head -3
fi

# Determine ROCm wheel version to use (using version comparison without bc)
MAJOR=$(echo "$ROCM_VERSION" | cut -d. -f1)
MINOR=$(echo "$ROCM_VERSION" | cut -d. -f2)

if [ "$MAJOR" -gt 6 ] || ([ "$MAJOR" -eq 6 ] && [ "$MINOR" -ge 1 ]); then
    WHEEL_VERSION="rocm6.1"
elif [ "$MAJOR" -eq 6 ] && [ "$MINOR" -eq 0 ]; then
    WHEEL_VERSION="rocm6.0"
elif [ "$MAJOR" -eq 5 ] && [ "$MINOR" -ge 7 ]; then
    WHEEL_VERSION="rocm5.7"
else
    echo "❌ ERROR: ROCm version $ROCM_VERSION is too old. Please upgrade to ROCm 5.7+"
    exit 1
fi

echo "🎯 Using PyTorch wheels for: $WHEEL_VERSION"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Using virtual environment: $VIRTUAL_ENV"
else
    echo "⚠️  WARNING: Not in a virtual environment. Consider using:"
    echo "   python3 -m venv venv && source venv/bin/activate"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. Please activate a virtual environment first."
        exit 1
    fi
fi

# Check if uv is available, prefer it over pip
if command -v uv &> /dev/null; then
    echo "✅ Using uv for faster installation"
    INSTALLER="uv pip"
else
    echo "📦 Using pip (consider installing uv for faster installs: pip install uv)"
    INSTALLER="pip"
fi

# Uninstall existing PyTorch if it's CUDA version
echo "🧹 Checking for existing PyTorch installation..."
CURRENT_TORCH=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
if [[ "$CURRENT_TORCH" != "none" && "$CURRENT_TORCH" != *"rocm"* ]]; then
    echo "   Found CUDA PyTorch: $CURRENT_TORCH"
    echo "   Uninstalling CUDA version..."
    $INSTALLER uninstall torch torchvision torchaudio -y
fi

# Install PyTorch with ROCm
echo "🔥 Installing PyTorch with ROCm support..."
$INSTALLER install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$WHEEL_VERSION

# Install other dependencies
echo "📚 Installing other dependencies..."
$INSTALLER install -r requirements-amd-rocm.txt

# Verify installation
echo "🔍 Verifying installation..."
python3 -c "
import torch
import sys
sys.path.append('src')

print('✅ PyTorch version:', torch.__version__)
print('✅ ROCm version:', torch.version.hip if hasattr(torch.version, 'hip') else 'None')
print('✅ GPU available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ Device count:', torch.cuda.device_count())
    print('✅ Device name:', torch.cuda.get_device_name(0))
else:
    print('⚠️  No GPU detected (this might be OK if setting up for later)')

# Test our device utilities
try:
    from device_utils import get_platform, get_platform_summary, validate_gpu_setup
    platform = get_platform()
    summary = get_platform_summary()
    validation = validate_gpu_setup()
    
    print('✅ Platform detected:', platform)
    print('✅ Platform summary:', summary)
    print('✅ Setup validation:', validation['message'])
    if validation['valid']:
        print('🎉 KernelBench AMD/ROCm setup complete and validated!')
    else:
        print('⚠️  Setup completed but GPU validation failed')
except ImportError:
    print('✅ Core installation complete (device_utils not found, that is OK)')
"

echo ""
echo "🎉 Installation Complete!"
echo "========================"
echo ""
echo "To test your setup, run:"
echo "  python3 scripts/run_and_check_triton.py ref_origin=local ref_arch_src_path=src/prompts/model_ex_add.py kernel_src_path=src/prompts/model_new_ex_add_triton.py"
echo ""
echo "For more information, see:"
echo "  - README.md"
echo "  - TRITON_INTEGRATION_GUIDE.md"
echo "" 