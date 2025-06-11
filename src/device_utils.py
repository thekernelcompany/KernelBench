import torch
import os
from typing import Union, Optional

def is_gpu_available() -> bool:
    """Check if either CUDA or ROCm GPU is available"""
    return torch.cuda.is_available()

def get_device_type() -> str:
    """Get the actual GPU device type (rocm/cuda/cpu)"""
    if not torch.cuda.is_available():
        return "cpu"
    
    # Check if we're using ROCm (PyTorch with ROCm has torch.version.hip)
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        return "rocm"
    else:
        return "cuda"

def get_default_device() -> torch.device:
    """Get the default GPU device"""
    device_type = get_device_type()
    if device_type in ["cuda", "rocm"]:
        # Both CUDA and ROCm use 'cuda' device type in PyTorch for compatibility
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")

def get_current_device() -> torch.device:
    """Get current GPU device, similar to torch.cuda.current_device()"""
    if is_gpu_available():
        return torch.cuda.current_device()
    else:
        return torch.device("cpu")

def set_device(device: Union[torch.device, int, str]):
    """Set the current GPU device"""
    if is_gpu_available():
        torch.cuda.set_device(device)

def synchronize(device: Optional[torch.device] = None):
    """Synchronize GPU operations"""
    if is_gpu_available():
        torch.cuda.synchronize(device=device)

def get_device_name(device: Optional[torch.device] = None) -> str:
    """Get the name of the GPU device"""
    if not is_gpu_available():
        return "CPU"
    
    try:
        if device is None:
            device = get_current_device()
        
        # For both CUDA and ROCm, use torch.cuda.get_device_name
        return torch.cuda.get_device_name(device)
    except Exception as e:
        # Fallback for AMD/ROCm devices where get_device_name might not work properly
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        if is_rocm:
            # Try to get device info from HIP/ROCm
            try:
                import subprocess
                result = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'Marketing Name:' in line and 'GPU' in line:
                            # Only get GPU marketing names, not CPU
                            return line.split('Marketing Name:')[1].strip()
                    # Fallback to a generic AMD GPU name
                    return "AMD GPU (ROCm)"
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                pass
        
        # Ultimate fallback
        return f"GPU (Unknown: {e})"

def get_device_properties(device: Optional[torch.device] = None):
    """Get device properties"""
    if not is_gpu_available():
        return None
    
    if device is None:
        device = get_current_device()
    
    try:
        return torch.cuda.get_device_properties(device)
    except Exception:
        # For ROCm, properties might not be fully available
        return None

def create_event(enable_timing: bool = True):
    """Create a GPU event for timing"""
    if is_gpu_available():
        return torch.cuda.Event(enable_timing=enable_timing)
    else:
        # Fallback for CPU-based timing
        return None

def to_device(tensor: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """Move tensor to specified device (handles both CUDA and ROCm transparently)"""
    if device is None:
        device = get_default_device()
    
    if device.type == "cuda" and is_gpu_available():
        # Works for both CUDA and ROCm (ROCm uses cuda device type internally)
        return tensor.cuda(device=device)
    else:
        return tensor.cpu()

def get_platform() -> str:
    """Get the actual GPU platform being used"""
    return get_device_type()

def get_backend_info() -> dict:
    """Get comprehensive information about the GPU backend and platform"""
    # Check if we're using ROCm (PyTorch with ROCm has torch.version.hip)
    is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
    is_cuda_available = torch.cuda.is_available()
    
    info = {
        "platform": get_platform(),  # rocm/cuda/cpu - what we're actually using
        "has_cuda": is_cuda_available and not is_rocm,  # True NVIDIA CUDA
        "has_rocm": is_rocm,  # AMD ROCm
        "device_count": 0,
        "backend": "none",
        "pytorch_device_type": "cpu"  # What PyTorch calls it internally
    }
    
    if is_cuda_available:
        info["device_count"] = torch.cuda.device_count()
        info["pytorch_device_type"] = "cuda"  # Both CUDA and ROCm use 'cuda' in PyTorch
        if is_rocm:
            info["backend"] = "rocm"
            info["rocm_version"] = torch.version.hip if hasattr(torch.version, 'hip') else None
        else:
            info["backend"] = "cuda"
            info["cuda_version"] = torch.version.cuda if torch.version.cuda else None
    
    return info 

def is_rocm_platform() -> bool:
    """Check if we're running on ROCm platform"""
    return get_platform() == "rocm"

def is_cuda_platform() -> bool:
    """Check if we're running on CUDA platform"""
    return get_platform() == "cuda"

def is_cpu_platform() -> bool:
    """Check if we're running on CPU only"""
    return get_platform() == "cpu"

def validate_gpu_setup() -> dict:
    """Validate GPU setup and return detailed status"""
    info = get_backend_info()
    status = {
        "valid": False,
        "platform": info["platform"],
        "message": "",
        "device_info": info
    }
    
    if info["platform"] == "cpu":
        status["message"] = "No GPU detected. Running on CPU."
        return status
    
    elif info["platform"] == "rocm":
        if info["device_count"] > 0:
            status["valid"] = True
            status["message"] = f"ROCm platform ready with {info['device_count']} AMD GPU(s)"
        else:
            status["message"] = "ROCm detected but no AMD GPUs found"
            
    elif info["platform"] == "cuda":
        if info["device_count"] > 0:
            status["valid"] = True
            status["message"] = f"CUDA platform ready with {info['device_count']} NVIDIA GPU(s)"
        else:
            status["message"] = "CUDA detected but no NVIDIA GPUs found"
    
    return status

def create_device(device_id: int = 0) -> torch.device:
    """Create a device object using the correct underlying PyTorch device type"""
    platform = get_platform()
    if platform in ["cuda", "rocm"]:
        # Both use 'cuda' device type in PyTorch
        return torch.device(f"cuda:{device_id}")
    else:
        return torch.device("cpu")

def get_platform_summary() -> str:
    """Get a human-readable summary of the current platform"""
    info = get_backend_info()
    platform = info["platform"]
    
    if platform == "rocm":
        device_name = get_device_name() if info["device_count"] > 0 else "Unknown"
        return f"AMD ROCm Platform (v{info.get('rocm_version', 'unknown')}) - {device_name}"
    elif platform == "cuda":
        device_name = get_device_name() if info["device_count"] > 0 else "Unknown"
        return f"NVIDIA CUDA Platform (v{info.get('cuda_version', 'unknown')}) - {device_name}"
    else:
        return "CPU Platform (No GPU)" 