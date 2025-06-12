import modal
import os
import sys
import json
from typing import Dict, Any, Optional

app = modal.App("triton_run_and_check")

"""
🚀 Modal Run and Check for Triton Kernels
Equivalent to: python scripts/run_and_check_triton.py but on Modal H100 with full backward pass support

Usage examples:
# Forward pass only against KernelBench dataset
python modal_run_and_check_triton.py ref_origin=kernelbench level=1 problem_id=3 kernel_src_path=my_kernel.py

# Forward + Backward pass evaluation with gradient testing
python modal_run_and_check_triton.py ref_origin=kernelbench level=1 problem_id=3 kernel_src_path=my_kernel.py test_backward_pass=True

# Against local reference with custom gradient settings
python modal_run_and_check_triton.py ref_origin=local ref_arch_src_path=reference.py kernel_src_path=my_kernel.py test_backward_pass=True num_gradient_trials=5 gradient_tolerance=1e-5
"""

# Modal image setup based on Dockerfile (updated for Modal 1.0)
image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel")
    .apt_install(
        "git",
        "build-essential",
        "ninja-build",
        "cmake",
        "vim",
        "curl",
        "wget"
    )
    .pip_install("triton")  # Install Triton first
    .add_local_file("requirements.txt", "/workspace/requirements.txt", copy=True)
    .run_commands("pip install -r /workspace/requirements.txt")
    .add_local_dir(".", "/workspace", copy=True)  # Copy entire project
    .workdir("/workspace")
    .run_commands("pip install -e .")  # Install KernelBench in development mode
    .env({
        "CUDA_VISIBLE_DEVICES": "0",
        "TRITON_PRINT_AUTOTUNING": "0"
    })
)

# Convenience function for command-line style usage
@app.function(
    image=image,
    gpu="H100",
    timeout=600  # 10 minute timeout
)
def run_triton_check_remote(
    kernel_src: str,
    ref_origin: str = "kernelbench",
    ref_src: Optional[str] = None,
    level: Optional[int] = None,
    problem_id: Optional[int] = None,
    num_correct_trials: int = 5,
    num_perf_trials: int = 100,
    verbose: bool = True,
    gpu_arch: str = "Hopper",
    test_backward_pass: bool = False,
    num_gradient_trials: int = 5,
    gradient_tolerance: float = 1e-4,
    measure_backward_performance: bool = True
):
    """Remote function to run Triton evaluation - exact same output as local terminal"""
    import sys
    import tempfile
    import os
    from pathlib import Path
    
    print(f"🔧 Python path: {sys.path}")
    print(f"💻 GPU Architecture: {gpu_arch}")
    print(f"📝 Verbose mode: {verbose}")
    
    try:
        # Handle the case where kernel_src might be a file path instead of content
        # This happens when using Modal CLI directly vs. the wrapper script
        if len(kernel_src) < 500 and ('/' in kernel_src or kernel_src.endswith('.py')):
            # This looks like a file path, not source code
            print(f"🔍 kernel_src appears to be a file path: {kernel_src}")
            if os.path.exists(f"/workspace/{kernel_src}"):
                with open(f"/workspace/{kernel_src}", 'r') as f:
                    kernel_src = f.read()
                print(f"✅ Read {len(kernel_src)} chars from file")
            elif os.path.exists(kernel_src):
                with open(kernel_src, 'r') as f:
                    kernel_src = f.read()
                print(f"✅ Read {len(kernel_src)} chars from file")
            else:
                print(f"❌ File not found: {kernel_src}")
                return {"status": "error", "message": f"Kernel file not found: {kernel_src}"}
        else:
            print(f"✅ kernel_src appears to be actual source code ({len(kernel_src)} chars)")
        
        # Import the actual main function from run_and_check_triton.py
        sys.path.append('/workspace/scripts')
        from run_and_check_triton import ScriptConfig
        from src.utils import set_gpu_arch
        import torch
        
        print(f"🎯 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🎯 GPU count: {torch.cuda.device_count()}")
            print(f"🎯 Current device: {torch.cuda.current_device()}")
            print(f"🎯 Device name: {torch.cuda.get_device_name()}")
        
        # Set GPU architecture based on gpu_arch parameter
        if gpu_arch == "H100":
            set_gpu_arch(["Hopper"])
        elif gpu_arch == "L40S" or gpu_arch == "L4":
            set_gpu_arch(["Ada"])
        elif gpu_arch == "A100" or gpu_arch == "A10G":
            set_gpu_arch(["Ampere"])
        elif gpu_arch == "T4":
            set_gpu_arch(["Turing"])
        else:
            # Default to Hopper for H100
            set_gpu_arch(["Hopper"])
        
        # Create temporary files for kernel and reference (if local)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as kernel_file:
            kernel_file.write(kernel_src)
            kernel_file.flush()  # Ensure content is written to disk
            kernel_file_path = kernel_file.name
        
        print(f"📝 Created temporary kernel file: {kernel_file_path}")
        
        ref_file_path = None
        if ref_origin == "local" and ref_src:
            # Handle the case where ref_src might also be a file path
            if len(ref_src) < 500 and ('/' in ref_src or ref_src.endswith('.py')):
                print(f"🔍 ref_src appears to be a file path: {ref_src}")
                if os.path.exists(f"/workspace/{ref_src}"):
                    with open(f"/workspace/{ref_src}", 'r') as f:
                        ref_src = f.read()
                    print(f"✅ Read {len(ref_src)} chars from reference file")
                elif os.path.exists(ref_src):
                    with open(ref_src, 'r') as f:
                        ref_src = f.read()
                    print(f"✅ Read {len(ref_src)} chars from reference file")
                else:
                    return {"status": "error", "message": f"Reference file not found: {ref_src}"}
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as ref_file:
                ref_file.write(ref_src)
                ref_file.flush()  # Ensure content is written to disk
                ref_file_path = ref_file.name
        
        # Create configuration object exactly like the command line would
        config = ScriptConfig()
        config.ref_origin = ref_origin
        config.kernel_src_path = kernel_file_path
        config.num_correct_trials = num_correct_trials
        config.num_perf_trials = num_perf_trials
        config.verbose = verbose
        config.test_backward_pass = test_backward_pass
        config.num_gradient_trials = num_gradient_trials
        config.gradient_tolerance = gradient_tolerance
        config.measure_backward_performance = measure_backward_performance
        config.auto_detect = True
        config.force_triton = False
        config.measure_performance = True
        config.timeout = 300
        config.build_dir_prefix = ""
        config.clear_cache = False
        
        # Set GPU architecture
        if gpu_arch == "H100":
            config.gpu_arch = ["Hopper"]
        elif gpu_arch == "L40S" or gpu_arch == "L4":
            config.gpu_arch = ["Ada"]
        elif gpu_arch == "A100" or gpu_arch == "A10G":
            config.gpu_arch = ["Ampere"]
        elif gpu_arch == "T4":
            config.gpu_arch = ["Turing"]
        else:
            config.gpu_arch = ["Hopper"]  # Default
        
        if ref_origin == "local":
            if ref_file_path is None:
                return {
                    "status": "error",
                    "message": "ref_src required for local reference"
                }
            config.ref_arch_src_path = ref_file_path
        elif ref_origin == "kernelbench":
            if level is None or problem_id is None:
                return {
                    "status": "error",
                    "message": "level and problem_id required for kernelbench reference"
                }
            config.dataset_name = "ScalingIntelligence/KernelBench"
            config.level = level
            config.problem_id = problem_id
        else:
            return {
                "status": "error",
                "message": f"Invalid ref_origin: {ref_origin}. Use 'kernelbench' or 'local'"
            }
        
        print("\n" + "="*60)
        print("🚀 STARTING EVALUATION WITH EXACT TERMINAL OUTPUT")
        print("="*60)
        
        # Ensure output is flushed in Modal environment
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Call the actual main function - this will produce the exact same output as terminal
        try:
            # Build command for subprocess
            cmd = [
                "python",
                "/workspace/scripts/run_and_check_triton.py",
                f"ref_origin={config.ref_origin}",
                f"kernel_src_path={config.kernel_src_path}",
                f"num_correct_trials={config.num_correct_trials}",
                f"num_perf_trials={config.num_perf_trials}",
                f"verbose={config.verbose}",
                f"test_backward_pass={config.test_backward_pass}",
                f"num_gradient_trials={config.num_gradient_trials}",
                f"gradient_tolerance={config.gradient_tolerance}",
                f"measure_backward_performance={config.measure_backward_performance}",
                f"auto_detect={config.auto_detect}",
                f"force_triton={config.force_triton}",
                f"measure_performance={config.measure_performance}",
                f"timeout={config.timeout}",
                f"build_dir_prefix={config.build_dir_prefix}",
                f"clear_cache={config.clear_cache}"
            ]
            
            # Add reference-specific arguments
            if config.ref_origin == "local":
                cmd.append(f"ref_arch_src_path={config.ref_arch_src_path}")
            elif config.ref_origin == "kernelbench":
                cmd.append(f"dataset_name={config.dataset_name}")
                cmd.append(f"level={config.level}")
                cmd.append(f"problem_id={config.problem_id}")
            
            # Add gpu_arch as a stringified list 
            gpu_arch_str = "[" + ",".join(f'"{arch}"' for arch in config.gpu_arch) + "]"
            cmd.append(f"gpu_arch={gpu_arch_str}")

            if verbose: # Print command if verbose
                print(f"Executing command: {' '.join(cmd)}", file=sys.stderr)
            
            import subprocess # Import here for clarity within the changed block
            process = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            script_json_output_content = None
            if process.returncode == 0 and process.stdout:
                import re
                match = re.search(r"INFO] Evaluation results saved to (runs/successful/run_.*?\.json)", process.stdout)
                if match:
                    json_file_path_in_modal = "/workspace/" + match.group(1)
                    print(f"Attempting to read script-generated JSON from: {json_file_path_in_modal}", file=sys.stderr)
                    try:
                        if os.path.exists(json_file_path_in_modal):
                            with open(json_file_path_in_modal, 'r') as f_json:
                                script_json_output_content = f_json.read()
                            print(f"Successfully read {len(script_json_output_content)} chars from script JSON.", file=sys.stderr)
                        else:
                            print(f"Script-generated JSON file not found at: {json_file_path_in_modal}", file=sys.stderr)
                    except Exception as e_read:
                        print(f"Error reading script-generated JSON: {e_read}", file=sys.stderr)
                else:
                    print("Could not find path to script-generated JSON in stdout.", file=sys.stderr)

            if process.returncode == 0:
                return {
                    "status": "completed",
                    "message": "Evaluation completed successfully via subprocess",
                    "detailed_stdout": process.stdout,
                    "detailed_stderr": process.stderr,
                    "script_output_json_content": script_json_output_content
                }
            else:
                return {
                    "status": "failed",
                    "error": f"Script exited with code {process.returncode}",
                    "detailed_stdout": process.stdout,
                    "detailed_stderr": process.stderr,
                    "script_output_json_content": script_json_output_content,
                    "error_type": "SubprocessError"
                }
            
        except Exception as e: # Catch other exceptions during setup/subprocess call
            print(f"❌ Error during subprocess execution or setup: {type(e).__name__} - {e}")
            import traceback
            tb_str = traceback.format_exc()
            print(tb_str, file=sys.stderr)
            return {
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
                "detailed_stdout": "",
                "detailed_stderr": tb_str,
                "script_output_json_content": None
            }
        
        finally:
            # Always restore original sys.argv (No longer needed as we don't modify it)
            # sys.argv = original_argv 
            
            # Clean up temporary files
            try:
                if 'kernel_file_path' in locals() and os.path.exists(kernel_file_path):
                    os.unlink(kernel_file_path)
                if 'ref_file_path' in locals() and ref_file_path and os.path.exists(ref_file_path):
                    os.unlink(ref_file_path)
            except:
                pass  # Ignore cleanup errors
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "error_type": type(e).__name__
        }


def run_triton_check(
    kernel_src_path: str,
    ref_origin: str = "kernelbench",
    ref_arch_src_path: Optional[str] = None,
    level: Optional[int] = None,
    problem_id: Optional[int] = None,
    num_correct_trials: int = 5,
    num_perf_trials: int = 100,
    verbose: bool = True,
    gpu: str = "H100",
    test_backward_pass: bool = False,
    num_gradient_trials: int = 3,
    gradient_tolerance: float = 1e-4,
    measure_backward_performance: bool = True
):
    """
    Run Triton kernel evaluation with command-line style interface
    
    Args:
        kernel_src_path: Path to your Triton kernel file
        ref_origin: "kernelbench" or "local"
        ref_arch_src_path: Path to reference file (if ref_origin="local") 
        level: KernelBench level (if ref_origin="kernelbench")
        problem_id: KernelBench problem ID (if ref_origin="kernelbench")
        num_correct_trials: Number of correctness trials
        num_perf_trials: Number of performance trials
        verbose: Enable verbose output
        gpu: GPU type ("H100", "L40S", etc.)
        test_backward_pass: Enable backward pass testing
        num_gradient_trials: Number of gradient correctness trials
        gradient_tolerance: Tolerance for gradient checking
        measure_backward_performance: Enable backward pass performance measurement
    """
    
    print(f"🚀 Starting Triton evaluation on {gpu}")
    if test_backward_pass:
        print("🔄 Mode: Forward + Backward Pass Evaluation")
    else:
        print("➡️ Mode: Forward Pass Only")
    print(f"📄 Kernel: {kernel_src_path}")
    
    # Read kernel source
    if not os.path.exists(kernel_src_path):
        print(f"❌ Error: Kernel file not found: {kernel_src_path}")
        return
        
    with open(kernel_src_path, 'r') as f:
        kernel_src = f.read()
    
    # Read reference source if local
    ref_src = None
    if ref_origin == "local":
        if not ref_arch_src_path:
            print("❌ Error: ref_arch_src_path required for local reference")
            return
        if not os.path.exists(ref_arch_src_path):
            print(f"❌ Error: Reference file not found: {ref_arch_src_path}")
            return
            
        with open(ref_arch_src_path, 'r') as f:
            ref_src = f.read()
        print(f"📋 Reference: {ref_arch_src_path}")
    else:
        print(f"📚 Reference: KernelBench Level {level}, Problem {problem_id}")
    
    # Run evaluation on Modal
    print("\n" + "="*60)
    print("🚀 STARTING MODAL H100 EVALUATION")
    print("="*60)
    
    with app.run():
        print("⏳ Launching Modal function on H100...")
        result = run_triton_check_remote.remote(
            kernel_src=kernel_src,
            ref_origin=ref_origin,
            ref_src=ref_src,
            level=level,
            problem_id=problem_id,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            verbose=verbose,
            gpu_arch=gpu,
            test_backward_pass=test_backward_pass,
            num_gradient_trials=num_gradient_trials,
            gradient_tolerance=gradient_tolerance,
            measure_backward_performance=measure_backward_performance
        )
        
        print("\n" + "="*60)
        print("🎯 MODAL EVALUATION COMPLETED!")
        print("="*60)
        
        # Print the detailed output received from the remote function
        if "detailed_stdout" in result and result["detailed_stdout"]:
            print("--- DETAILED STDOUT FROM MODAL FUNCTION ---")
            print(result["detailed_stdout"])
            print("--- END DETAILED STDOUT ---")
        
        if "detailed_stderr" in result and result["detailed_stderr"]:
            print("--- DETAILED STDERR FROM MODAL FUNCTION ---", file=sys.stderr)
            print(result["detailed_stderr"], file=sys.stderr)
            print("--- END DETAILED STDERR ---", file=sys.stderr)

        if result["status"] == "completed":
            print(f"✅ Evaluation completed successfully")
        elif result["status"] == "failed":
            print(f"❌ Evaluation failed: {result.get('error', 'Unknown error')}")
        elif result["status"] == "error":
            print(f"⚠️ Configuration error: {result.get('message', 'Unknown error')}")
        
        print("="*60)
        print("📊 All output above shows the exact same format as local terminal execution")
        return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run and check Triton kernels on Modal")
    parser.add_argument("--kernel_src_path", required=True, help="Path to Triton kernel file")
    parser.add_argument("--ref_origin", default="kernelbench", choices=["kernelbench", "local"], 
                       help="Reference origin: kernelbench or local")
    parser.add_argument("--ref_arch_src_path", help="Path to reference file (for local)")
    parser.add_argument("--level", type=int, help="KernelBench level")
    parser.add_argument("--problem_id", type=int, help="KernelBench problem ID")
    parser.add_argument("--num_correct_trials", type=int, default=5, help="Number of correctness trials")
    parser.add_argument("--num_perf_trials", type=int, default=100, help="Number of performance trials")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--gpu", default="H100", help="GPU type (H100, L40S, A100, etc.)")
    parser.add_argument("--test_backward_pass", action="store_true", help="Enable backward pass testing")
    parser.add_argument("--num_gradient_trials", type=int, default=3, help="Number of gradient correctness trials")
    parser.add_argument("--gradient_tolerance", type=float, default=1e-4, help="Tolerance for gradient checking")
    parser.add_argument("--measure_backward_performance", action="store_true", default=True, 
                       help="Enable backward pass performance measurement")
    
    args = parser.parse_args()
    
    result = run_triton_check(
        kernel_src_path=args.kernel_src_path,
        ref_origin=args.ref_origin,
        ref_arch_src_path=args.ref_arch_src_path,
        level=args.level,
        problem_id=args.problem_id,
        num_correct_trials=args.num_correct_trials,
        num_perf_trials=args.num_perf_trials,
        verbose=args.verbose,
        gpu=args.gpu,
        test_backward_pass=args.test_backward_pass,
        num_gradient_trials=args.num_gradient_trials,
        gradient_tolerance=args.gradient_tolerance,
        measure_backward_performance=args.measure_backward_performance
    )
    # Print the result dictionary as a JSON string to stdout
    # This allows the calling script (run_triton_modal.py) to capture and parse it.
    print(json.dumps(result)) 