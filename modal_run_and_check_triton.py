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
    num_gradient_trials: int = 3,
    gradient_tolerance: float = 1e-4,
    measure_backward_performance: bool = True
):
    """Remote function to run Triton evaluation"""
    import sys
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    print(f"🔧 Python path: {sys.path}")
    print(f"💻 GPU Architecture: {gpu_arch}")
    print(f"📝 Verbose mode: {verbose}")
    
    # Capture all output from the evaluation
    captured_output = io.StringIO()
    captured_error = io.StringIO()
    evaluation_output = ""
    evaluation_error = ""
    
    try:
        # Import required modules first (outside capture to avoid import noise)
        from src.eval import eval_kernel_against_ref_auto, detect_triton_kernel
        from src.utils import set_gpu_arch
        from datasets import load_dataset
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
        
        # Get reference source code
        if ref_origin == "kernelbench":
            if level is None or problem_id is None:
                return {
                    "status": "error",
                    "message": "level and problem_id required for kernelbench reference"
                }
            
            print(f"📚 Loading KernelBench Level {level}, Problem {problem_id}")
            dataset = load_dataset("ScalingIntelligence/KernelBench")
            level_dataset = dataset[f"level_{level}"]
            
            # Filter to get the specific problem
            problem = level_dataset.filter(lambda x: x["problem_id"] == problem_id)
            if len(problem) == 0:
                return {
                    "status": "error", 
                    "message": f"Problem {problem_id} not found in level {level}"
                }
                
            reference_src = problem["code"][0]
            problem_name = problem["name"][0]
            
            print(f"📋 Problem: {problem_name}")
            
        elif ref_origin == "local":
            if ref_src is None:
                return {
                    "status": "error",
                    "message": "ref_src required for local reference"
                }
            reference_src = ref_src
            problem_name = "local_reference"
            
        else:
            return {
                "status": "error",
                "message": f"Invalid ref_origin: {ref_origin}. Use 'kernelbench' or 'local'"
            }
        
        # Detect kernel type
        is_triton = detect_triton_kernel(kernel_src)
        kernel_type = "Triton" if is_triton else "CUDA"
        
        print(f"🔍 Detected kernel type: {kernel_type}")
        if verbose:
            print(f"🔧 Running evaluation with {num_correct_trials} correctness trials, {num_perf_trials} performance trials")
        
        # Now capture the detailed evaluation output
        print("\n" + "="*60)
        if test_backward_pass:
            print("🚀 STARTING DETAILED EVALUATION (WITH BACKWARD PASS)")
        else:
            print("🚀 STARTING DETAILED EVALUATION (FORWARD PASS ONLY)")
        print("="*60)
        

        
        with redirect_stdout(captured_output), redirect_stderr(captured_error):
            # Import necessary evaluation functions
            import sys
            sys.path.append('/workspace/scripts')
            from generate_baseline_time import measure_program_time
            
            # Run evaluation with explicit device handling
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
            if test_backward_pass:
                # Import backward pass evaluation function
                from src.eval import eval_kernel_backward_pass_auto
                
                # Run BOTH forward and backward pass evaluations to get complete metrics
                print("[INFO] Evaluating forward pass performance")
                forward_result = eval_kernel_against_ref_auto(
                    original_model_src=reference_src,
                    custom_model_src=kernel_src,
                    verbose=verbose,
                    measure_performance=True,
                    num_correct_trials=num_correct_trials,
                    num_perf_trials=num_perf_trials,
                    device=device
                )
                
                print("[INFO] Evaluating backward pass with gradient correctness")
                backward_result = eval_kernel_backward_pass_auto(
                    original_model_src=reference_src,
                    custom_model_src=kernel_src,
                    verbose=verbose,
                    measure_performance=measure_backward_performance,
                    num_correct_trials=num_correct_trials,
                    num_perf_trials=num_perf_trials,
                    num_gradient_trials=num_gradient_trials,
                    gradient_tolerance=gradient_tolerance,
                    device=device
                )
                
                # Combine results for unified processing
                result = backward_result  # Use backward result as primary
                # Store forward pass results in metadata for separate display
                result.metadata["forward_pass_result"] = {
                    "compiled": forward_result.compiled,
                    "correctness": forward_result.correctness,
                    "runtime": forward_result.runtime,
                    "runtime_stats": forward_result.runtime_stats,
                    "metadata": forward_result.metadata
                }
            else:
                # 1. Evaluate kernel against reference (forward pass only)
                print("[INFO] Evaluating kernel against reference code")
                result = eval_kernel_against_ref_auto(
                    original_model_src=reference_src,
                    custom_model_src=kernel_src,
                    verbose=verbose,
                    measure_performance=True,
                    num_correct_trials=num_correct_trials,
                    num_perf_trials=num_perf_trials,
                    device=device
                )
            # Note: result.runtime has units mismatch in KernelBench - it's actually in ms, not μs as documented
            kernel_exec_time = result.runtime if result.runtime > 0 else None  # Already in ms due to codebase bug
            
            # 2. Measure baseline reference times (like original script)
            print("[INFO] Measuring reference program time")
            
            # Measure PyTorch Eager baseline
            ref_time_eager_result = measure_program_time(
                ref_arch_name="Reference Program", 
                ref_arch_src=reference_src, 
                num_trials=num_perf_trials,
                use_torch_compile=False,
                device=device
            )
            ref_exec_eager_time = ref_time_eager_result.get("mean", None)
            
            # Measure torch.compile baseline
            ref_time_compile_result = measure_program_time(
                ref_arch_name="Reference Program", 
                ref_arch_src=reference_src, 
                num_trials=num_perf_trials,
                use_torch_compile=True,
                torch_compile_backend="inductor",
                torch_compile_options="default",
                device=device
            )
            ref_exec_compile_time = ref_time_compile_result.get("mean", None)
            
            # Just print the basic results like original run_and_check_triton.py
            kernel_type_str = result.metadata.get("kernel_type", "triton")
            print("="*40)
            print(f"[Eval] {kernel_type_str} kernel eval result: compiled={result.compiled}, correctness={result.correctness}, runtime={kernel_exec_time}ms")
            print("-"*40)
            print(f"[Timing] PyTorch Reference Eager exec time: {ref_exec_eager_time} ms")
            print(f"[Timing] PyTorch Reference torch.compile time: {ref_exec_compile_time} ms")
            print(f"[Timing] Custom {kernel_type_str} Kernel exec time: {kernel_exec_time} ms")
            print("-"*40)   
            
            if result.correctness and kernel_exec_time and ref_exec_eager_time and ref_exec_compile_time:
                speedup_eager = ref_exec_eager_time / kernel_exec_time
                speedup_compile = ref_exec_compile_time / kernel_exec_time
                print(f"[Speedup] Speedup over eager: {speedup_eager:.2f}x")
                print(f"[Speedup] Speedup over torch.compile: {speedup_compile:.2f}x")
            else:
                print("[Speedup] Speedup Not Available as Kernel did not pass correctness or timing failed")
            
            print("="*40)
        
        # Get captured output
        evaluation_output = captured_output.getvalue()
        evaluation_error = captured_error.getvalue()
        
        # Print all captured output so it appears in Modal logs
        if evaluation_output:
            print("📋 DETAILED EVALUATION OUTPUT:")
            print(evaluation_output)
        
        if evaluation_error:
            print("⚠️ EVALUATION WARNINGS/ERRORS:")
            print(evaluation_error)
        
        # Parse timing results from captured output
        speedup_eager = None
        speedup_compile = None
        ref_exec_eager_time = None
        ref_exec_compile_time = None
        kernel_exec_time = result.runtime / 1000.0 if result.runtime > 0 else None
        
        # Extract timing information from the captured output
        import re
        if evaluation_output:
            # Look for timing lines in the output
            eager_match = re.search(r'PyTorch Reference Eager exec time: ([\d.]+) ms', evaluation_output)
            compile_match = re.search(r'PyTorch Reference torch\.compile time: ([\d.]+) ms', evaluation_output)
            speedup_eager_match = re.search(r'Speedup over eager: ([\d.]+)x', evaluation_output)
            speedup_compile_match = re.search(r'Speedup over torch\.compile: ([\d.]+)x', evaluation_output)
            
            ref_exec_eager_time = float(eager_match.group(1)) if eager_match else None
            ref_exec_compile_time = float(compile_match.group(1)) if compile_match else None
            speedup_eager = float(speedup_eager_match.group(1)) if speedup_eager_match else None
            speedup_compile = float(speedup_compile_match.group(1)) if speedup_compile_match else None
        
        # Format results
        response = {
            "status": "completed",
            "problem_name": problem_name,
            "kernel_type": kernel_type,
            "test_backward_pass": test_backward_pass,
            "evaluation_output": evaluation_output,  # Include captured output
            "evaluation_error": evaluation_error
        }
        
        if test_backward_pass:
            # Extract forward pass results from stored metadata
            forward_pass_result = result.metadata.get("forward_pass_result", {})
            # The forward pass result.runtime is already in milliseconds from eval_triton_kernel_against_ref
            forward_runtime = forward_pass_result.get("runtime", 0) if forward_pass_result.get("runtime", 0) > 0 else None
            forward_runtime_stats = forward_pass_result.get("runtime_stats", {})
            
            # Calculate forward pass speedups
            forward_speedup_eager = None
            forward_speedup_compile = None
            if forward_runtime and ref_exec_eager_time and ref_exec_compile_time:
                forward_speedup_eager = ref_exec_eager_time / forward_runtime
                forward_speedup_compile = ref_exec_compile_time / forward_runtime
            
            # Extract backward pass metadata
            backward_pass_correctness = result.metadata.get("backward_pass_correctness", False)
            gradient_trials = result.metadata.get("gradient_trials", [])
            gradient_correctness_str = result.metadata.get("gradient_correctness", "Unknown")
            
            # Count successful gradient trials
            if gradient_trials:
                passed_trials = sum(1 for trial in gradient_trials if trial.get("passed", False))
                total_trials = len(gradient_trials)
                gradient_trials_summary = f"({passed_trials} / {total_trials})"
            else:
                gradient_trials_summary = gradient_correctness_str
            
            # Extract backward pass performance data
            backward_runtime = result.runtime if result.runtime > 0 else None
            
            # Calculate backward pass speedups
            backward_speedup_eager = None
            backward_speedup_compile = None
            if backward_runtime and ref_exec_eager_time and ref_exec_compile_time:
                backward_speedup_eager = ref_exec_eager_time / backward_runtime
                backward_speedup_compile = ref_exec_compile_time / backward_runtime
            
            # Create structured response with separate forward and backward pass results
            response.update({
                "compiled": result.compiled,
                "forward_pass": {
                    "correctness": forward_pass_result.get("correctness", result.correctness),
                    "runtime_us": forward_runtime * 1000 if forward_runtime else None,  # Convert ms to μs for consistency
                    "runtime_ms": forward_runtime,
                    "runtime_stats": forward_runtime_stats,
                    "ref_eager_time_ms": ref_exec_eager_time,
                    "ref_compile_time_ms": ref_exec_compile_time,
                    "speedup_over_eager": forward_speedup_eager,
                    "speedup_over_compile": forward_speedup_compile
                },
                "backward_pass": {
                    "passed": backward_pass_correctness,
                    "gradient_correctness": gradient_trials_summary,
                    "gradient_trials_details": gradient_trials,
                    "gradient_tolerance": gradient_tolerance,
                    "kernel_backward_exec_time_ms": backward_runtime,
                    "ref_backward_eager_time_ms": ref_exec_eager_time,
                    "ref_backward_compile_time_ms": ref_exec_compile_time,
                    "speedup_over_eager": backward_speedup_eager,
                    "speedup_over_torch_compile": backward_speedup_compile,
                    "gpu_memory_info": result.metadata.get("gpu_memory_info", {}),
                    "performance_stats": result.runtime_stats
                },
                "metadata": result.metadata
            })
        else:
            # Forward pass only results
            response.update({
                "compiled": result.compiled,
                "correctness": result.correctness,
                "runtime_us": result.runtime,  # runtime is in microseconds
                "runtime_ms": kernel_exec_time,  # runtime in milliseconds
                "ref_eager_time_ms": ref_exec_eager_time,
                "ref_compile_time_ms": ref_exec_compile_time,
                "speedup_over_eager": speedup_eager,
                "speedup_over_compile": speedup_compile,
                "metadata": result.metadata
            })
        
        if ref_origin == "kernelbench":
            response["level"] = level
            response["problem_id"] = problem_id
            
        # Print results summary
        print("\n" + "="*60)
        if test_backward_pass:
            print("📊 EVALUATION RESULTS (WITH BACKWARD PASS)")
        else:
            print("📊 EVALUATION RESULTS")
        print("="*60)
        print(f"Problem: {problem_name}")
        print(f"Kernel Type: {kernel_type}")
        print(f"✅ Compiled: {result.compiled}")
        
        if test_backward_pass:
            # Extract forward and backward pass results from metadata  
            forward_pass_result = result.metadata.get("forward_pass_result", {})
            # The forward pass result.runtime is already in milliseconds from eval_triton_kernel_against_ref
            forward_runtime = forward_pass_result.get("runtime", 0) if forward_pass_result.get("runtime", 0) > 0 else None
            
            backward_pass_correctness = result.metadata.get("backward_pass_correctness", False)
            gradient_trials = result.metadata.get("gradient_trials", [])
            gradient_correctness_str = result.metadata.get("gradient_correctness", "Unknown")
            
            # Count successful gradient trials
            if gradient_trials:
                passed_trials = sum(1 for trial in gradient_trials if trial.get("passed", False))
                total_trials = len(gradient_trials)
                gradient_trials_summary = f"({passed_trials} / {total_trials})"
            else:
                gradient_trials_summary = gradient_correctness_str
                
            print(f"✅ Forward Pass Correctness: {forward_pass_result.get('correctness', result.correctness)}")
            print(f"✅ Backward Pass Correctness: {backward_pass_correctness}")
            print(f"✅ Gradient Trials: {gradient_trials_summary}")
            
            # Show FORWARD pass performance
            if forward_runtime:
                print(f"⚡ Forward Pass Runtime: {forward_runtime:.3f} ms")
                
                # Calculate and show forward speedups
                if ref_exec_eager_time and ref_exec_eager_time > 0:
                    forward_speedup_eager = ref_exec_eager_time / forward_runtime
                    print(f"🚀 Forward Speedup over PyTorch Eager: {forward_speedup_eager:.2f}x")
                
                if ref_exec_compile_time and ref_exec_compile_time > 0:
                    forward_speedup_compile = ref_exec_compile_time / forward_runtime
                    print(f"🚀 Forward Speedup over torch.compile: {forward_speedup_compile:.2f}x")
            
            # Show BACKWARD pass performance
            if result.runtime > 0:
                print(f"⚡ Backward Pass Runtime: {result.runtime:.3f} ms")
                
                # Calculate and show backward speedups
                if ref_exec_eager_time and ref_exec_eager_time > 0:
                    backward_speedup_eager = ref_exec_eager_time / result.runtime
                    print(f"🚀 Backward Speedup over PyTorch Eager: {backward_speedup_eager:.2f}x")
                
                if ref_exec_compile_time and ref_exec_compile_time > 0:
                    backward_speedup_compile = ref_exec_compile_time / result.runtime
                    print(f"🚀 Backward Speedup over torch.compile: {backward_speedup_compile:.2f}x")
                    
            # Show reference times for context
            if ref_exec_eager_time and ref_exec_eager_time > 0:
                print(f"📊 Reference Eager time: {ref_exec_eager_time:.3f} ms")
            if ref_exec_compile_time and ref_exec_compile_time > 0:
                print(f"📊 Reference Compile time: {ref_exec_compile_time:.3f} ms")
            
            # Show memory info if available
            gpu_memory_info = result.metadata.get("gpu_memory_info", {})
            if gpu_memory_info:
                print(f"💾 GPU Memory: {gpu_memory_info}")
        else:
            print(f"✅ Correctness: {result.correctness}")
            
            if result.runtime > 0:
                print(f"⚡ Forward Runtime: {result.runtime / 1000.0:.3f} ms ({result.runtime:.1f} μs)")
            
        if result.metadata and verbose:
            print(f"📋 Metadata: {result.metadata}")
            
        # Print JSON output for easy parsing
        print("\n" + "="*60)
        print("📊 JSON RESULT")
        print("="*60)
        print(json.dumps(response, indent=2))
        print("="*60)
            
        return response
        
    except Exception as e:
        # Also capture any exception output
        evaluation_output = captured_output.getvalue()
        evaluation_error = captured_error.getvalue()
        
        error_response = {
            "status": "failed",
            "error": str(e),
            "error_type": type(e).__name__,
            "evaluation_output": evaluation_output,
            "evaluation_error": evaluation_error
        }
        print(f"❌ Error: {e}")
        if evaluation_output:
            print(f"📋 Output before error: {evaluation_output}")
        if evaluation_error:
            print(f"⚠️ Error output: {evaluation_error}")
        return error_response


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
        print("📊 MODAL EVALUATION RESULTS") 
        print("="*60)
        
        if result["status"] == "completed":
            print(f"✅ Status: {result['status']}")
            print(f"📋 Problem: {result['problem_name']}")
            print(f"🔍 Kernel Type: {result['kernel_type']}")
            
            if result.get('test_backward_pass', False):
                print("🔄 Mode: Forward + Backward Pass Evaluation")
                
                # Handle structured backward pass results
                if 'forward_pass' in result and 'backward_pass' in result:
                    # General compilation status
                    print(f"✅ Compiled: {result.get('compiled', 'Unknown')}")
                    
                    # Forward pass results  
                    fp = result['forward_pass']
                    print(f"✅ Forward Pass Correctness: {fp.get('correctness', 'Unknown')}")
                    if fp.get('runtime_ms') is not None:
                        print(f"⚡ Forward Pass Runtime: {fp['runtime_ms']:.3f} ms")
                    if fp.get('speedup_over_eager') and fp.get('speedup_over_compile'):
                        print(f"🚀 Forward Speedup over PyTorch Eager: {fp['speedup_over_eager']:.2f}x")
                        print(f"🚀 Forward Speedup over torch.compile: {fp['speedup_over_compile']:.2f}x")
                    
                    # Backward pass results
                    bp = result['backward_pass']
                    print(f"✅ Backward Pass Correctness: {bp.get('passed', 'Unknown')}")
                    print(f"✅ Gradient Trials: {bp.get('gradient_correctness', 'Unknown')}")
                    
                    # Show backward pass performance
                    if bp.get('kernel_backward_exec_time_ms') is not None:
                        print(f"⚡ Backward Pass Runtime: {bp['kernel_backward_exec_time_ms']:.3f} ms")
                    
                    # Show backward speedup results
                    if bp.get('speedup_over_eager') and bp.get('speedup_over_torch_compile'):
                        print(f"🚀 Backward Speedup over PyTorch Eager: {bp['speedup_over_eager']:.2f}x")
                        print(f"🚀 Backward Speedup over torch.compile: {bp['speedup_over_torch_compile']:.2f}x")
                    
                    # Show reference timing context
                    if fp.get('ref_eager_time_ms') or bp.get('ref_backward_eager_time_ms'):
                        eager_time = fp.get('ref_eager_time_ms') or bp.get('ref_backward_eager_time_ms')
                        print(f"📊 Reference Eager time: {eager_time:.3f} ms")
                    if fp.get('ref_compile_time_ms') or bp.get('ref_backward_compile_time_ms'):
                        compile_time = fp.get('ref_compile_time_ms') or bp.get('ref_backward_compile_time_ms')
                        print(f"📊 Reference Compile time: {compile_time:.3f} ms")
                    
                    # Show GPU memory info
                    gpu_memory = bp.get('gpu_memory_info', {})
                    if gpu_memory:
                        print(f"💾 GPU Memory: {gpu_memory}")
                        
                else:
                    # Legacy format fallback
                    print(f"✅ Compiled: {result.get('compiled', 'Unknown')}")
                    print(f"✅ Forward Pass Correctness: {result.get('correctness', 'Unknown')}")
                    if result.get('runtime_ms') is not None:
                        print(f"⚡ Forward Runtime: {result['runtime_ms']:.3f} ms")
                    print("⚠️ Backward pass attempted (check detailed logs for results)")
                    
            else:
                # Forward pass only results
                print("➡️ Mode: Forward Pass Only")
                print(f"✅ Compiled: {result.get('compiled', 'Unknown')}")
                print(f"✅ Correctness: {result.get('correctness', 'Unknown')}")
                
                if result.get('runtime_ms') and result['runtime_ms'] is not None:
                    print(f"⚡ Runtime: {result['runtime_ms']:.3f} ms ({result.get('runtime_us', 0):.1f} μs)")
                    
                # Show speedup results if available
                if result.get('speedup_over_eager') and result.get('speedup_over_compile'):
                    print(f"🚀 Speedup over PyTorch Eager: {result['speedup_over_eager']:.2f}x")
                    print(f"🚀 Speedup over torch.compile: {result['speedup_over_compile']:.2f}x")
                    print(f"📊 Reference Eager time: {result.get('ref_eager_time_ms', 0):.3f} ms")
                    print(f"📊 Reference Compile time: {result.get('ref_compile_time_ms', 0):.3f} ms")
                
            if result.get('level') and result.get('problem_id'):
                print(f"📚 Level: {result['level']}, Problem: {result['problem_id']}")
            
            # Show the detailed evaluation output with speedups!
            if result.get('evaluation_output'):
                print("\n" + "="*60)
                print("📋 DETAILED EVALUATION LOGS (WITH SPEEDUPS)")
                print("="*60)
                print(result['evaluation_output'])
                
            if result.get('evaluation_error'):
                print("\n⚠️ EVALUATION WARNINGS:")
                print(result['evaluation_error'])
                
            if result.get('metadata') and verbose:
                print(f"\n📋 Detailed Metadata:")
                for key, value in result['metadata'].items():
                    print(f"  {key}: {value}")
                    
        elif result["status"] == "failed":
            print(f"❌ Status: {result['status']}")
            print(f"❌ Error: {result['error']}")
            print(f"🔧 Error Type: {result['error_type']}")
            
            # Show captured output even for failures
            if result.get('evaluation_output'):
                print(f"\n📋 Output before failure:")
                print(result['evaluation_output'])
            if result.get('evaluation_error'):
                print(f"\n⚠️ Error details:")
                print(result['evaluation_error'])
            
        elif result["status"] == "error":
            print(f"⚠️  Status: {result['status']}")
            print(f"⚠️  Message: {result['message']}")
            
        print("\n🎯 Modal evaluation completed!")
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