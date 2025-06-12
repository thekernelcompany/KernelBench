import modal
import os
import sys
from typing import Dict, Any, Optional

# New imports
import json
import datetime
import re

app = modal.App("triton_run_and_check")

"""
🚀 Modal Run and Check for Triton Kernels
Equivalent to: python scripts/run_and_check_triton.py but on Modal H100

Usage examples:
# Against KernelBench dataset
python modal_run_and_check_triton.py ref_origin=kernelbench level=1 problem_id=3 kernel_src_path=my_kernel.py

# Against local reference
python modal_run_and_check_triton.py ref_origin=local ref_arch_src_path=reference.py kernel_src_path=my_kernel.py
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
    kernel_file_identifier: Optional[str] = None,
    reference_file_identifier: Optional[str] = None
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
        print("🚀 STARTING DETAILED EVALUATION")
        print("="*60)
        

        
        with redirect_stdout(captured_output), redirect_stderr(captured_error):
            # Import measure_program_time for baseline measurements
            import sys
            sys.path.append('/workspace/scripts')
            from generate_baseline_time import measure_program_time
            
            # Run evaluation with explicit device handling
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
            # 1. Evaluate kernel against reference
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
            
            # 3. Print results in original format
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
            "status": "completed", # This outer status means the Modal function itself completed
            "problem_name": problem_name,
            "kernel_type": kernel_type,
            "compiled": result.compiled,
            "correctness": result.correctness,
            "runtime_us": result.runtime,
            "runtime_ms": kernel_exec_time, 
            "ref_eager_time_ms": ref_exec_eager_time,
            "ref_compile_time_ms": ref_exec_compile_time,
            "speedup_over_eager": speedup_eager,
            "speedup_over_compile": speedup_compile,
            "metadata": result.metadata,
            "evaluation_output": evaluation_output,
            "evaluation_error": evaluation_error
        }
        
        if ref_origin == "kernelbench":
            response["level"] = level
            response["problem_id"] = problem_id
            
        # Print results summary
        print("\n" + "="*60)
        print("📊 EVALUATION RESULTS")  
        print("="*60)
        print(f"Problem: {problem_name}")
        print(f"Kernel Type: {kernel_type}")
        print(f"✅ Compiled: {result.compiled}")
        print(f"✅ Correctness: {result.correctness}")
        
        if result.runtime > 0:
            print(f"⚡ Runtime: {result.runtime / 1000.0:.3f} ms ({result.runtime:.1f} μs)")
            
        if result.metadata and verbose:
            print(f"📋 Metadata: {result.metadata}")
            
        # --- BEGIN REVISED JSON LOG PREPARATION ---
        run_data = {
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
            "triton_code_path": kernel_file_identifier if kernel_file_identifier else "N/A (ran remotely via Modal - path not passed)",
            "reference_code_identifier": reference_file_identifier if reference_file_identifier else "Local Reference (path not passed)",
            "kernel_type": kernel_type
            # status, compilation_passed, correctness_passed will be set below
        }

        # Determine problem_name_for_filename and specific reference_code_identifier
        problem_name_for_filename = "unknown_problem"
        if ref_origin == "kernelbench":
            # problem_name is already fetched from dataset earlier in this function
            run_data["reference_code_identifier"] = f"kernelbench:level_{level}:problem_{problem_id}:{problem_name}"
            problem_name_for_filename = problem_name
        elif ref_origin == "local":
            run_data["reference_code_identifier"] = reference_file_identifier if reference_file_identifier else "Local Reference (path not passed)"
            if reference_file_identifier:
                problem_name_for_filename = os.path.splitext(os.path.basename(reference_file_identifier))[0]
            else:
                problem_name_for_filename = "local_reference_unknown_name"

        sanitized_ref_name = re.sub(r'[^a-zA-Z0-9_]', '_', problem_name_for_filename)
        json_filename_stem_base = f"run_{run_data['timestamp']}_{sanitized_ref_name}"

        if result.compiled and result.correctness:
            run_data["status"] = "success"
            run_data["compilation_passed"] = True
            run_data["correctness_passed"] = True
            run_data["forward_pass"] = {
                "kernel_exec_time_ms": kernel_exec_time * 1000 if kernel_exec_time is not None else None,
                "ref_exec_eager_time_ms": ref_exec_eager_time,
                "ref_exec_compile_time_ms": ref_exec_compile_time
            }
            if kernel_exec_time is not None and kernel_exec_time > 0:
                _kernel_time_ms = kernel_exec_time * 1000
                if ref_exec_eager_time is not None:
                    _speedup_eager = ref_exec_eager_time / _kernel_time_ms
                    run_data["forward_pass"]["speedup_over_eager"] = float(f"{_speedup_eager:.2f}")
                    run_data["forward_pass"]["beat_eager"] = _speedup_eager > 1.0
                if ref_exec_compile_time is not None:
                    _speedup_compile = ref_exec_compile_time / _kernel_time_ms
                    run_data["forward_pass"]["speedup_over_torch_compile"] = float(f"{_speedup_compile:.2f}")
                    run_data["forward_pass"]["beat_torch_compile"] = _speedup_compile > 1.0
            
            json_filename_stem = f"{json_filename_stem_base}_successful.json"
            relative_json_path = os.path.join("successful", json_filename_stem)
        else: # Either not compiled or not correct
            run_data["status"] = "failure"
            run_data["compilation_passed"] = result.compiled
            run_data["correctness_passed"] = result.correctness # Will be False if not compiled or explicitly False

            error_metadata = result.metadata if hasattr(result, 'metadata') and result.metadata else {}
            if not result.compiled:
                run_data["error_category"] = error_metadata.get("error_category", "unknown_compilation_error")
                run_data["error_message"] = error_metadata.get("compilation_error", error_metadata.get("runtime_error", "No compilation error message provided."))
            elif not result.correctness: # Compiled but not correct
                run_data["error_category"] = error_metadata.get("error_category", "correctness_failure_or_runtime_error_in_eval")
                error_msg = error_metadata.get("runtime_error") # This usually has the specific error for correctness phase
                if not error_msg:
                    error_msg = error_metadata.get("correctness_issue", "Correctness check failed.")
                run_data["error_message"] = error_msg
            else: # Should not be reached if previous conditions are correct
                run_data["error_category"] = "unknown_failure_state_in_json_logic"
                run_data["error_message"] = "Inconsistent state detected during JSON log preparation."

            json_filename_stem = f"{json_filename_stem_base}_failed.json"
            relative_json_path = os.path.join("failed", json_filename_stem)

        response["json_log_content"] = json.dumps(run_data, indent=4)
        response["json_log_filename"] = relative_json_path
        # --- END REVISED JSON LOG PREPARATION ---
            
        return response
        
    except Exception as e:
        # Also capture any exception output
        evaluation_output = captured_output.getvalue()
        evaluation_error = captured_error.getvalue()
        
        # --- BEGIN JSON LOG PREPARATION FOR FAILURE ---
        # Basic info for failure log
        _problem_name_on_fail = "unknown_problem"
        _reference_identifier_on_fail = "unknown_reference"

        if ref_origin == "kernelbench":
            if 'problem_name' in locals() and problem_name: # problem_name from dataset
                _problem_name_on_fail = problem_name
                _reference_identifier_on_fail = f"kernelbench:level_{level}:problem_{problem_id}:{problem_name}"
            else: # Fallback
                _problem_name_on_fail = f"level{level}_problem{problem_id}"
                _reference_identifier_on_fail = f"kernelbench:level_{level}:problem_{problem_id}:unknown_name"
        elif ref_origin == "local":
            if reference_file_identifier:
                _problem_name_on_fail = os.path.splitext(os.path.basename(reference_file_identifier))[0]
                _reference_identifier_on_fail = reference_file_identifier
            else:
                _problem_name_on_fail = "local_ref_unknown_name"
                _reference_identifier_on_fail = "Local Reference (path not passed)"
        
        _kernel_type_on_fail = "unknown"
        if 'kernel_type' in locals() and kernel_type:
            _kernel_type_on_fail = kernel_type
        else:
            # Try to detect if not already set
            try:
                _is_triton_detected = detect_triton_kernel(kernel_src)
                _kernel_type_on_fail = "Triton" if _is_triton_detected else "CUDA"
            except:
                pass # keep as unknown

        run_data_failure = {
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
            "triton_code_path": kernel_file_identifier if kernel_file_identifier else "N/A (ran remotely via Modal - path not passed)",
            "reference_code_identifier": _reference_identifier_on_fail,
            "kernel_type": _kernel_type_on_fail,
            "status": "failure",
            "compilation_passed": False, # Assume False on general exception, specific errors might refine this
            "correctness_passed": False,
            "error_category": type(e).__name__,
            "error_message": str(e),
            "evaluation_output_at_failure": evaluation_output,
            "evaluation_error_at_failure": evaluation_error
        }

        # Update compilation/correctness if some results were obtained before exception
        if 'result' in locals() and result and hasattr(result, 'compiled'):
            run_data_failure["compilation_passed"] = result.compiled
        if 'result' in locals() and result and hasattr(result, 'correctness'):
            run_data_failure["correctness_passed"] = result.correctness
        if 'result' in locals() and result and hasattr(result, 'metadata') and result.metadata:
            if result.metadata.get("error_category"):
                 run_data_failure["error_category"] = result.metadata.get("error_category")
            if result.metadata.get("compilation_error"):
                 run_data_failure["error_message"] = result.metadata.get("compilation_error")
            elif result.metadata.get("runtime_error"):
                 run_data_failure["error_message"] = result.metadata.get("runtime_error")

        sanitized_ref_name_fail = re.sub(r'[^a-zA-Z0-9_]', '_', _problem_name_on_fail)
        json_filename_stem_fail = f"run_{run_data_failure['timestamp']}_{sanitized_ref_name_fail}_failed.json"
        relative_json_path_fail = os.path.join("failed", json_filename_stem_fail)
        # --- END JSON LOG PREPARATION FOR FAILURE ---

        error_response = {
            "status": "failed",
            "error": str(e),
            "error_type": type(e).__name__,
            "evaluation_output": evaluation_output,
            "evaluation_error": evaluation_error,
            "json_log_content": json.dumps(run_data_failure, indent=4),
            "json_log_filename": relative_json_path_fail
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
    gpu: str = "H100"
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
    """
    
    print(f"🚀 Starting Triton evaluation on {gpu}")
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
            kernel_file_identifier=kernel_src_path,
            reference_file_identifier=ref_arch_src_path if ref_origin == "local" else None
        )
        
        print("\n" + "="*60)
        print("📊 MODAL EVALUATION RESULTS") 
        print("="*60)
        
        if result["status"] == "completed":
            print(f"✅ Status: {result['status']}")
            print(f"📋 Problem: {result['problem_name']}")
            print(f"🔍 Kernel Type: {result['kernel_type']}")
            print(f"✅ Compiled: {result['compiled']}")
            print(f"✅ Correctness: {result['correctness']}")
            
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
            
        # --- BEGIN LOCAL JSON LOG SAVING ---
        if result.get("json_log_content") and result.get("json_log_filename"):
            json_content = result["json_log_content"]
            # json_log_filename from remote already includes successful/ or failed/
            relative_json_path = result["json_log_filename"]
            
            output_base_dir = "runs" # Local base directory for logs
            local_json_full_path = os.path.join(output_base_dir, relative_json_path)
            
            # Ensure the local directory structure exists (e.g., runs/successful/ or runs/failed/)
            local_json_dir = os.path.dirname(local_json_full_path)
            os.makedirs(local_json_dir, exist_ok=True)
            
            try:
                with open(local_json_full_path, 'w') as f:
                    f.write(json_content)
                print(f"[INFO] Evaluation results saved locally to {local_json_full_path}")
            except Exception as e:
                print(f"[ERROR] Failed to save JSON log locally: {e}")
        else:
            print("[WARNING] JSON log content or filename not found in Modal results. Skipping local save.")
        # --- END LOCAL JSON LOG SAVING ---
            
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
        gpu=args.gpu
    ) 