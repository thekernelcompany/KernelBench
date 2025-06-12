import shutil
import torch
import pydra
from pydra import REQUIRED, Config
import os
from datasets import load_dataset

# New imports
import json
import datetime
# os is already imported

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import eval as kernel_eval
from src import utils as kernel_utils
from src.utils import read_file

# Import measure_program_time from generate_baseline_time
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
from generate_baseline_time import measure_program_time

"""
Run a pair of KernelBench format (problem, solution) to check if solution is correct and compute speedup

You will need two files
1. Reference: PyTorch reference (module Model) implementation with init and input shapes
2. Solution: PyTorch solution (module ModelNew) with CUDA/Triton kernels

The Reference could be either
1. a local file: specify the path to the file
2. a kernelbench problem: specify level and problem id

Supports both forward and backward pass evaluation with comprehensive performance comparison.

====================================================
Usage:
1. PyTorch reference is a local file (forward pass only)
python3 scripts/run_and_check_triton.py ref_origin=local ref_arch_src_path=src/prompts/model_ex_add.py kernel_src_path=src/prompts/model_new_ex_add_triton.py

2. PyTorch reference is a kernelbench problem (forward pass only)
python3 scripts/run_and_check_triton.py ref_origin=kernelbench level=<level> problem_id=<problem_id> kernel_src_path=<path to kernel>

3. Auto-detect kernel type (CUDA or Triton)
python3 scripts/run_and_check_triton.py ref_origin=kernelbench level=<level> problem_id=<problem_id> kernel_src_path=<path to kernel> auto_detect=True

4. Test backward pass (gradient computation) - Triton kernels with torch.autograd.Function
python3 scripts/run_and_check_triton.py ref_origin=kernelbench level=<level> problem_id=<problem_id> kernel_src_path=<path to backward kernel> test_backward_pass=True

5. Comprehensive evaluation with backward pass and custom settings
python3 scripts/run_and_check_triton.py ref_origin=kernelbench level=1 problem_id=19 kernel_src_path=src/prompts/model_new_ex_relu_backward_triton.py test_backward_pass=True num_gradient_trials=5 gradient_tolerance=1e-4 verbose=True

Backward Pass Features:
- Compares custom kernel vs PyTorch eager vs torch.compile for both forward and backward passes
- Tests gradient correctness using torch.autograd.gradcheck
- Measures performance for complete forward+backward execution
- Supports memory-optimized evaluation for limited GPU environments
- Provides comprehensive JSON logging with both forward and backward metrics
====================================================

"""

torch.set_printoptions(precision=4, threshold=10)

class ScriptConfig(Config):
    def __init__(self):

        # Problem and Solution definition
        # Input src origin definition
        self.ref_origin = REQUIRED # either local or kernelbench
        # ref_origin is local, specify local file path
        self.ref_arch_src_path = ""
        # ref_origin is kernelbench, specify level and problem id
        self.dataset_name = "ScalingIntelligence/KernelBench"
        self.level = ""
        self.problem_id = ""
        # Solution src definition
        self.kernel_src_path = ""

        # Evaluation mode
        self.auto_detect = True  # Auto-detect kernel type (CUDA vs Triton)
        self.force_triton = False  # Force Triton evaluation even if not detected

        # KernelBench Eval specific
        # number of trials to run for correctness
        self.num_correct_trials = 5
        # number of trials to run for performance
        self.num_perf_trials = 100
        # timeout for each trial
        self.timeout = 300
        # verbose logging
        self.verbose = False
        self.measure_performance = True
        self.build_dir_prefix = "" # if you want to specify a custom build directory
        self.clear_cache = False # TODO

        # Backward pass evaluation
        self.test_backward_pass = False  # Enable backward pass testing
        self.num_gradient_trials = 5  # Number of gradient correctness trials
        self.gradient_tolerance = 1e-4  # Tolerance for gradient checking
        self.measure_backward_performance = True  # Measure backward pass performance

        # Replace with your NVIDIA GPU architecture, e.g. ["Hopper"]
        self.gpu_arch = ["Ada"] 

    def __repr__(self):
        return f"ScriptConfig({self.to_dict()})"

def evaluate_single_sample_src(ref_arch_src: str, kernel_src: str, configs: dict, device: torch.device) -> kernel_eval.KernelExecResult:
    """
    Evaluate a single sample source code against a reference source code
    Supports both CUDA and Triton kernels with auto-detection
    """

    kernel_hash = str(hash(kernel_src))
    build_dir = os.path.join(configs["build_dir_prefix"], "test_build", kernel_hash)
    
    if configs["clear_cache"]: # fresh kernel build
        print(f"[INFO] Clearing cache for build directory: {build_dir}")
        shutil.rmtree(build_dir, ignore_errors=True)
    
    num_correct_trials = configs["num_correct_trials"]
    num_perf_trials = configs["num_perf_trials"]    
    verbose = configs["verbose"]
    measure_performance = configs["measure_performance"]
    auto_detect = configs.get("auto_detect", True)
    force_triton = configs.get("force_triton", False)
    
    try:
        # Choose evaluation function based on kernel type
        if force_triton:
            if verbose:
                print("[INFO] Force using Triton evaluation")
            eval_result = kernel_eval.eval_triton_kernel_against_ref(
                original_model_src=ref_arch_src,
                custom_model_src=kernel_src,
                measure_performance=measure_performance,
                verbose=verbose,
                num_correct_trials=num_correct_trials,
                num_perf_trials=num_perf_trials,
                build_dir=build_dir,
                device=device
            )
        elif auto_detect:
            if verbose:
                print("[INFO] Auto-detecting kernel type")
            eval_result = kernel_eval.eval_kernel_against_ref_auto(
                original_model_src=ref_arch_src,
                custom_model_src=kernel_src,
                measure_performance=measure_performance,
                verbose=verbose,
                num_correct_trials=num_correct_trials,
                num_perf_trials=num_perf_trials,
                build_dir=build_dir,
                device=device
            )
        else:
            # Default to original CUDA evaluation
            if verbose:
                print("[INFO] Using CUDA evaluation")
            eval_result = kernel_eval.eval_kernel_against_ref(
                original_model_src=ref_arch_src,
                custom_model_src=kernel_src,
                measure_performance=measure_performance,
                verbose=verbose,
                num_correct_trials=num_correct_trials,
                num_perf_trials=num_perf_trials,
                build_dir=build_dir,
                device=device
            )
        return eval_result
    except Exception as e:
        print(f"[WARNING] Last level catch: Some issue evaluating for kernel: {e} ")
        if "CUDA error" in str(e): 
            # NOTE: count this as compilation failure as it is not runnable code
            metadata = {"cuda_error": f"CUDA Error: {str(e)}",
                        "hardware": torch.cuda.get_device_name(device=device),
                        "device": str(device)
                        }
            eval_result = kernel_eval.KernelExecResult(compiled=False, correctness=False, 
                                                metadata=metadata)
            return eval_result
        else:
            metadata = {"other_error": f"error: {str(e)}",
                        "hardware": torch.cuda.get_device_name(device=device),
                        "device": str(device)
                        }
            eval_result = kernel_eval.KernelExecResult(compiled=False, correctness=False, 
                                                metadata=metadata)
            return eval_result


def evaluate_backward_pass_src(ref_arch_src: str, kernel_src: str, configs: dict, device: torch.device) -> kernel_eval.KernelExecResult:
    """
    Evaluate backward pass (gradient computation) for a kernel against a reference
    Supports both CUDA and Triton kernels with auto-detection
    """
    kernel_hash = str(hash(kernel_src))
    build_dir = os.path.join(configs["build_dir_prefix"], "test_build", kernel_hash)
    
    if configs["clear_cache"]: # fresh kernel build
        print(f"[INFO] Clearing cache for backward pass build directory: {build_dir}")
        shutil.rmtree(build_dir, ignore_errors=True)
    
    num_correct_trials = configs["num_correct_trials"]
    num_gradient_trials = configs.get("num_gradient_trials", 3)
    num_perf_trials = configs["num_perf_trials"]    
    gradient_tolerance = configs.get("gradient_tolerance", 1e-4)
    verbose = configs["verbose"]
    measure_performance = configs.get("measure_backward_performance", True)
    auto_detect = configs.get("auto_detect", True)
    force_triton = configs.get("force_triton", False)
    
    try:
        # Choose evaluation function based on kernel type
        if force_triton:
            if verbose:
                print("[INFO] Force using Triton backward pass evaluation")
            eval_result = kernel_eval.eval_triton_backward_pass(
                original_model_src=ref_arch_src,
                custom_model_src=kernel_src,
                seed_num=42,
                num_correct_trials=num_correct_trials,
                num_gradient_trials=num_gradient_trials,
                num_perf_trials=num_perf_trials,
                gradient_tolerance=gradient_tolerance,
                verbose=verbose,
                measure_performance=measure_performance,
                build_dir=build_dir,
                device=device
            )
        elif auto_detect:
            if verbose:
                print("[INFO] Auto-detecting kernel type for backward pass")
            eval_result = kernel_eval.eval_kernel_backward_pass_auto(
                original_model_src=ref_arch_src,
                custom_model_src=kernel_src,
                seed_num=42,
                num_correct_trials=num_correct_trials,
                num_gradient_trials=num_gradient_trials,
                num_perf_trials=num_perf_trials,
                gradient_tolerance=gradient_tolerance,
                verbose=verbose,
                measure_performance=measure_performance,
                build_dir=build_dir,
                device=device
            )
        else:
            # Default to CUDA evaluation (when implemented)
            raise NotImplementedError(
                "CUDA backward pass evaluation not yet implemented. "
                "Use auto_detect=True or force_triton=True for backward pass testing."
            )
        return eval_result
    except Exception as e:
        print(f"[WARNING] Last level catch for backward pass: Some issue evaluating kernel: {e} ")
        metadata = {
            "backward_error": f"Backward pass error: {str(e)}",
            "hardware": torch.cuda.get_device_name(device=device),
            "device": str(device),
            "evaluation_type": "backward_pass"
        }
        eval_result = kernel_eval.KernelExecResult(compiled=False, correctness=False, 
                                            metadata=metadata)
        return eval_result


def measure_backward_pass_time(ref_arch_name: str, ref_arch_src: str, num_trials: int = 10, 
                             use_torch_compile: bool = False, device: torch.device = None) -> dict:
    """
    Measure backward pass execution time for reference PyTorch models
    """
    try:
        # Clear GPU memory
        if device and device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Load model dynamically
        context = {}
        exec(ref_arch_src, context)
        
        if "Model" not in context:
            return {"error": "Model class not found in reference code"}
        
        if "get_init_inputs" not in context:
            return {"error": "get_init_inputs function not found in reference code"}
            
        if "get_inputs" not in context:
            return {"error": "get_inputs function not found in reference code"}
        
        Model = context["Model"]
        get_init_inputs = context["get_init_inputs"]
        get_inputs = context["get_inputs"]
        
        # Initialize model
        with torch.no_grad():
            init_inputs = get_init_inputs()
            if device:
                init_inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs]
            model = Model(*init_inputs)
            if device:
                model = model.cuda(device=device)
        
        # Apply torch.compile if requested
        if use_torch_compile:
            model = torch.compile(model, backend="inductor")
        
        times = []
        
        for trial in range(num_trials):
            # Generate inputs with gradients enabled
            inputs = get_inputs()
            if device:
                inputs = [x.cuda(device=device).requires_grad_(True) if isinstance(x, torch.Tensor) else x for x in inputs]
            else:
                inputs = [x.requires_grad_(True) if isinstance(x, torch.Tensor) else x for x in inputs]
            
            # Warm up
            if trial == 0:
                for _ in range(3):
                    outputs = model(*inputs)
                    if isinstance(outputs, torch.Tensor):
                        loss = outputs.sum()
                    else:
                        loss = sum(o.sum() for o in outputs if isinstance(o, torch.Tensor))
                    loss.backward()
                    if device and device.type == 'cuda':
                        torch.cuda.synchronize(device=device)
                    # Clear gradients
                    for inp in inputs:
                        if isinstance(inp, torch.Tensor) and inp.grad is not None:
                            inp.grad.zero_()
            
            # Actual timing
            if device and device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                outputs = model(*inputs)
                if isinstance(outputs, torch.Tensor):
                    loss = outputs.sum()
                else:
                    loss = sum(o.sum() for o in outputs if isinstance(o, torch.Tensor))
                loss.backward()
                end_event.record()
                
                torch.cuda.synchronize(device=device)
                elapsed_time = start_event.elapsed_time(end_event)
                times.append(elapsed_time)
            else:
                import time
                start_time = time.time()
                outputs = model(*inputs)
                if isinstance(outputs, torch.Tensor):
                    loss = outputs.sum()
                else:
                    loss = sum(o.sum() for o in outputs if isinstance(o, torch.Tensor))
                loss.backward()
                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000  # Convert to ms
                times.append(elapsed_time)
            
            # Clear gradients for next iteration
            for inp in inputs:
                if isinstance(inp, torch.Tensor) and inp.grad is not None:
                    inp.grad.zero_()
        
        # Calculate statistics
        mean_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
        
        result = {
            "mean": mean_time,
            "min": min_time,
            "max": max_time,
            "std": std_time,
            "num_trials": num_trials,
            "use_torch_compile": use_torch_compile,
            "ref_arch_name": ref_arch_name
        }
        
        if device:
            result["device"] = str(device)
            result["hardware"] = torch.cuda.get_device_name(device=device)
        
        return result
        
    except Exception as e:
        return {"error": f"Error measuring backward pass time: {str(e)}"}


@pydra.main(base=ScriptConfig)
def main(config: ScriptConfig):

    print("Running with config", config)

    # Fetch reference and kernel code

    assert config.ref_origin == "local" or config.ref_origin == "kernelbench", "ref_origin must be either local or kernelbench"
    assert config.kernel_src_path != "", "kernel_src_path is required"  
    
    if config.ref_origin == "local":
        assert config.ref_arch_src_path != "", "ref_arch_src_path is required"
        ref_arch_src = read_file(config.ref_arch_src_path)
    elif config.ref_origin == "kernelbench":
        assert config.dataset_name != "", "dataset_name is required"
        assert config.level != "", "level is required"
        assert config.problem_id != "", "problem_id is required"

        # for now use the HuggingFace dataset
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]

        curr_problem_row = curr_level_dataset.filter(lambda x: x["problem_id"] == config.problem_id)
        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]

        problem_number = int(problem_name.split("_")[0])
        assert problem_number == config.problem_id, f"Problem number in filename ({problem_number}) does not match config problem_id ({config.problem_id})"

        print(f"Fetched problem {config.problem_id} from KernelBench level {config.level}: {problem_name}")


    else:
        raise ValueError("Invalid ref_origin")
    
    kernel_src = read_file(config.kernel_src_path)

    # Detect kernel type if auto_detect is enabled
    if config.auto_detect:
        is_triton = kernel_eval.detect_triton_kernel(kernel_src)
        kernel_type = "Triton" if is_triton else "CUDA"
        print(f"[INFO] Auto-detected kernel type: {kernel_type}")

    # Start Evaluation
    device = torch.device("cuda:0") # default device
    kernel_utils.set_gpu_arch(config.gpu_arch)

    print("[INFO] Evaluating kernel against reference code")
    # Evaluate kernel against reference code
    kernel_eval_result = evaluate_single_sample_src(
        ref_arch_src=ref_arch_src,
        kernel_src=kernel_src,
        configs=config.to_dict(),
        device=device
    )
    kernel_exec_time = kernel_eval_result.runtime

    # Measure baseline time
    print("[INFO] Measuring reference program time")
    # Default using PyTorch Eager here
    ref_time_eager_result = measure_program_time(ref_arch_name="Reference Program", 
                                                ref_arch_src=ref_arch_src, 
                                                num_trials=config.num_perf_trials,
                                                use_torch_compile=False,
                                                device=device)
    ref_exec_eager_time = ref_time_eager_result.get("mean", None)

    # Measure Torch Compile time
    ref_time_compile_result = measure_program_time(ref_arch_name="Reference Program", 
                                                ref_arch_src=ref_arch_src, 
                                                num_trials=config.num_perf_trials,
                                                use_torch_compile=True,
                                                torch_compile_backend="inductor",
                                                torch_compile_options="default",
                                                device=device)
    ref_exec_compile_time = ref_time_compile_result.get("mean", None)

    # Backward pass evaluation (if enabled)
    backward_eval_result = None
    ref_backward_eager_time = None
    ref_backward_compile_time = None
    kernel_backward_exec_time = None

    if config.test_backward_pass:
        print("\n[INFO] Starting backward pass evaluation")
        
        # Evaluate custom kernel backward pass
        print("[INFO] Evaluating custom kernel backward pass")
        backward_eval_result = evaluate_backward_pass_src(
            ref_arch_src=ref_arch_src,
            kernel_src=kernel_src,
            configs=config.to_dict(),
            device=device
        )
        kernel_backward_exec_time = backward_eval_result.runtime if backward_eval_result else None

        # Measure reference backward pass times
        if config.measure_backward_performance:
            print("[INFO] Measuring reference backward pass times")
            
            # Eager backward pass
            ref_backward_eager_result = measure_backward_pass_time(
                ref_arch_name="Reference Backward (Eager)", 
                ref_arch_src=ref_arch_src, 
                num_trials=config.num_perf_trials,
                use_torch_compile=False,
                device=device
            )
            ref_backward_eager_time = ref_backward_eager_result.get("mean", None)

            # Torch compile backward pass
            ref_backward_compile_result = measure_backward_pass_time(
                ref_arch_name="Reference Backward (Compile)", 
                ref_arch_src=ref_arch_src, 
                num_trials=config.num_perf_trials,
                use_torch_compile=True,
                device=device
            )
            ref_backward_compile_time = ref_backward_compile_result.get("mean", None)

    # Print results
    kernel_type_str = kernel_eval_result.metadata.get("kernel_type", "CUDA")
    print("="*60)
    print(f"[FORWARD PASS RESULTS]")
    print("="*60)
    print(f"[Eval] {kernel_type_str} kernel eval result: {kernel_eval_result}")
    print("-"*60)
    print(f"[Timing] PyTorch Reference Eager exec time: {ref_exec_eager_time} ms")
    print(f"[Timing] PyTorch Reference torch.compile time: {ref_exec_compile_time} ms")
    print(f"[Timing] Custom {kernel_type_str} Kernel exec time: {kernel_exec_time} ms")
    print("-"*60)   
    
    if kernel_eval_result.correctness:
        print(f"[Speedup] Forward Speedup over eager: {ref_exec_eager_time / kernel_exec_time:.2f}x")
        print(f"[Speedup] Forward Speedup over torch.compile: {ref_exec_compile_time / kernel_exec_time:.2f}x")
    else:
        print("[Speedup] Forward Speedup Not Available - Kernel did not pass correctness")

    # Print backward pass results if tested
    if config.test_backward_pass:
        print("\n" + "="*60)
        print(f"[BACKWARD PASS RESULTS]")
        print("="*60)
        if backward_eval_result:
            print(f"[Eval] {kernel_type_str} backward pass result: {backward_eval_result}")
            
            # Print backward pass correctness details
            if backward_eval_result.metadata:
                if 'forward_correctness_trials' in backward_eval_result.metadata:
                    print(f"[Correctness] Forward trials: {backward_eval_result.metadata['forward_correctness_trials']}")
                if 'gradient_correctness' in backward_eval_result.metadata:
                    print(f"[Correctness] Gradient trials: {backward_eval_result.metadata['gradient_correctness']}")
                if 'backward_pass_correctness' in backward_eval_result.metadata:
                    overall_status = "✅ PASS" if backward_eval_result.metadata['backward_pass_correctness'] else "❌ FAIL"
                    print(f"[Correctness] Overall backward pass: {overall_status}")
        else:
            print("[Eval] Backward pass evaluation failed")
        
        print("-"*60)
        
        if config.measure_backward_performance:
            print(f"[Timing] PyTorch Reference Backward Eager time: {ref_backward_eager_time} ms")
            print(f"[Timing] PyTorch Reference Backward torch.compile time: {ref_backward_compile_time} ms")
            print(f"[Timing] Custom {kernel_type_str} Backward Kernel time: {kernel_backward_exec_time} ms")
            print("-"*60)
            
            if (backward_eval_result and backward_eval_result.correctness and 
                kernel_backward_exec_time is not None and kernel_backward_exec_time > 0):
                if ref_backward_eager_time:
                    backward_speedup_eager = ref_backward_eager_time / kernel_backward_exec_time
                    print(f"[Speedup] Backward Speedup over eager: {backward_speedup_eager:.2f}x")
                if ref_backward_compile_time:
                    backward_speedup_compile = ref_backward_compile_time / kernel_backward_exec_time
                    print(f"[Speedup] Backward Speedup over torch.compile: {backward_speedup_compile:.2f}x")
            else:
                print("[Speedup] Backward Speedup Not Available - Kernel did not pass backward correctness")

    print("="*60)

    # --- BEGIN NEW JSON LOGGING CODE ---
    output_base_dir = "runs"
    successful_dir = os.path.join(output_base_dir, "successful")
    failed_dir = os.path.join(output_base_dir, "failed")

    # Create directories if they don't exist
    # Ensure this is relative to the script's directory or a defined workspace root if necessary
    # For simplicity, let's assume it's relative to where the script is run.
    # If KernelBench has a defined results path, that might be better.
    # For now, creating 'runs' where the script is executed.
    os.makedirs(successful_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)

    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # --- MODIFICATION FOR DESCRIPTIVE FILENAME ---
    ref_name_for_filename = "unknown_ref"
    if config.ref_origin == "local":
        if config.ref_arch_src_path:
            ref_name_for_filename = os.path.splitext(os.path.basename(config.ref_arch_src_path))[0]
    elif config.ref_origin == "kernelbench":
        # problem_name is defined earlier in the main function if ref_origin is kernelbench
        if 'problem_name' in locals() and problem_name:
            ref_name_for_filename = problem_name
        else: # Fallback if problem_name is not available for some reason
            ref_name_for_filename = f"level{config.level}_problem{config.problem_id}"
    
    # Sanitize the reference name for use in a filename
    # Replace non-alphanumeric characters (and not underscore) with underscore
    import re
    sanitized_ref_name = re.sub(r'[^a-zA-Z0-9_]', '_', ref_name_for_filename)
    # --- END MODIFICATION FOR DESCRIPTIVE FILENAME ---

    run_data = {
        "timestamp": run_timestamp,
        "triton_code_path": os.path.abspath(config.kernel_src_path),
        "test_backward_pass": config.test_backward_pass
    }

    # Determine reference path or identifier
    if config.ref_origin == "local":
        run_data["reference_code_identifier"] = os.path.abspath(config.ref_arch_src_path)
    elif config.ref_origin == "kernelbench":
        # problem_name is defined earlier in the main function for this case
        run_data["reference_code_identifier"] = f"kernelbench:level_{config.level}:problem_{config.problem_id}:{problem_name}"

    # Determine kernel_type for logging
    log_kernel_type = None
    if kernel_eval_result and "kernel_type" in kernel_eval_result.metadata:
        log_kernel_type = kernel_eval_result.metadata["kernel_type"]
    elif config.force_triton:
        log_kernel_type = "triton"
    else:
        # kernel_src is defined earlier in main and is the source code string
        is_triton_detected_for_log = kernel_eval.detect_triton_kernel(kernel_src)
        log_kernel_type = "triton" if is_triton_detected_for_log else "CUDA"
    run_data["kernel_type"] = log_kernel_type

    output_path = None

    if kernel_eval_result is None:
        run_data["status"] = "critical_eval_failure"
        run_data["error_message"] = "Evaluation function returned None, possibly due to file system lock/permission issues during compilation."
        run_data["compilation_passed"] = False
        run_data["correctness_passed"] = False
        
        json_filename = f"run_{run_timestamp}_{sanitized_ref_name}_failed.json"
        output_path = os.path.join(failed_dir, json_filename)
        
    elif kernel_eval_result.correctness: # Successfully compiled and correct
        run_data["status"] = "success"
        run_data["compilation_passed"] = kernel_eval_result.compiled # Should be True
        run_data["correctness_passed"] = True
        
        # Forward pass results
        run_data["forward_pass"] = {
            "kernel_exec_time_ms": kernel_exec_time,
            "ref_exec_eager_time_ms": ref_exec_eager_time,
            "ref_exec_compile_time_ms": ref_exec_compile_time
        }
        
        if kernel_exec_time is not None and kernel_exec_time > 0:
            if ref_exec_eager_time is not None:
                speedup_eager = ref_exec_eager_time / kernel_exec_time
                run_data["forward_pass"]["speedup_over_eager"] = float(f"{speedup_eager:.2f}")
                run_data["forward_pass"]["beat_eager"] = speedup_eager > 1.0
            else:
                run_data["forward_pass"]["speedup_over_eager"] = None
                run_data["forward_pass"]["beat_eager"] = False

            if ref_exec_compile_time is not None:
                speedup_compile = ref_exec_compile_time / kernel_exec_time
                run_data["forward_pass"]["speedup_over_torch_compile"] = float(f"{speedup_compile:.2f}")
                run_data["forward_pass"]["beat_torch_compile"] = speedup_compile > 1.0
            else:
                run_data["forward_pass"]["speedup_over_torch_compile"] = None
                run_data["forward_pass"]["beat_torch_compile"] = False
        else:
            run_data["forward_pass"]["speedup_over_eager"] = None
            run_data["forward_pass"]["beat_eager"] = False
            run_data["forward_pass"]["speedup_over_torch_compile"] = None
            run_data["forward_pass"]["beat_torch_compile"] = False
        
        # Backward pass results (if tested)
        if config.test_backward_pass:
            run_data["backward_pass"] = {
                "tested": True,
                "passed": backward_eval_result.correctness if backward_eval_result else False
            }
            
            if backward_eval_result:
                run_data["backward_pass"]["compilation_passed"] = backward_eval_result.compiled
                run_data["backward_pass"]["gradient_correctness"] = backward_eval_result.metadata.get("gradient_correctness", "N/A")
                run_data["backward_pass"]["backward_pass_correctness"] = backward_eval_result.metadata.get("backward_pass_correctness", False)
                
                # Backward pass timing results
                if config.measure_backward_performance:
                    run_data["backward_pass"]["kernel_backward_exec_time_ms"] = kernel_backward_exec_time
                    run_data["backward_pass"]["ref_backward_eager_time_ms"] = ref_backward_eager_time
                    run_data["backward_pass"]["ref_backward_compile_time_ms"] = ref_backward_compile_time
                    
                    # Backward pass speedups
                    if (kernel_backward_exec_time is not None and kernel_backward_exec_time > 0 and 
                        backward_eval_result.correctness):
                        if ref_backward_eager_time:
                            backward_speedup_eager = ref_backward_eager_time / kernel_backward_exec_time
                            run_data["backward_pass"]["speedup_over_eager"] = float(f"{backward_speedup_eager:.2f}")
                            run_data["backward_pass"]["beat_eager"] = backward_speedup_eager > 1.0
                        else:
                            run_data["backward_pass"]["speedup_over_eager"] = None
                            run_data["backward_pass"]["beat_eager"] = False
                        
                        if ref_backward_compile_time:
                            backward_speedup_compile = ref_backward_compile_time / kernel_backward_exec_time
                            run_data["backward_pass"]["speedup_over_torch_compile"] = float(f"{backward_speedup_compile:.2f}")
                            run_data["backward_pass"]["beat_torch_compile"] = backward_speedup_compile > 1.0
                        else:
                            run_data["backward_pass"]["speedup_over_torch_compile"] = None
                            run_data["backward_pass"]["beat_torch_compile"] = False
                    else:
                        run_data["backward_pass"]["speedup_over_eager"] = None
                        run_data["backward_pass"]["beat_eager"] = False
                        run_data["backward_pass"]["speedup_over_torch_compile"] = None
                        run_data["backward_pass"]["beat_torch_compile"] = False
                        
                # Include GPU memory info if available
                if backward_eval_result.metadata.get("gpu_memory_info"):
                    run_data["backward_pass"]["gpu_memory_info"] = backward_eval_result.metadata["gpu_memory_info"]
            else:
                run_data["backward_pass"]["error"] = "Backward pass evaluation returned None"
        else:
            run_data["backward_pass"] = {"tested": False}
            
        json_filename = f"run_{run_timestamp}_{sanitized_ref_name}_successful.json"
        output_path = os.path.join(successful_dir, json_filename)
        
    else: # Failed (either compilation or correctness, but kernel_eval_result exists)
        run_data["status"] = "failure"
        run_data["compilation_passed"] = kernel_eval_result.compiled
        run_data["correctness_passed"] = kernel_eval_result.correctness # Should be False
        
        if not kernel_eval_result.compiled:
            run_data["error_category"] = kernel_eval_result.metadata.get("error_category", "unknown_compilation_error")
            run_data["error_message"] = kernel_eval_result.metadata.get("compilation_error", "No compilation error message provided.")
        elif not kernel_eval_result.correctness: # Compiled but not correct or runtime error during correctness
            run_data["error_category"] = kernel_eval_result.metadata.get("error_category", "unknown_runtime_error_or_correctness_issue")
            # Prefer runtime_error if present, else correctness_issue
            error_msg = kernel_eval_result.metadata.get("runtime_error")
            if not error_msg:
                error_msg = kernel_eval_result.metadata.get("correctness_issue", "No specific error message provided.")
            run_data["error_message"] = error_msg
        else: # Should not happen if kernel_eval_result.correctness is False
            run_data["error_category"] = "unknown_failure_state"
            run_data["error_message"] = "Inconsistent state: Correctness is false but no specific error captured."

        json_filename = f"run_{run_timestamp}_{sanitized_ref_name}_failed.json"
        output_path = os.path.join(failed_dir, json_filename)

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(run_data, f, indent=4)
        print(f"[INFO] Evaluation results saved to {output_path}")
    else:
        print("[ERROR] Could not determine output path for JSON log.")

    # --- END NEW JSON LOGGING CODE ---

if __name__ == "__main__":
    main() 