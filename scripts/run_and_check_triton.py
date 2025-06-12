import shutil
import torch
import pydra
from pydra import REQUIRED, Config
import os
from datasets import load_dataset

# New imports
import json
import datetime
import re

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
Run a pair of KernelBench format (problem, solution) to check if Triton solution is correct and compute speedup

You will need two files
1. Reference: PyTorch reference (module Model) implementation with init and input shapes
2. Solution: PyTorch solution (module ModelNew) with Triton kernels

The Reference could be either
1. a local file: specify the path to the file
2. a kernelbench problem: specify level and problem id

====================================================
Usage:
1. PyTorch reference is a local file
python3 scripts/run_and_check_triton.py ref_origin=local ref_arch_src_path=src/prompts/model_ex_add.py kernel_src_path=src/prompts/model_new_ex_add_triton.py

2. PyTorch reference is a kernelbench problem
python3 scripts/run_and_check_triton.py ref_origin=kernelbench level=<level> problem_id=<problem_id> kernel_src_path=<path to triton kernel>

3. Auto-detect kernel type (CUDA or Triton)
python3 scripts/run_and_check_triton.py ref_origin=kernelbench level=<level> problem_id=<problem_id> kernel_src_path=<path to kernel> auto_detect=True
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

    # Print results
    kernel_type_str = kernel_eval_result.metadata.get("kernel_type", "CUDA")
    print("="*40)
    print(f"[Eval] {kernel_type_str} kernel eval result: {kernel_eval_result}")
    print("-"*40)
    print(f"[Timing] PyTorch Reference Eager exec time: {ref_exec_eager_time} ms")
    print(f"[Timing] PyTorch Reference torch.compile time: {ref_exec_compile_time} ms")
    print(f"[Timing] Custom {kernel_type_str} Kernel exec time: {kernel_exec_time} ms")
    print("-"*40)   
    
    if kernel_eval_result.correctness:
        print(f"[Speedup] Speedup over eager: {ref_exec_eager_time / kernel_exec_time:.2f}x")
        print(f"[Speedup] Speedup over torch.compile: {ref_exec_compile_time / kernel_exec_time:.2f}x")
    else:
        print("[Speedup] Speedup Not Available as Kernel did not pass correctness")

    print("="*40)

    # --- BEGIN NEW JSON LOGGING CODE ---
    output_base_dir = "runs"
    successful_dir = os.path.join(output_base_dir, "successful")
    failed_dir = os.path.join(output_base_dir, "failed")

    os.makedirs(successful_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)

    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    ref_name_for_filename = "unknown_ref"
    if config.ref_origin == "local":
        if config.ref_arch_src_path:
            ref_name_for_filename = os.path.splitext(os.path.basename(config.ref_arch_src_path))[0]
    elif config.ref_origin == "kernelbench":
        # problem_name is defined earlier in the main function if ref_origin is kernelbench
        # and has been asserted to exist.
        ref_name_for_filename = problem_name 
    
    sanitized_ref_name = re.sub(r'[^a-zA-Z0-9_]', '_', ref_name_for_filename)

    run_data = {
        "timestamp": run_timestamp,
        "triton_code_path": os.path.abspath(config.kernel_src_path)
    }

    if config.ref_origin == "local":
        run_data["reference_code_identifier"] = os.path.abspath(config.ref_arch_src_path)
    elif config.ref_origin == "kernelbench":
        run_data["reference_code_identifier"] = f"kernelbench:level_{config.level}:problem_{config.problem_id}:{problem_name}"

    log_kernel_type = None
    if kernel_eval_result and hasattr(kernel_eval_result, 'metadata') and "kernel_type" in kernel_eval_result.metadata:
        log_kernel_type = kernel_eval_result.metadata["kernel_type"]
    elif config.force_triton:
        log_kernel_type = "triton"
    else: # auto_detect or default CUDA
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

    elif kernel_eval_result.correctness:
        run_data["status"] = "success"
        run_data["compilation_passed"] = kernel_eval_result.compiled
        run_data["correctness_passed"] = True
        
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
            
        json_filename = f"run_{run_timestamp}_{sanitized_ref_name}_successful.json"
        output_path = os.path.join(successful_dir, json_filename)
        
    else: # Failed (either compilation or correctness, but kernel_eval_result exists)
        run_data["status"] = "failure"
        run_data["compilation_passed"] = kernel_eval_result.compiled
        run_data["correctness_passed"] = kernel_eval_result.correctness
        
        error_metadata = kernel_eval_result.metadata if hasattr(kernel_eval_result, 'metadata') else {}

        if not kernel_eval_result.compiled:
            run_data["error_category"] = error_metadata.get("error_category", "unknown_compilation_error")
            run_data["error_message"] = error_metadata.get("compilation_error", error_metadata.get("runtime_error", "No compilation error message provided."))
        elif not kernel_eval_result.correctness:
            run_data["error_category"] = error_metadata.get("error_category", "unknown_runtime_error_or_correctness_issue")
            error_msg = error_metadata.get("runtime_error")
            if not error_msg:
                error_msg = error_metadata.get("correctness_issue", "No specific error message provided.")
            run_data["error_message"] = error_msg
        else: 
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