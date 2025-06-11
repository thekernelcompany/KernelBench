import shutil
import torch
import pydra
from pydra import REQUIRED, Config
import os
from datasets import load_dataset


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import eval as kernel_eval
from src import utils as kernel_utils
from src import device_utils
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

        # Replace with your GPU architecture, e.g. ["Hopper"] for NVIDIA, ["RDNA3"] for AMD
        self.gpu_arch = ["Ada"]  # Will be auto-detected based on available hardware 

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
        platform = device_utils.get_platform()
        if "CUDA error" in str(e) or "HIP error" in str(e): 
            # NOTE: count this as compilation failure as it is not runnable code
            metadata = {
                "gpu_error": f"{platform.upper()} Error: {str(e)}",
                "platform": platform,
                "hardware": device_utils.get_device_name(device=device),
                "device": str(device)
            }
            eval_result = kernel_eval.KernelExecResult(compiled=False, correctness=False, 
                                                metadata=metadata)
            return eval_result
        else:
            metadata = {
                "other_error": f"error: {str(e)}",
                "platform": platform,
                "hardware": device_utils.get_device_name(device=device),
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

    # Start Evaluation - Validate GPU setup
    gpu_status = device_utils.validate_gpu_setup()
    if not gpu_status["valid"]:
        print(f"[ERROR] {gpu_status['message']}")
        print(f"[INFO] Platform detected: {gpu_status['platform']}")
        if gpu_status['platform'] != 'cpu':
            print(f"[INFO] Backend info: {gpu_status['device_info']}")
        return
    
    # Display platform information
    platform_summary = device_utils.get_platform_summary()
    device = device_utils.get_default_device()
    print(f"[INFO] Platform: {platform_summary}")
    print(f"[INFO] PyTorch device: {device} (internal compatibility layer)")
    
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


if __name__ == "__main__":
    main() 