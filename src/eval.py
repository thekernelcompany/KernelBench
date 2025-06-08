"""
Helpers for Evaluations
"""

import requests
import torch
import torch.nn as nn
import os, subprocess
from pydantic import BaseModel
import numpy as np
import random
import json
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import sys
import os

# Fix import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import utils

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")


def fetch_kernel_from_database(
    run_name: str, problem_id: int, sample_id: int, server_url: str
):
    """
    Intenral to us with our django database
    Return a dict with kernel hash, kernel code, problem_id
    """
    response = requests.get(
        f"{server_url}/get_kernel_by_run_problem_sample/{run_name}/{problem_id}/{sample_id}",
        json={"run_name": run_name, "problem_id": problem_id, "sample_id": sample_id},
    )
    assert response.status_code == 200
    response_json = response.json()
    assert str(response_json["problem_id"]) == str(problem_id)
    return response_json


def fetch_ref_arch_from_problem_id(problem_id, problems, with_name=False) -> str:
    """
    Fetches the reference architecture in string for a given problem_id
    """
    if isinstance(problem_id, str):
        problem_id = int(problem_id)

    problem_path = problems[problem_id]

    # problem_path = os.path.join(REPO_ROOT_PATH, problem)
    if not os.path.exists(problem_path):
        raise FileNotFoundError(f"Problem file at {problem_path} does not exist.")

    ref_arch = utils.read_file(problem_path)
    if not with_name:
        return ref_arch
    else:
        return (problem_path, ref_arch)


def fetch_ref_arch_from_level_problem_id(level, problem_id, with_name=False):
    PROBLEM_DIR = os.path.join(KERNEL_BENCH_PATH, "level" + str(level))
    dataset = utils.construct_problem_dataset_from_problem_dir(PROBLEM_DIR)
    return fetch_ref_arch_from_problem_id(problem_id, dataset, with_name)


def set_seed(seed: int):
    torch.manual_seed(seed)
    # NOTE: this only sets on current cuda device
    torch.cuda.manual_seed(seed)


class KernelExecResult(BaseModel):
    """
    Single Kernel Execution
    """

    compiled: bool = False
    correctness: bool = False
    metadata: dict = {}
    runtime: float = -1.0  # in us, only recorded if we decide to measure performance
    runtime_stats: dict = {}  # only recorded if we decide to measure performance


def load_original_model_and_inputs(
    model_original_src: str, context: dict
) -> tuple[nn.Module, callable, callable]:
    """
    Load class from original NN.module pytorch code
    this is pytorch reference and we feed that to model to see if there will be any improvement
    """

    try:
        compile(model_original_src, "<string>", "exec")
    except SyntaxError as e:
        print(f"Syntax Error in original code {e}")
        return None

    try:
        exec(model_original_src, context)  # expose to current namespace
    except Exception as e:
        print(f"Error in executing original code {e}")
        return None

    # these should be defined in the original model code and present in the context
    get_init_inputs_fn = context.get("get_init_inputs")
    get_inputs_fn = context.get("get_inputs")
    Model = context.get("Model")
    return (Model, get_init_inputs_fn, get_inputs_fn)


def detect_triton_kernel(model_src: str) -> bool:
    """
    Detect if the model source code uses Triton kernels
    """
    triton_indicators = [
        "import triton",
        "from triton",
        "@triton.jit",
        "triton.language",
        "tl.program_id",
        "tl.load",
        "tl.store"
    ]
    return any(indicator in model_src for indicator in triton_indicators)


def _cleanup_triton_cache():
    """Helper function to cleanup Triton cache"""
    import shutil
    
    # Triton cache locations
    triton_cache_paths = [
        os.path.join(os.path.expanduser("~"), ".triton", "cache"),
        "/tmp/triton"  # Alternative cache location
    ]
    
    for cache_path in triton_cache_paths:
        if os.path.exists(cache_path):
            try:
                shutil.rmtree(cache_path)
            except Exception as e:
                # Don't fail if we can't clean cache
                pass


def load_custom_model(
    model_custom_src: str, context: dict, build_directory: str = None
) -> nn.Module:
    """
    Load class from custom NN.module pytorch code
    this is the code output by LLM with calls to custom cuda kernels
    """
    if build_directory:
        context["BUILD_DIRECTORY"] = build_directory
        # Add import at the start of the source code
        model_custom_src = (
            "import os\n" f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_directory}'\n"
        ) + model_custom_src

    try:
        compile(model_custom_src, "<string>", "exec")
        exec(model_custom_src, context)
        # DANGER: need to delete refernece from global namespace
    except SyntaxError as e:
        print(f"Syntax Error in custom generated code or Compilation Error {e}")
        return None

    ModelNew = context.get("ModelNew")
    return ModelNew


def load_custom_model_triton(
    model_custom_src: str, context: dict, build_directory: str = None
) -> nn.Module:
    """
    Load class from custom NN.module pytorch code with Triton kernels
    This is the Triton equivalent of load_custom_model with the same robustness
    
    This function writes the source code to a persistent file and imports it
    to work around Triton's inspect.getsourcelines() requirement
    """
    import tempfile
    import importlib.util
    import sys
    import hashlib
    
    # Set build directory and environment variables (matching CUDA behavior)
    if build_directory:
        context["BUILD_DIRECTORY"] = build_directory
        # Set Triton cache directory
        triton_cache_dir = os.path.join(build_directory, "triton_cache")
        os.makedirs(triton_cache_dir, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = triton_cache_dir
        
        # Use build directory for the temporary module (like CUDA uses TORCH_EXTENSIONS_DIR)
        temp_dir = build_directory
        
        # Add environment setup to source code (matching CUDA pattern)
        model_custom_src = (
            "import os\n"
            f"os.environ['TRITON_CACHE_DIR'] = '{triton_cache_dir}'\n"
        ) + model_custom_src
    else:
        # Use system temp directory
        temp_dir = tempfile.gettempdir()

    try:
        # First, verify the code compiles (matching CUDA behavior)
        compile(model_custom_src, "<string>", "exec")
        
        # Create a hash-based filename to avoid conflicts (like CUDA kernel hashing)
        code_hash = hashlib.md5(model_custom_src.encode()).hexdigest()
        module_filename = f"triton_kernel_{code_hash}.py"
        module_path = os.path.join(temp_dir, module_filename)
        
        # Ensure the directory exists
        os.makedirs(temp_dir, exist_ok=True)
        
        # Write the source code to a persistent file
        try:
            with open(module_path, 'w', encoding='utf-8') as f:
                f.write(model_custom_src)
        except (IOError, OSError) as e:
            # Handle file write errors (like CUDA handles directory/lock errors)
            if "lock" in str(e).lower() or "permission" in str(e).lower():
                raise RuntimeError(f"File lock or permission error: {e}")
            raise e
        
        module_name = f"triton_kernel_module_{code_hash}"
        
        try:
            # Import the module from the file
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create module spec for {module_path}")
                
            module = importlib.util.module_from_spec(spec)
            
            # Add to sys.modules to make it findable by inspect (before execution)
            sys.modules[module_name] = module
            
            # Store cleanup info in context (before execution in case it fails)
            context["_triton_module_path"] = module_path
            context["_triton_module_name"] = module_name
            
            # Execute the module
            spec.loader.exec_module(module)
            
            # Extract ModelNew from the module (matching CUDA error handling)
            ModelNew = getattr(module, "ModelNew", None)
            if ModelNew is None:
                raise AttributeError("ModelNew class not found in Triton kernel code")
                
            # Update context with all module attributes for compatibility
            for attr_name in dir(module):
                if not attr_name.startswith('_'):
                    context[attr_name] = getattr(module, attr_name)
            
            return ModelNew
            
        except Exception as e:
            # Clean up on error (more comprehensive like CUDA)
            cleanup_error = None
            try:
                if module_name in sys.modules:
                    del sys.modules[module_name]
            except Exception as cleanup_e:
                cleanup_error = cleanup_e
                
            try:
                if os.path.exists(module_path):
                    os.unlink(module_path)
            except Exception as cleanup_e:
                if cleanup_error is None:
                    cleanup_error = cleanup_e
            
            # Remove from context if we added it
            context.pop("_triton_module_path", None)
            context.pop("_triton_module_name", None)
            
            # Re-raise original error (not cleanup error)
            raise e
                
    except SyntaxError as e:
        print(f"Syntax Error in Triton kernel code or Compilation Error {e}")
        return None
    except ImportError as e:
        print(f"Import Error (check if triton is installed): {e}")
        return None
    except Exception as e:
        print(f"Error loading Triton kernel: {e}")
        return None


def _cleanup_cuda_extensions():
    """Helper function to cleanup compiled CUDA extensions"""
    # SIMON NOTE: is this necessary?
    import shutil

    torch_extensions_path = os.path.join(
        os.path.expanduser("~"), ".cache", "torch_extensions"
    )
    if os.path.exists(torch_extensions_path):
        shutil.rmtree(torch_extensions_path)


def graceful_eval_cleanup(curr_context: dict, device: torch.device):
    """
    Clean up env, gpu cache, and compiled CUDA extensions after evaluation
    """  # delete ran-specific function definitions before next eval run
    del curr_context
    # Clear CUDA cache and reset GPU state
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

        # does this help?
        torch.cuda.reset_peak_memory_stats(device=device)

        torch.cuda.synchronize(
            device=device
        )  # Wait for all CUDA operations to complete

    # _cleanup_cuda_extensions() # SIMON NOTE: is this necessary?


def build_compile_cache_triton(
    custom_model_src: str,
    verbose: bool = False,
    build_dir: os.PathLike = None,
) -> tuple[bool, str, str]:
    """
    Pre-warm Triton kernels by doing a test compilation
    This is the Triton equivalent of build_compile_cache with the same robustness
    
    Triton kernels are JIT compiled on first use, so we trigger compilation here
    Should be able to run on CPUs to do this massively in parallel
    
    Returns:
        tuple[bool, str, str]: whether compilation is successful, stdout content as string, error message
    """
    context = {}
    stdout_buffer = StringIO()

    if verbose:
        print("[Compilation] Pre-warming Triton kernels")

    try:
        # Set Triton environment variables (equivalent to CUDA's TORCH_USE_CUDA_DSA)
        os.environ["TRITON_PRINT_AUTOTUNING"] = "0"  # reduce noise during compilation
        
        # Set Triton cache directory if build_dir is provided
        if build_dir:
            triton_cache_dir = os.path.join(build_dir, "triton_cache")
            os.makedirs(triton_cache_dir, exist_ok=True)
            os.environ["TRITON_CACHE_DIR"] = triton_cache_dir

        # Capture stdout during compilation (matching CUDA pattern)
        with redirect_stdout(stdout_buffer), redirect_stderr(stdout_buffer):
            ModelNew = load_custom_model_triton(custom_model_src, context, build_dir)
            
            # Try to instantiate to trigger any immediate compilation errors
            if ModelNew:
                try:
                    test_model = ModelNew()
                    # Successful instantiation means basic compilation worked
                    if verbose:
                        print(f"[Compilation] Triton model instantiation successful")
                except Exception as e:
                    if verbose:
                        print(f"[Compilation] Model instantiation failed (may need specific inputs): {e}")
                    # This is not necessarily a compilation failure for Triton kernels
                    # that require specific input shapes for JIT compilation
            else:
                raise RuntimeError("ModelNew class not found after compilation")

        if verbose:
            print(f"[Compilation] Triton kernel pre-warming successful, saved cache at: {build_dir}")
            
    except Exception as e:
        error_msg = f"Failed to compile Triton kernel. Unable to cache, \nError: {e}"
        print(f"[Compilation] {error_msg}")
        return False, stdout_buffer.getvalue(), str(e)

    return True, stdout_buffer.getvalue(), None


def build_compile_cache_auto(
    custom_model_src: str,
    verbose: bool = False,
    build_dir: os.PathLike = None,
) -> tuple[bool, str, str]:
    """
    Auto-detect kernel type and use appropriate build cache function
    """
    if detect_triton_kernel(custom_model_src):
        if verbose:
            print("[Compilation] Detected Triton kernel, using Triton compilation")
        return build_compile_cache_triton(custom_model_src, verbose, build_dir)
    else:
        if verbose:
            print("[Compilation] Detected CUDA kernel, using CUDA compilation")
        return build_compile_cache(custom_model_src, verbose, build_dir)


def build_compile_cache_legacy(
    custom_model_src: str,
    verbose: bool = False,
    build_dir: os.PathLike = None,
) -> tuple[bool, str, str]:
    """
    Try to build the compiled cuda code for sample and store in the cache directory
    Should be able to run on CPUs to do this massively in parallel

    Don't limit ninja to set default number of workers, let it use all the cpu cores possible

    NOTE: currently stdout_buffer does not capture all the compiler warning and failure messages
    Returns:
        tuple[bool, str]: whether compilation is successful, stdout content as string
    """
    context = {}
    stdout_buffer = StringIO()

    if verbose:
        print("[Compilation] Pre-compile custom cuda binaries")

    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        # sys.stdout.flush()

        # Capture stdout during compilation
        with redirect_stdout(stdout_buffer), redirect_stderr(stdout_buffer):
            load_custom_model(custom_model_src, context, build_dir)
            # sys.stdout.flush()

        if verbose:
            print(f"[Compilation] Compilation Successful, saved cache at: {build_dir}")
    except Exception as e:
        print(f"[Compilation] Failed to compile custom CUDA kernel. Unable to cache, \nError: {e}")
        return False, stdout_buffer.getvalue(), str(e)
    
    return True, stdout_buffer.getvalue(), None


def build_compile_cache(
    custom_model_src: str,
    verbose: bool = False,
    build_dir: os.PathLike = None,
) -> tuple[bool, str, str]:
    """
    Try to build the compiled cuda code for sample and store in the cache directory
    Should be able to run on CPUs to do this massively in parallel

    Don't limit ninja to set default number of workers, let it use all the cpu cores possible
    # try do this with a subprocess
    NOTE: currently stdout_buffer does not capture all the compiler warning and failure messages
    Returns:
        tuple[bool, str]: whether compilation is successful, stdout content as string
    """
    context = {}
    stdout_buffer = StringIO()

    if verbose:
        print("[Compilation] Pre-compile custom cuda binaries")

    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        # sys.stdout.flush()

        # Capture stdout during compilation
        with redirect_stdout(stdout_buffer), redirect_stderr(stdout_buffer):
            load_custom_model(custom_model_src, context, build_dir)
            # sys.stdout.flush()

        if verbose:
            print(f"[Compilation] Compilation Successful, saved cache at: {build_dir}")
    except Exception as e:
        print(f"[Compilation] Failed to compile custom CUDA kernel. Unable to cache, \nError: {e}")
        return False, stdout_buffer.getvalue(), str(e)

    return True, stdout_buffer.getvalue(), None


def build_compile_cache_with_capturing(
    custom_model_src: str,
    verbose: bool = False,
    build_dir: os.PathLike = None
) -> tuple[int, str, str]:
    """
    Write a temporary python file to compile the custom model on CPU
    Captures the return code, stdout, and stderr
    This works for capturing, build_compile_cache does not
    """
    if build_dir:
        # Add import at the start of the source code
        custom_model_src = (
            "import os\n" f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_dir}'\n"
        ) + custom_model_src

    kernel_hash = hash(custom_model_src)
    # tmp is a temp python file we write to for compilation
    tmp = os.path.join(build_dir, f"tmp_{kernel_hash}.py")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)

    with open(tmp, "w", encoding="utf-8") as f:
        f.write(custom_model_src)

    # Execute the temporary Python file and capture output
    process = subprocess.Popen(['python', tmp], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    returncode = process.returncode

    # Clean up temporary file
    os.remove(tmp)


    if verbose:
        print("[CPU Precompile] return code: ", returncode)
        print("[CPU Precompile] stdout: \n", stdout.decode('utf-8'))
        print("[CPU Precompile] stderr: \n", stderr.decode('utf-8')) 

    return returncode, stdout.decode('utf-8'), stderr.decode('utf-8')


def eval_kernel_against_ref(
    original_model_src: str,
    custom_model_src: str,
    seed_num: int = 42,
    num_correct_trials: int = 1,
    num_perf_trials: int = 10,
    verbose: bool = False,
    measure_performance: bool = False,
    build_dir: os.PathLike = None,
    device: torch.device = torch.cuda.current_device() if torch.cuda.is_available() else None, # have to run on GPU
) -> KernelExecResult:
    """
    Evaluate the custom kernel against the original model

    num_correct_trials: number of trials to initialize different random inputs; correctness pass only if all trials pass
    num_perf_trials: run the evalutation many times to take the average
    device: GPU (cuda) device to run the evalutation on
    """
    # Check device availability and status
    assert torch.cuda.is_available(), "CUDA is not available, cannot run Eval"
    
    # Verify device is valid
    if device.type != 'cuda':
        raise ValueError(f"Device must be CUDA device, got {device}")
    
    # Check if device is accessible
    try:
        torch.cuda.set_device(device)
        torch.cuda.current_device()
    except Exception as e:
        raise RuntimeError(f"Cannot access CUDA device {device}: {e}")
    torch.set_printoptions(
        precision=4,  # Decimal places
        threshold=10,  # Total number of elements before truncating
        edgeitems=3,  # Number of elements at beginning and end of dimensions
        linewidth=80,  # Maximum width before wrapping
    )

    # set CUDA device
    torch.cuda.set_device(device)

    context = {}

    if verbose:
        print(f"[Eval] Start Evalulation! on device: {device}")
        print("[Eval] Loading Original Model")

    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        original_model_src, context
    )
    set_seed(seed_num)  # set seed for reproducible input
    init_inputs = get_init_inputs()
    init_inputs = [
        x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs
    ]

    with torch.no_grad():
        set_seed(seed_num)  # set seed for reproducible weights
        original_model = Model(*init_inputs)
        assert hasattr(original_model, "forward")
        if verbose:
            print("[Eval] Original Model Loaded")
    if verbose:
        print("[Eval] Loading and Compiling New Model with Custom CUDA Kernel")

    metadata = {}  # for storing result metadata
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)  # for debugging

    # this is where compilation happens
    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        # add hash for later to distinguish between multi-turn kernels
        ModelNew = load_custom_model(custom_model_src, context, build_dir)
        torch.cuda.synchronize(device=device)  # not sure if this is too much
    except Exception as e:
        print(
            f"Failed to compile custom CUDA kernel: Record as compilation failure. \nError: {e}"
        )
        # Categorize and add detailed metadata for compilation errors
        error_str = str(e)
        metadata["compilation_error"] = error_str
        
        # Categorize error types for better debugging
        if "lock" in error_str.lower() or "no such file or directory" in error_str.lower():
            metadata["error_category"] = "file_system_error"
            # this is a lock file error, likely due to concurrent compilation
            # this does not necessarily mean the compilation failed, but we should retry
            print(
                f"[Eval] Lock file error during compilation, Please retry. Error: {e}"
            )
            graceful_eval_cleanup(context, device)
            return None
        elif "permission" in error_str.lower():
            metadata["error_category"] = "permission_error"
        elif "cuda" in error_str.lower():
            metadata["error_category"] = "cuda_compilation_error"
        elif "syntax" in error_str.lower():
            metadata["error_category"] = "syntax_error"
        elif "import" in error_str.lower():
            metadata["error_category"] = "import_error"
        else:
            metadata["error_category"] = "unknown_compilation_error"
            
        graceful_eval_cleanup(context, device)
        return KernelExecResult(
            compiled=False, metadata=metadata
        )  # skip further steps

    # at this point we passed compilation
    try:
        with torch.no_grad():
            set_seed(seed_num)  # set seed for reproducible weights
            custom_model = ModelNew(*init_inputs)
            assert hasattr(custom_model, "forward")
            torch.cuda.synchronize(device=device)
        if verbose:
            print("[Eval] New Model with Custom CUDA Kernel Loaded")
    except RuntimeError as e:
        print(
            f"Failed to load custom CUDA kernel; Compiled but not able to run, count as runtime error. \nError: {e}"
        )
        # Categorize and add detailed metadata for runtime errors
        error_str = str(e)
        metadata["runtime_error"] = error_str
        
        # Categorize runtime error types for better debugging
        if "cuda" in error_str.lower():
            if "illegal memory access" in error_str.lower():
                metadata["error_category"] = "cuda_illegal_memory_access"
            elif "out of memory" in error_str.lower():
                metadata["error_category"] = "cuda_out_of_memory"
            elif "invalid device" in error_str.lower():
                metadata["error_category"] = "cuda_invalid_device"
            else:
                metadata["error_category"] = "cuda_runtime_error"
        elif "kernel" in error_str.lower():
            metadata["error_category"] = "kernel_launch_error"
        elif "dimension" in error_str.lower() or "shape" in error_str.lower():
            metadata["error_category"] = "tensor_dimension_error"
        else:
            metadata["error_category"] = "unknown_runtime_error"
            
        graceful_eval_cleanup(context, device)
        return KernelExecResult(
            compiled=True, correctness=False, metadata=metadata
        )  # skip further steps

    kernel_exec_result = None

    # Check Correctness
    if verbose:
        print("[Eval] Checking Correctness")
    try:
        kernel_exec_result = run_and_check_correctness(
            original_model,
            custom_model,
            get_inputs,
            metadata=metadata,
            num_correct_trials=num_correct_trials,
            verbose=verbose,
            seed=seed_num,
            device=device,
        )
    except Exception as e:
        # Categorize and add detailed metadata for correctness check errors
        error_str = str(e)
        metadata["runtime_error"] = error_str
        
        # Categorize error types during correctness checking
        if "cuda" in error_str.lower():
            if "illegal memory access" in error_str.lower():
                metadata["error_category"] = "cuda_illegal_memory_access"
            elif "out of memory" in error_str.lower():
                metadata["error_category"] = "cuda_out_of_memory"
            else:
                metadata["error_category"] = "cuda_runtime_error"
        elif "kernel" in error_str.lower():
            metadata["error_category"] = "kernel_launch_error"
        elif "dimension" in error_str.lower() or "shape" in error_str.lower():
            metadata["error_category"] = "tensor_dimension_error"
        else:
            metadata["error_category"] = "correctness_check_error"
            
        kernel_exec_result = KernelExecResult(
            compiled=True, correctness=False, metadata=metadata
        )

    # Measure Performance [Optional] | conditioned on compilation + correctness + no exception so far
    if measure_performance:
        try:
            if kernel_exec_result and kernel_exec_result.correctness:
                if verbose:
                    print("[Eval] Measuring Performance as Sample is Correct")

                torch.cuda.synchronize(device=device)
                set_seed(seed_num)
                inputs = get_inputs()
                inputs = [
                    x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                    for x in inputs
                ]
                model_new = custom_model.cuda(device=device)
                torch.cuda.synchronize(device=device)

                elapsed_times = time_execution_with_cuda_event(
                    model_new,
                    *inputs,
                    num_trials=num_perf_trials,
                    verbose=verbose,
                    device=device,
                )
                runtime_stats = get_timing_stats(elapsed_times, device=device)

                if verbose:
                    print(f"[Eval] Performance Stats: {runtime_stats}")
                kernel_exec_result.runtime = runtime_stats["mean"]
                kernel_exec_result.runtime_stats = runtime_stats
        except Exception as e:
            if verbose:
                print(f"[Eval] Error in Measuring Performance: {e}")
            kernel_exec_result.metadata["error_during_performance"] = e

    graceful_eval_cleanup(context, device)
    return kernel_exec_result


def graceful_eval_cleanup_triton(curr_context: dict, device: torch.device):
    """
    Clean up env, gpu cache, and Triton cache after evaluation
    """
    # Clean up temporary Triton module files if they exist
    if "_triton_module_path" in curr_context:
        try:
            module_path = curr_context["_triton_module_path"]
            if os.path.exists(module_path):
                os.unlink(module_path)
        except Exception:
            pass  # Ignore cleanup errors
    
    if "_triton_module_name" in curr_context:
        try:
            import sys
            module_name = curr_context["_triton_module_name"]
            if module_name in sys.modules:
                del sys.modules[module_name]
        except Exception:
            pass  # Ignore cleanup errors
    
    del curr_context
    # Clear CUDA cache and reset GPU state
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.synchronize(device=device)

    # Clean Triton cache
    _cleanup_triton_cache()


def eval_triton_kernel_against_ref(
    original_model_src: str,
    custom_model_src: str,
    seed_num: int = 42,
    num_correct_trials: int = 1,
    num_perf_trials: int = 10,
    verbose: bool = False,
    measure_performance: bool = False,
    build_dir: os.PathLike = None,
    device: torch.device = torch.cuda.current_device() if torch.cuda.is_available() else None,
) -> KernelExecResult:
    """
    Evaluate Triton kernel against the original model
    This is the Triton equivalent of eval_kernel_against_ref with the same robustness
    
    num_correct_trials: number of trials to initialize different random inputs; correctness pass only if all trials pass
    num_perf_trials: run the evaluation many times to take the average
    device: GPU (cuda) device to run the evaluation on
    """
    # Check device availability and status
    assert torch.cuda.is_available(), "CUDA is not available, cannot run Triton kernels"
    
    # Verify device is valid
    if device.type != 'cuda':
        raise ValueError(f"Device must be CUDA device, got {device}")
    
    # Check if device is accessible
    try:
        torch.cuda.set_device(device)
        torch.cuda.current_device()
    except Exception as e:
        raise RuntimeError(f"Cannot access CUDA device {device}: {e}")
    torch.set_printoptions(
        precision=4,  # Decimal places
        threshold=10,  # Total number of elements before truncating
        edgeitems=3,  # Number of elements at beginning and end of dimensions
        linewidth=80,  # Maximum width before wrapping
    )

    # set CUDA device
    torch.cuda.set_device(device)

    context = {}

    if verbose:
        print(f"[Eval] Start Triton Evaluation! on device: {device}")
        print("[Eval] Loading Original Model")

    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        original_model_src, context
    )
    set_seed(seed_num)  # set seed for reproducible input
    init_inputs = get_init_inputs()
    init_inputs = [
        x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs
    ]

    with torch.no_grad():
        set_seed(seed_num)  # set seed for reproducible weights
        original_model = Model(*init_inputs)
        assert hasattr(original_model, "forward")
        if verbose:
            print("[Eval] Original Model Loaded")
    if verbose:
        print("[Eval] Loading and Compiling Triton Kernel")

    metadata = {}  # for storing result metadata
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)  # for debugging
    metadata["kernel_type"] = "triton"

    # this is where compilation happens (matching CUDA structure)
    try:
        # Set Triton environment variables (equivalent to CUDA's TORCH_USE_CUDA_DSA)
        os.environ["TRITON_PRINT_AUTOTUNING"] = "0"  # reduce noise during eval
        # add hash for later to distinguish between multi-turn kernels
        ModelNew = load_custom_model_triton(custom_model_src, context, build_dir)
        torch.cuda.synchronize(device=device)  # not sure if this is too much
    except Exception as e:
        print(
            f"Failed to compile Triton kernel: Record as compilation failure. \nError: {e}"
        )
        # Categorize and add detailed metadata for compilation errors (Triton-specific)
        error_str = str(e)
        metadata["compilation_error"] = error_str
        
        # Categorize error types for better debugging
        if "lock" in error_str.lower() or "no such file or directory" in error_str.lower():
            metadata["error_category"] = "file_system_error"
            # this is a lock file error, likely due to concurrent compilation
            # this does not necessarily mean the compilation failed, but we should retry
            print(
                f"[Eval] Lock file or permission error during Triton compilation, Please retry. Error: {e}"
            )
            graceful_eval_cleanup_triton(context, device)
            return None
        elif "permission" in error_str.lower():
            metadata["error_category"] = "permission_error"
        elif "triton" in error_str.lower():
            if "not installed" in error_str.lower() or "import" in error_str.lower():
                metadata["error_category"] = "triton_import_error"
            elif "jit" in error_str.lower():
                metadata["error_category"] = "triton_jit_error"
            else:
                metadata["error_category"] = "triton_compilation_error"
        elif "syntax" in error_str.lower():
            metadata["error_category"] = "syntax_error"
        elif "import" in error_str.lower():
            metadata["error_category"] = "import_error"
        elif "could not get source code" in error_str.lower():
            metadata["error_category"] = "triton_source_inspection_error"
        else:
            metadata["error_category"] = "unknown_compilation_error"
            
        graceful_eval_cleanup_triton(context, device)
        return KernelExecResult(
            compiled=False, metadata=metadata
        )  # skip further steps

    # at this point we passed compilation
    try:
        with torch.no_grad():
            set_seed(seed_num)  # set seed for reproducible weights
            custom_model = ModelNew(*init_inputs)
            assert hasattr(custom_model, "forward")
            torch.cuda.synchronize(device=device)
        if verbose:
            print("[Eval] Triton Model Loaded")
    except RuntimeError as e:
        print(
            f"Failed to load Triton kernel; Compiled but not able to run, count as runtime error. \nError: {e}"
        )
        # Categorize and add detailed metadata for runtime errors
        error_str = str(e)
        metadata["runtime_error"] = error_str
        
        # Categorize runtime error types for better debugging (Triton-specific)
        if "triton" in error_str.lower():
            if "compilation" in error_str.lower():
                metadata["error_category"] = "triton_jit_compilation_error"
            elif "autotuning" in error_str.lower():
                metadata["error_category"] = "triton_autotuning_error"
            else:
                metadata["error_category"] = "triton_runtime_error"
        elif "cuda" in error_str.lower():
            if "illegal memory access" in error_str.lower():
                metadata["error_category"] = "cuda_illegal_memory_access"
            elif "out of memory" in error_str.lower():
                metadata["error_category"] = "cuda_out_of_memory"
            elif "invalid device" in error_str.lower():
                metadata["error_category"] = "cuda_invalid_device"
            else:
                metadata["error_category"] = "cuda_runtime_error"
        elif "kernel" in error_str.lower():
            metadata["error_category"] = "kernel_launch_error"
        elif "dimension" in error_str.lower() or "shape" in error_str.lower():
            metadata["error_category"] = "tensor_dimension_error"
        elif "could not get source code" in error_str.lower():
            metadata["error_category"] = "triton_source_inspection_error"
        else:
            metadata["error_category"] = "unknown_runtime_error"
            
        graceful_eval_cleanup_triton(context, device)
        return KernelExecResult(
            compiled=True, correctness=False, metadata=metadata
        )  # skip further steps

    kernel_exec_result = None

    # Check Correctness
    if verbose:
        print("[Eval] Checking Correctness")
    try:
        kernel_exec_result = run_and_check_correctness(
            original_model,
            custom_model,
            get_inputs,
            metadata=metadata,
            num_correct_trials=num_correct_trials,
            verbose=verbose,
            seed=seed_num,
            device=device,
        )
    except Exception as e:
        # Categorize and add detailed metadata for correctness check errors (Triton-specific)
        error_str = str(e)
        metadata["runtime_error"] = error_str
        
        # Categorize error types during correctness checking
        if "triton" in error_str.lower():
            if "compilation" in error_str.lower():
                metadata["error_category"] = "triton_jit_compilation_error"
            elif "autotuning" in error_str.lower():
                metadata["error_category"] = "triton_autotuning_error"
            else:
                metadata["error_category"] = "triton_runtime_error"
        elif "cuda" in error_str.lower():
            if "illegal memory access" in error_str.lower():
                metadata["error_category"] = "cuda_illegal_memory_access"
            elif "out of memory" in error_str.lower():
                metadata["error_category"] = "cuda_out_of_memory"
            else:
                metadata["error_category"] = "cuda_runtime_error"
        elif "kernel" in error_str.lower():
            metadata["error_category"] = "kernel_launch_error"
        elif "dimension" in error_str.lower() or "shape" in error_str.lower():
            metadata["error_category"] = "tensor_dimension_error"
        elif "could not get source code" in error_str.lower():
            metadata["error_category"] = "triton_source_inspection_error"
        else:
            metadata["error_category"] = "correctness_check_error"
            
        kernel_exec_result = KernelExecResult(
            compiled=True, correctness=False, metadata=metadata
        )

    # Measure Performance [Optional] | conditioned on compilation + correctness + no exception so far
    if measure_performance:
        try:
            if kernel_exec_result and kernel_exec_result.correctness:
                if verbose:
                    print("[Eval] Measuring Performance as Triton kernel is Correct")

                torch.cuda.synchronize(device=device)
                set_seed(seed_num)
                inputs = get_inputs()
                inputs = [
                    x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                    for x in inputs
                ]
                model_new = custom_model.cuda(device=device)
                torch.cuda.synchronize(device=device)

                elapsed_times = time_execution_with_cuda_event(
                    model_new,
                    *inputs,
                    num_trials=num_perf_trials,
                    verbose=verbose,
                    device=device,
                )
                runtime_stats = get_timing_stats(elapsed_times, device=device)

                if verbose:
                    print(f"[Eval] Performance Stats: {runtime_stats}")
                kernel_exec_result.runtime = runtime_stats["mean"]
                kernel_exec_result.runtime_stats = runtime_stats
        except Exception as e:
            if verbose:
                print(f"[Eval] Error in Measuring Performance: {e}")
            kernel_exec_result.metadata["error_during_performance"] = str(e)

    graceful_eval_cleanup_triton(context, device)
    return kernel_exec_result


def eval_kernel_against_ref_auto(
    original_model_src: str,
    custom_model_src: str,
    seed_num: int = 42,
    num_correct_trials: int = 1,
    num_perf_trials: int = 10,
    verbose: bool = False,
    measure_performance: bool = False,
    build_dir: os.PathLike = None,
    device: torch.device = torch.cuda.current_device() if torch.cuda.is_available() else None,
) -> KernelExecResult:
    """
    Automatically detect kernel type and use appropriate evaluation function
    """
    if detect_triton_kernel(custom_model_src):
        if verbose:
            print("[Eval] Detected Triton kernel, using Triton evaluation")
        return eval_triton_kernel_against_ref(
            original_model_src=original_model_src,
            custom_model_src=custom_model_src,
            seed_num=seed_num,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            verbose=verbose,
            measure_performance=measure_performance,
            build_dir=build_dir,
            device=device,
        )
    else:
        if verbose:
            print("[Eval] Detected CUDA kernel, using CUDA evaluation")
        return eval_kernel_against_ref(
            original_model_src=original_model_src,
            custom_model_src=custom_model_src,
            seed_num=seed_num,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            verbose=verbose,
            measure_performance=measure_performance,
            build_dir=build_dir,
            device=device,
        )


def register_and_format_exception(
    exception_type: str,
    exception_msg: Exception | str,
    metadata: dict,
    verbose: bool = False,
    truncate=False,
    max_length=200,
):
    """
    max_length characters

    NOTE: I can't get torch truncate to work during exception handling so I have this for now
    """
    # Truncate exception message if too long
    exception_str = str(exception_msg)
    if truncate and len(exception_str) > max_length:
        exception_str = exception_str[: max_length - 3] + "..."

    if verbose:
        print(f"[Exception {exception_type}] {exception_str} ")
    metadata[exception_type] = exception_str

    return metadata


def time_execution_with_cuda_event(
    kernel_fn: callable,
    *args,
    num_warmup: int = 3,
    num_trials: int = 10,
    verbose: bool = True,
    device: torch.device = None,
) -> list[float]:
    """
    Time a CUDA kernel function over multiple trials using torch.cuda.Event

    Args:
        kernel_fn: Function to time
        *args: Arguments to pass to kernel_fn
        num_trials: Number of timing trials to run
        verbose: Whether to print per-trial timing info
        device: CUDA device to use, if None, use current device

    Returns:
        List of elapsed times in milliseconds
    """
    if device is None:
        if verbose:
            print(f"Using current device: {torch.cuda.current_device()}")
        device = torch.cuda.current_device()

    # Warm ups
    for _ in range(num_warmup):
        kernel_fn(*args)
        torch.cuda.synchronize(device=device)

    print(
        f"[Profiling] Using device: {device} {torch.cuda.get_device_name(device)}, warm up {num_warmup}, trials {num_trials}"
    )
    elapsed_times = []

    # Actual trials
    for trial in range(num_trials):
        # create event marker default is not interprocess
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        kernel_fn(*args)
        end_event.record()

        # Synchronize to ensure the events have completed
        torch.cuda.synchronize(device=device)

        # Calculate the elapsed time in milliseconds
        elapsed_time_ms = start_event.elapsed_time(end_event)
        if verbose:
            print(f"Trial {trial + 1}: {elapsed_time_ms:.3g} ms")
        elapsed_times.append(elapsed_time_ms)

    return elapsed_times


def run_and_check_correctness(
    original_model_instance: nn.Module,
    new_model_instance: nn.Module,
    get_inputs_fn: callable,
    metadata: dict,
    num_correct_trials: int,
    verbose=False,
    seed=42,
    device=None,
) -> KernelExecResult:
    """
    run the model and check correctness,
    assume model already loaded and compiled (loaded and compiled in the caller)
    this is all on GPU, requiring cuda device and transfer .cuda()

    num_correct_trials: run the evalutation multiple times with (ideally) different random inputs to ensure correctness
    """
    pass_count = 0

    # Generate num_correct_trials seeds deterministically from the initial seed
    torch.manual_seed(seed)
    correctness_trial_seeds = [
        torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_correct_trials)
    ]

    with torch.no_grad():

        for trial in range(num_correct_trials):

            trial_seed = correctness_trial_seeds[trial]
            if verbose:
                print(f"[Eval] Generating Random Input with seed {trial_seed}")

            set_seed(trial_seed)
            inputs = get_inputs_fn()
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]

            set_seed(trial_seed)
            model = original_model_instance.cuda(device=device)

            set_seed(trial_seed)
            model_new = new_model_instance.cuda(device=device)

            output = model(*inputs)
            torch.cuda.synchronize(device=device)
            # ensure all GPU operations are completed before checking results

            try:
                output_new = model_new(*inputs)
                torch.cuda.synchronize(device=device)
                if output.shape != output_new.shape:
                    metadata = register_and_format_exception(
                        "correctness_issue",
                        f"Output shape mismatch: Expected {output.shape}, got {output_new.shape}",
                        metadata,
                    )
                    if verbose:
                        print(
                            f"[FAIL] trial {trial}: Output shape mismatch: Expected {output.shape}, got {output_new.shape}"
                        )
                    return KernelExecResult(
                        compiled=True, correctness=False, metadata=metadata
                    )

                # check output value difference
                if not torch.allclose(
                    output, output_new, atol=1e-02, rtol=1e-02
                ):  # fail
                    max_diff = torch.max(torch.abs(output - output_new)).item()
                    avg_diff = torch.mean(torch.abs(output - output_new)).item()
                    metadata.setdefault("max_difference", []).append(f"{max_diff:.6f}")
                    metadata.setdefault("avg_difference", []).append(f"{avg_diff:.6f}")
                    metadata["correctness_issue"] = "Output mismatch"
                    if verbose:
                        print(f"[FAIL] trial {trial}: Output mismatch")
                else:  # pass
                    pass_count += 1
                    if verbose:
                        print(f"[PASS] trial {trial}: New Model matches Model")

            except Exception as e:
                print("[Error] Exception happens during correctness check")
                print(f"Error in launching kernel for ModelNew: {e}")

                metadata = register_and_format_exception(
                    "runtime_error", e, metadata, truncate=True
                )
                return KernelExecResult(
                    compiled=True, correctness=False, metadata=metadata
                )
                # break

    if verbose:
        print(
            f"[Eval] Pass count: {pass_count}, num_correct_trials: {num_correct_trials}"
        )

    # put all the useful info here!
    metadata["correctness_trials"] = f"({pass_count} / {num_correct_trials})"

    if pass_count == num_correct_trials:
        return KernelExecResult(compiled=True, correctness=True, metadata=metadata)
    else:
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)


def check_metadata_serializable(metadata: dict):
    """
    Ensure metadata is JSON serializable,
    if not, convert non-serializable values to strings
    """
    try:
        json.dumps(metadata)
    except (TypeError, OverflowError) as e:
        print(f"[WARNING] Metadata is not JSON serializable, error: {str(e)}")
        # Convert non-serializable values to strings
        metadata = {
            "eval_0": {
                k: (
                    str(v)
                    if not isinstance(
                        v, (dict, list, str, int, float, bool, type(None))
                    )
                    else v
                )
                for k, v in metadata["eval_0"].items()
            }
        }
        print(
            f"[WARNING] Metadata now converted to string: {metadata} to be JSON serializable"
        )

    return metadata

def check_metadata_serializable_all_types(metadata: dict):
    """
    Ensure metadata is JSON serializable,
    if not, convert non-serializable values to strings recursively
    """
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

    try:
        json.dumps(metadata)
        return metadata
    except (TypeError, OverflowError) as e:
        print(f"[WARNING] Metadata is not JSON serializable, error: {str(e)}")
        # Convert non-serializable values to strings recursively
        converted_metadata = convert_to_serializable(metadata)
        print(
            f"[WARNING] Metadata now converted to be JSON serializable: {converted_metadata}"
        )
        return converted_metadata


################################################################################
# Performance Eval
################################################################################


def fetch_baseline_time(
    level_name: str, problem_id: int, dataset: list[str], baseline_time_filepath: str
) -> dict:
    """
    Fetch the baseline time from the time
    """
    if not os.path.exists(baseline_time_filepath):
        raise FileNotFoundError(
            f"Baseline time file not found at {baseline_time_filepath}"
        )

    with open(baseline_time_filepath, "r") as f:
        baseline_json = json.load(f)

    problem_name = dataset[problem_id].split("/")[-1]
    baseline_time = baseline_json[level_name].get(problem_name, None)
    return baseline_time


def get_timing_stats(elapsed_times: list[float], device: torch.device = None) -> dict:
    """Get timing statistics from a list of elapsed times.

    Args:
        elapsed_times: List of elapsed times in milliseconds
        device: CUDA device, record device info
    Returns:
        Dict containing mean, std, min, max and num_trials
        all timing are in ms
    """

    stats = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
    }

    if device:
        stats["hardware"] = torch.cuda.get_device_name(device=device)
        stats["device"] = str(device)  # for debugging

    return stats


# if __name__ == "__main__":
# fetch_kernel_from_database("kernelbench_prompt_v2_level_2", 1, 1, "http://localhost:9091")
# print(fetch_ref_arch_from_level_problem_id("2", 1, with_name=True))
# fetch_baseline_time("level1", 0, ["1_Square_matrix_multiplication_.py"], "tests/baseline_time_matx3.json")
