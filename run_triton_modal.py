#!/usr/bin/env python3
"""
🚀 Simple wrapper to run Triton kernel evaluation on Modal H100

This script handles workspace mounting and provides a clean interface for running
Triton kernels against KernelBench problems or local references.

Usage:
    # Against KernelBench problem
    python run_triton_modal.py --kernel my_triton_kernel.py --level 1 --problem_id 3
    
    # Against local reference
    python run_triton_modal.py --kernel my_triton_kernel.py --reference my_reference.py
    
    # Quick test (fewer trials)
    python run_triton_modal.py --kernel my_triton_kernel.py --level 1 --problem_id 3 --quick
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Run Triton kernel evaluation on Modal H100")
    
    # Required arguments
    parser.add_argument("--kernel", required=True, help="Path to your Triton kernel file")
    
    # Reference options (mutually exclusive)
    ref_group = parser.add_mutually_exclusive_group(required=True)
    ref_group.add_argument("--reference", help="Path to local reference file")
    ref_group.add_argument("--level", type=int, help="KernelBench level (requires --problem_id)")
    
    parser.add_argument("--problem_id", type=int, help="KernelBench problem ID (requires --level)")
    
    # Evaluation options
    parser.add_argument("--quick", action="store_true", help="Quick test (1 correctness, 10 perf trials)")
    parser.add_argument("--num_correct_trials", type=int, default=5, help="Number of correctness trials")
    parser.add_argument("--num_perf_trials", type=int, default=100, help="Number of performance trials")
    parser.add_argument("--gpu", default="H100", choices=["H100", "L40S", "A100", "L4", "T4", "A10G"], help="GPU type")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    # Backward pass options
    parser.add_argument("--test_backward_pass", action="store_true", help="Enable backward pass testing")
    parser.add_argument("--num_gradient_trials", type=int, default=5, help="Number of gradient correctness trials")
    parser.add_argument("--gradient_tolerance", type=float, default=1e-4, help="Tolerance for gradient checking")
    parser.add_argument("--measure_backward_performance", action="store_true", default=True, 
                       help="Enable backward pass performance measurement")
    
    args = parser.parse_args()
    
    # Validation
    if args.level is not None and args.problem_id is None:
        parser.error("--problem_id is required when using --level")
    if args.problem_id is not None and args.level is None:
        parser.error("--level is required when using --problem_id")
    
    # Quick test overrides
    if args.quick:
        args.num_correct_trials = 1
        args.num_perf_trials = 10
        print("Quick test mode: 1 correctness trial, 10 performance trials")
    
    # Check if kernel file exists
    kernel_path = Path(args.kernel)
    if not kernel_path.exists():
        print(f"❌ Error: Kernel file not found: {args.kernel}")
        sys.exit(1)
    
    # Check if reference file exists (for local mode)
    if args.reference:
        ref_path = Path(args.reference)
        if not ref_path.exists():
            print(f"❌ Error: Reference file not found: {args.reference}")
            sys.exit(1)
    
    # Determine reference mode
    if args.reference:
        ref_origin = "local"
        print(f"Reference: {args.reference} (local)")
    else:
        ref_origin = "kernelbench"  
        print(f"Reference: KernelBench Level {args.level}, Problem {args.problem_id}")
    
    print(f"Kernel: {args.kernel}")
    print(f"GPU: {args.gpu}")
    if args.test_backward_pass:
        print("Mode: Forward + Backward Pass Evaluation")
    else:
        print("Mode: Forward Pass Only")
    print(f"Trials: {args.num_correct_trials} correctness, {args.num_perf_trials} performance")
    if args.test_backward_pass:
        print(f"Gradient Trials: {args.num_gradient_trials}, Tolerance: {args.gradient_tolerance}")
    
    # Prepare Modal command
    modal_script = "modal_run_and_check_triton.py"
    
    # Check if Modal script exists
    if not Path(modal_script).exists():
        print(f"❌ Error: Modal script not found: {modal_script}")
        print("Make sure you're running this from the KernelBench root directory")
        sys.exit(1)
    
    # Build Modal command
    cmd_parts = [
        "modal", "run", modal_script,
        "--kernel_src_path", str(kernel_path.absolute()),
        "--ref_origin", ref_origin,
        "--num_correct_trials", str(args.num_correct_trials),
        "--num_perf_trials", str(args.num_perf_trials),
        "--gpu", args.gpu
    ]
    
    if args.reference:
        cmd_parts.extend(["--ref_arch_src_path", str(Path(args.reference).absolute())])
    else:
        cmd_parts.extend(["--level", str(args.level), "--problem_id", str(args.problem_id)])
    
    if args.verbose:
        cmd_parts.append("--verbose")
    
    # Add workspace mounting
    workspace_mount = f"--mount,{Path.cwd()}:/workspace"
    
    # Use the Python function interface (simpler than CLI)
    final_cmd = f"python3 {modal_script} --kernel_src_path {args.kernel}"
    final_cmd += f" --ref_origin {ref_origin}"
    final_cmd += f" --num_correct_trials {args.num_correct_trials}"
    final_cmd += f" --num_perf_trials {args.num_perf_trials}"
    final_cmd += f" --gpu {args.gpu}"
    
    if args.reference:
        final_cmd += f" --ref_arch_src_path {args.reference}"
    else:
        final_cmd += f" --level {args.level} --problem_id {args.problem_id}"
        
    if args.verbose:
        final_cmd += " --verbose"
        
    # Add backward pass parameters
    if args.test_backward_pass:
        final_cmd += " --test_backward_pass"
        final_cmd += f" --num_gradient_trials {args.num_gradient_trials}"
        final_cmd += f" --gradient_tolerance {args.gradient_tolerance}"
        if args.measure_backward_performance:
            final_cmd += " --measure_backward_performance"
        
    # Set Modal environment for verbose logging
    env = os.environ.copy()
    env["MODAL_LOGS_LEVEL"] = "DEBUG" if args.verbose else "INFO"
    
    print("\n" + "="*60)
    print("LAUNCHING MODAL EVALUATION")
    print("="*60)
    print(f"Command: {final_cmd}")
    print()
    print("Modal logs will appear below...")
    print("-" * 60)
    
    # Execute command with environment and capture output for JSON saving
    try:
        process = subprocess.run(
            final_cmd.split(),
            env=env,
            text=True,
            capture_output=True,  # Capture output to extract JSON
            check=True
        )
        exit_code = process.returncode
        
        # Print the output in real-time style
        print(process.stdout)
        if process.stderr:
            print("STDERR:", process.stderr)
            
    except subprocess.CalledProcessError as e:
        exit_code = e.returncode
        print(f"Command failed with exit code: {exit_code}")
        if e.stdout:
            print("STDOUT:", e.stdout) 
        if e.stderr:
            print("STDERR:", e.stderr)
    
    print("-" * 60)
    if exit_code == 0:
        print("Modal evaluation completed successfully!")
        
        # Try to save JSON results locally
        try:
            # The run_triton_check function in modal_run_and_check_triton.py now returns
            # the full result dictionary. We need to pass that to save_json_results.
            # The stdout from the subprocess is now in result['detailed_stdout'].
            # The actual JSON content from the script is in result['script_output_json_content'].
            
            # We need to get the result object from the modal_run_and_check_triton.py script
            # For now, let's assume the run_triton_check function in modal_run_and_check_triton.py
            # is called and its result is what we need to pass.
            # This part of the code (`run_triton_modal.py`) primarily constructs a command and runs it.
            # The actual result dictionary for save_json_results would typically be constructed
            # by calling the main function of modal_run_and_check_triton.py if it were used as a library,
            # or by parsing its stdout if it printed a structured result.

            # Given the current structure, process.stdout contains the *entire* output from
            # modal_run_and_check_triton.py. We need to find the result structure within it.
            # However, the previous changes made run_triton_check return the detailed output
            # which is then passed to this script's `save_json_results` if we were calling it
            # as a library. Since we are using subprocess here, we need to adapt.

            # For simplicity, let's assume the necessary 'result' dict with 'script_output_json_content'
            # would be available if this script called modal_run_and_check_triton.py as a library.
            # The current `subprocess.run` captures `process.stdout` and `process.stderr`.
            # The `modal_run_and_check_triton.py` script itself now prints the detailed output.
            
            # The `save_json_results` expects a dictionary that mirrors the return of `run_triton_check_remote`.
            # This is not directly available from `subprocess.run` output easily without complex parsing.
            # The previous step correctly modified `save_json_results` to expect `modal_output_dict`
            # which should have `script_output_json_content`.

            # This `run_triton_modal.py` calls `modal_run_and_check_triton.py` as a command line script.
            # `modal_run_and_check_triton.py` (when run via its `if __name__ == "__main__":`) calls `run_triton_check`.
            # `run_triton_check` invokes `run_triton_check_remote.remote()` and gets a result dict.
            # This result dict (containing `script_output_json_content`) is what `save_json_results` needs.

            # The simplest way is to assume `modal_run_and_check_triton.py` could print its final result dict as JSON string
            # at the very end, which this script could then parse.
            # For now, we pass a dictionary derived from what's available.
            # The crucial part is that `modal_run_and_check_triton.py` now internally handles reading the script's JSON.

            # The `save_json_results` needs the *result dictionary* from the Modal call.
            # The `process.stdout` is the full textual output.
            # This requires `modal_run_and_check_triton.py`'s `if __name__ == "__main__"` block
            # to print its `result` dictionary as JSON at the end for this script to parse it.

            # Let's assume for now `save_json_results` can get what it needs from some structured output.
            # The most robust way is for `modal_run_and_check_triton.py` to print its final result dict.
            # If `modal_run_and_check_triton.py`'s main block does:
            # result = run_triton_check(...)
            # print(json.dumps(result))
            # Then this script can parse it:
            
            # Attempt to parse the full output from `modal_run_and_check_triton.py`
            # to find the JSON result it should now be printing from its main block.
            # This is a bit of a workaround because we're calling a script that calls another script.
            
            # The `save_json_results` was modified to take `modal_output_dict`.
            # We need to construct this dict.
            # The `modal_run_and_check_triton.py` prints the stdout and stderr.
            # If it also prints the JSON string of its result dict, we can grab that.
            
            # For now, let's make a placeholder dict and rely on the fact that
            # `script_output_json_content` will be correctly populated if modal_run_and_check_triton.py's
            # main block ensures the `result` dictionary (containing it) is made available to `save_json_results`.
            # This current `run_triton_modal.py` isn't set up to directly receive the `result` dict
            # from `modal_run_and_check_triton.py`'s `run_triton_check` function when using subprocess.
            # The `save_json_results` *expects* that dictionary.

            # The simplest fix for *this* script without changing the other one again:
            # `save_json_results` will try to find `script_output_json_content` by parsing `process.stdout`
            # This means `modal_run_and_check_triton.py` must print its result dictionary as JSON.
            # I will modify `modal_run_and_check_triton.py` to do so in the next step if this doesn't work.

            # For now, the `save_json_results` call passes a dictionary derived from what's available here.
            # It will likely fall into the "script_output_json_content not found" case in save_json_results
            # unless modal_run_and_check_triton.py prints its result dict.

            # Let's assume `modal_run_and_check_triton.py` is changed to print its result dict as a JSON string
            # as the last part of its stdout.
            
            # Try to find the JSON blob that modal_run_and_check_triton.py *should* print.
            # This is getting complicated. The fundamental issue is that run_triton_modal.py
            # calls modal_run_and_check_triton.py as a script, which *then* calls the modal function.
            # The `script_output_json_content` is inside the result of the modal function.
            # `modal_run_and_check_triton.py`'s `if __name__ == "__main__"` needs to print that dict.
            
            # Let's assume modal_run_and_check_triton.py's main prints json.dumps(result)
            # Find the last JSON object in the output.
            parsed_result_from_modal_script = {}
            try:
                # A common pattern is to print JSON at the very end.
                # Try to find a JSON object in the output.
                # Look for a line that starts with { and ends with } (basic check)
                json_lines = [line for line in process.stdout.splitlines() if line.strip().startswith('{') and line.strip().endswith('}')]
                if json_lines:
                    # Try parsing the last such line
                    parsed_result_from_modal_script = json.loads(json_lines[-1])
            except Exception:
                # If parsing fails, parsed_result_from_modal_script remains empty
                pass

            if not parsed_result_from_modal_script.get("script_output_json_content"):
                 # If we couldn't parse the full dict, or it's missing the key, make a synthetic one for save_json_results
                 parsed_result_from_modal_script = {
                    "detailed_stdout": process.stdout,
                    "detailed_stderr": process.stderr,
                    "script_output_json_content": None # Explicitly None if not found
                 }

            save_json_results(args, parsed_result_from_modal_script)

        except Exception as e:
            print(f"Could not save JSON results: {e}")
    else:
        print(f"Modal evaluation failed with exit code: {exit_code}")
        
    return exit_code

def save_json_results(args, modal_output_dict):
    """Save the JSON results from Modal evaluation to a local file"""
    if not modal_output_dict:
        print("⚠️ No modal_output_dict to save")
        return
    
    # Create results directory
    results_dir = Path("modal_results")
    results_dir.mkdir(exist_ok=True)
    
    # Generate timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    kernel_name = Path(args.kernel).stem
    
    if args.level and args.problem_id:
        filename = f"{timestamp}_{kernel_name}_L{args.level}P{args.problem_id}"
    else:
        filename = f"{timestamp}_{kernel_name}_local"
    
    if args.test_backward_pass:
        filename += "_backward"
    else:
        filename += "_forward"
        
    filename += ".json"
    output_path = results_dir / filename

    final_json_to_save = {}
    script_generated_json_str = modal_output_dict.get("script_output_json_content")

    if script_generated_json_str:
        try:
            parsed_script_json = json.loads(script_generated_json_str)
            final_json_to_save = parsed_script_json
            print(f"Successfully parsed JSON content from 'script_output_json_content'.")
            # Optionally, add some wrapper-specific metadata if needed, ensuring not to overwrite
            # For example:
            # final_json_to_save['modal_wrapper_timestamp'] = timestamp
            # final_json_to_save['modal_wrapper_args'] = vars(args)
        except json.JSONDecodeError as e:
            print(f"Could not parse 'script_output_json_content': {e}. Storing raw data instead.")
            # Fallback to storing raw output if parsing fails
            final_json_to_save = {
                "modal_wrapper_timestamp": timestamp,
                "modal_wrapper_evaluation_config": vars(args),
                "error_parsing_script_json": str(e),
                "raw_script_output_json_content": script_generated_json_str,
                "raw_modal_stdout": modal_output_dict.get("detailed_stdout"),
                "raw_modal_stderr": modal_output_dict.get("detailed_stderr")
            }
    else:
        print("'script_output_json_content' not found or empty. Storing basic Modal call info and raw logs.")
        final_json_to_save = {
            "modal_wrapper_timestamp": timestamp,
            "modal_wrapper_evaluation_config": vars(args),
            "message": "No specific JSON from KernelBench script was found.",
            "raw_modal_stdout": modal_output_dict.get("detailed_stdout"),
            "raw_modal_stderr": modal_output_dict.get("detailed_stderr")
        }

    # Save to file
    with open(output_path, 'w') as f:
        json.dump(final_json_to_save, f, indent=2)
    
    print(f"Results saved to: {output_path}")
    
    # Also save a simple summary
    summary_path = results_dir / f"{timestamp}_{kernel_name}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Modal Triton Evaluation Summary\n")
        f.write(f"================================\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Kernel: {args.kernel}\n")
        if args.level:
            f.write(f"KernelBench Level: {args.level}\n")
            f.write(f"Problem ID: {args.problem_id}\n")
        else:
            f.write(f"Reference: {args.reference}\n")
        f.write(f"GPU: {args.gpu}\n")
        f.write(f"Backward Pass: {args.test_backward_pass}\n")
        f.write(f"Quick Mode: {args.quick}\n")
        f.write(f"\nFull JSON: {output_path}\n")
    
    print(f"📄 Summary saved to: {summary_path}")

if __name__ == "__main__":
    main() 