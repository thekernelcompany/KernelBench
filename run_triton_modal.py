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
    parser.add_argument("--num_gradient_trials", type=int, default=3, help="Number of gradient correctness trials")
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
        print("🏃 Quick test mode: 1 correctness trial, 10 performance trials")
    
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
        print(f"📋 Reference: {args.reference} (local)")
    else:
        ref_origin = "kernelbench"  
        print(f"📚 Reference: KernelBench Level {args.level}, Problem {args.problem_id}")
    
    print(f"📄 Kernel: {args.kernel}")
    print(f"🚀 GPU: {args.gpu}")
    if args.test_backward_pass:
        print("🔄 Mode: Forward + Backward Pass Evaluation")
    else:
        print("➡️ Mode: Forward Pass Only")
    print(f"🔧 Trials: {args.num_correct_trials} correctness, {args.num_perf_trials} performance")
    if args.test_backward_pass:
        print(f"🔧 Gradient Trials: {args.num_gradient_trials}, Tolerance: {args.gradient_tolerance}")
    
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
    print("🚀 LAUNCHING MODAL EVALUATION")
    print("="*60)
    print(f"Command: {final_cmd}")
    print()
    print("📡 Modal logs will appear below...")
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
        print("✅ Modal evaluation completed successfully!")
        
        # Try to save JSON results locally
        try:
            save_json_results(args, process.stdout if exit_code == 0 else None)
        except Exception as e:
            print(f"⚠️ Could not save JSON results: {e}")
    else:
        print(f"❌ Modal evaluation failed with exit code: {exit_code}")
        
    return exit_code

def save_json_results(args, modal_output):
    """Save the JSON results from Modal evaluation to a local file"""
    if not modal_output:
        print("⚠️ No output to save")
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
    
    # Try to extract JSON from the modal output
    # The modal function should return a JSON structure
    json_data = {
        "timestamp": timestamp,
        "evaluation_config": {
            "kernel_path": args.kernel,
            "ref_origin": "kernelbench" if args.level else "local",
            "level": args.level,
            "problem_id": args.problem_id,
            "reference_path": args.reference,
            "test_backward_pass": args.test_backward_pass,
            "num_correct_trials": args.num_correct_trials,
            "num_perf_trials": args.num_perf_trials,
            "num_gradient_trials": args.num_gradient_trials if args.test_backward_pass else None,
            "gradient_tolerance": args.gradient_tolerance if args.test_backward_pass else None,
            "gpu": args.gpu,
            "quick_mode": args.quick
        },
        "modal_output": modal_output,
        "raw_logs": modal_output  # Keep the full modal output for reference
    }
    
    # Try to parse the JSON result section from modal output
    import re
    
    # Look for the JSON RESULT section
    json_result_pattern = r'📊 JSON RESULT\n=+\n(.*?)\n=+'
    json_match = re.search(json_result_pattern, modal_output, re.DOTALL)
    
    if json_match:
        try:
            json_content = json_match.group(1).strip()
            parsed_result = json.loads(json_content)
            json_data["modal_result"] = parsed_result
            print(f"✅ Successfully extracted structured results")
        except json.JSONDecodeError as e:
            print(f"⚠️ Could not parse JSON result: {e}")
            # Fallback to looking for any JSON-like structures
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            potential_jsons = re.findall(json_pattern, modal_output)
            
            for potential_json in potential_jsons:
                try:
                    parsed = json.loads(potential_json)
                    if isinstance(parsed, dict) and ('status' in parsed or 'compiled' in parsed):
                        json_data["modal_result"] = parsed
                        break
                except json.JSONDecodeError:
                    continue
    else:
        print("⚠️ Could not find JSON RESULT section in output")
    
    # Save to file
    output_path = results_dir / filename
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"💾 Results saved to: {output_path}")
    
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