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
from pathlib import Path

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
    print(f"🔧 Trials: {args.num_correct_trials} correctness, {args.num_perf_trials} performance")
    
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
    
    # Execute command with environment and real-time output
    try:
        process = subprocess.run(
            final_cmd.split(),
            env=env,
            text=True,
            capture_output=False,  # Show output in real-time
            check=True
        )
        exit_code = process.returncode
    except subprocess.CalledProcessError as e:
        exit_code = e.returncode
    
    print("-" * 60)
    if exit_code == 0:
        print("✅ Modal evaluation completed successfully!")
    else:
        print(f"❌ Modal evaluation failed with exit code: {exit_code}")
        
    return exit_code

if __name__ == "__main__":
    main() 