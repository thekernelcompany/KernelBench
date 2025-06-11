#!/usr/bin/env python3
"""
Demonstration script showing how to use the enhanced run_and_check_triton.py 
for comprehensive forward and backward pass evaluation.

This script demonstrates:
1. Forward pass evaluation only (standard)
2. Comprehensive backward pass evaluation
3. Performance comparison across PyTorch eager, torch.compile, and custom kernels
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and display results"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print(f"{'-'*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # Print stdout
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        # Print stderr if there are errors
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        print(f"Exit code: {result.returncode}")
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    print("KernelBench Enhanced Backward Pass Testing Demo")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("scripts/run_and_check_triton.py"):
        print("Error: Must be run from KernelBench root directory")
        return
    
    # Example 1: Standard forward pass evaluation
    print("\n🔄 Example 1: Standard Forward Pass Evaluation")
    cmd1 = ("python scripts/run_and_check_triton.py "
            "ref_origin=kernelbench level=1 problem_id=19 "
            "kernel_src_path=src/prompts/model_new_ex_relu_backward_triton.py "
            "num_correct_trials=1 num_perf_trials=3 "
            "verbose=False")
    
    success1 = run_command(cmd1, "Forward Pass Only - ReLU")
    
    # Example 2: Comprehensive backward pass evaluation
    print("\n🔄 Example 2: Comprehensive Backward Pass Evaluation")
    cmd2 = ("python scripts/run_and_check_triton.py "
            "ref_origin=kernelbench level=1 problem_id=19 "
            "kernel_src_path=src/prompts/model_new_ex_relu_backward_triton.py "
            "test_backward_pass=True "
            "num_correct_trials=1 num_gradient_trials=1 num_perf_trials=3 "
            "gradient_tolerance=1e-3 "
            "verbose=True")
    
    success2 = run_command(cmd2, "Forward + Backward Pass - ReLU")
    
    # Example 3: Memory-conservative evaluation for limited GPU
    print("\n🔄 Example 3: Memory-Conservative Evaluation")
    cmd3 = ("python scripts/run_and_check_triton.py "
            "ref_origin=kernelbench level=1 problem_id=19 "
            "kernel_src_path=src/prompts/model_new_ex_relu_backward_triton.py "
            "test_backward_pass=True "
            "num_correct_trials=1 num_gradient_trials=1 num_perf_trials=2 "
            "gradient_tolerance=1e-3 measure_backward_performance=True "
            "verbose=False")
    
    success3 = run_command(cmd3, "Memory-Conservative Backward Pass")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    results = [
        ("Forward Pass Only", success1),
        ("Forward + Backward Pass", success2), 
        ("Memory-Conservative", success3)
    ]
    
    for desc, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{desc:<25}: {status}")
    
    print("\n📁 Check 'runs/successful/' and 'runs/failed/' directories for detailed JSON results")
    
    # Show available parameters
    print("\n" + "="*60)
    print("AVAILABLE PARAMETERS")
    print("="*60)
    print("""
Key Parameters for Backward Pass Testing:

Core Settings:
  test_backward_pass=True          # Enable backward pass testing
  num_gradient_trials=3            # Number of gradient correctness trials
  gradient_tolerance=1e-4          # Tolerance for gradient checking
  measure_backward_performance=True # Measure backward pass performance

Performance Settings:
  num_correct_trials=5             # Forward correctness trials
  num_perf_trials=100              # Performance measurement trials
  verbose=True                     # Enable detailed output

Memory Management:
  # The system automatically handles memory constraints with:
  # - Progressive input scaling (1.0 → 0.5 → 0.25)
  # - Memory cleanup between trials
  # - Double precision for accurate gradient checking
  # - Fast mode gradient checking to reduce memory usage

Output:
  # Results saved to runs/successful/ or runs/failed/
  # JSON format includes both forward and backward metrics
  # Comprehensive performance comparison: Custom vs Eager vs torch.compile
""")

if __name__ == "__main__":
    main() 