#!/usr/bin/env python3
"""
Test script for Triton backward pass evaluation infrastructure

This script tests the new backward pass evaluation functions with a ReLU example.
"""

import sys
import os
sys.path.append('.')

import torch
from src.eval import eval_triton_backward_pass
from src.utils import read_file

def main():
    print("=" * 60)
    print("Testing Triton Backward Pass Evaluation Infrastructure")
    print("=" * 60)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Skipping test.")
        return
    
    print(f"✅ CUDA available. GPU: {torch.cuda.get_device_name()}")
    
    # Check GPU memory
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    reserved_memory = torch.cuda.memory_reserved(device)
    free_memory = total_memory - allocated_memory
    
    print(f"💾 GPU Memory: {total_memory / 1024**3:.2f} GB total")
    print(f"💾 Available: {free_memory / 1024**3:.2f} GB")
    print(f"💾 Allocated: {allocated_memory / 1024**3:.2f} GB")
    print(f"💾 Reserved: {reserved_memory / 1024**3:.2f} GB")
    
    if free_memory < 500 * 1024**2:  # Less than 500 MB free
        print("⚠️  WARNING: Limited GPU memory available. Test may use reduced input sizes.")
    
    # Load reference ReLU model
    ref_model_path = "KernelBench/level1/19_ReLU.py"
    if not os.path.exists(ref_model_path):
        print(f"❌ Reference model not found: {ref_model_path}")
        return
    
    ref_model_src = read_file(ref_model_path)
    print(f"✅ Loaded reference ReLU model: {ref_model_path}")
    
    # Load custom Triton backward pass model
    custom_model_path = "src/prompts/model_new_ex_relu_backward_triton.py"
    if not os.path.exists(custom_model_path):
        print(f"❌ Custom model not found: {custom_model_path}")
        return
    
    custom_model_src = read_file(custom_model_path)
    print(f"✅ Loaded custom Triton backward pass model: {custom_model_path}")
    
    print("\n" + "=" * 50)
    print("Running Backward Pass Evaluation")
    print("=" * 50)
    
    # Clear GPU memory before testing
    torch.cuda.empty_cache()
    
    try:
        # Run backward pass evaluation with memory-conservative settings
        result = eval_triton_backward_pass(
            original_model_src=ref_model_src,
            custom_model_src=custom_model_src,
            seed_num=42,
            num_correct_trials=1,      # Reduce trials for limited memory
            num_gradient_trials=1,     # Test gradients once (memory intensive)
            num_perf_trials=2,         # Minimal performance trials
            gradient_tolerance=1e-3,   # Slightly relaxed tolerance
            verbose=True,              # Enable verbose output
            measure_performance=False, # Skip performance to save memory
            device=torch.device(f'cuda:{torch.cuda.current_device()}')
        )
        
        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        
        print(f"📊 Compiled: {result.compiled}")
        print(f"📊 Forward + Backward Correctness: {result.correctness}")
        
        if result.metadata:
            print(f"📊 Hardware: {result.metadata.get('hardware', 'Unknown')}")
            print(f"📊 Kernel Type: {result.metadata.get('kernel_type', 'Unknown')}")
            print(f"📊 Evaluation Type: {result.metadata.get('evaluation_type', 'Unknown')}")
            
            if 'correctness_trials' in result.metadata:
                print(f"📊 Forward Correctness Trials: {result.metadata['correctness_trials']}")
            
            if 'gradient_correctness' in result.metadata:
                print(f"📊 Gradient Correctness Trials: {result.metadata['gradient_correctness']}")
                
            if 'backward_pass_correctness' in result.metadata:
                print(f"📊 Backward Pass Overall: {'✅ PASS' if result.metadata['backward_pass_correctness'] else '❌ FAIL'}")
            
            # Display memory information if available
            if 'gpu_memory_info' in result.metadata:
                mem_info = result.metadata['gpu_memory_info']
                print(f"💾 Final Memory State:")
                print(f"💾   Total: {mem_info['total_gb']} GB")
                print(f"💾   Allocated: {mem_info['allocated_gb']} GB") 
                print(f"💾   Reserved: {mem_info['reserved_gb']} GB")
            
            # Display gradient trial details if available
            if 'gradient_trials' in result.metadata:
                print(f"📋 Gradient Trial Details:")
                for trial_info in result.metadata['gradient_trials']:
                    status = "✅ PASS" if trial_info['passed'] else "❌ FAIL"
                    trial_num = trial_info['trial']
                    if 'scale_factor' in trial_info:
                        scale = trial_info['scale_factor']
                        print(f"📋   Trial {trial_num}: {status} (input scale: {scale})")
                    elif 'error' in trial_info:
                        error = trial_info['error']
                        print(f"📋   Trial {trial_num}: {status} ({error})")
                    else:
                        print(f"📋   Trial {trial_num}: {status}")
        
        if result.runtime > 0:
            print(f"⚡ Backward Pass Runtime: {result.runtime:.3f} ms")
            
        if result.runtime_stats:
            print(f"⚡ Runtime Stats: {result.runtime_stats}")
        
        # Overall verdict
        if result.compiled and result.correctness:
            if result.metadata.get('backward_pass_correctness', False):
                print(f"\n🎉 SUCCESS: Triton ReLU backward pass works correctly!")
                print(f"   ✅ Forward pass correctness: PASS")
                print(f"   ✅ Backward pass correctness: PASS")
                if result.runtime > 0:
                    print(f"   ⚡ Performance measured: {result.runtime:.3f} ms")
            else:
                print(f"\n❌ PARTIAL SUCCESS: Forward pass works, but backward pass failed")
        else:
            print(f"\n❌ FAILURE: Kernel compilation or forward pass failed")
            if result.metadata.get('compilation_error'):
                print(f"   Compilation Error: {result.metadata['compilation_error']}")
            
    except Exception as e:
        print(f"\n❌ ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("Triton Backward Pass Test Complete")
    print("=" * 60)
    
    # Final cleanup
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        final_allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"🧹 Final GPU memory allocated: {final_allocated:.2f} GB")

if __name__ == "__main__":
    main() 