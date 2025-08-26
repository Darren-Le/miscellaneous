#!/usr/bin/env python3
"""
Test script for the fixed Numba version
"""

import time
import numpy as np
from ms_data import MSData
from ms_solve import MarketSplit  # Original version
from ms_solve_numba import MarketSplitNumbaFixed, ms_run_numba_fixed

def test_synthetic():
    """Test with simple synthetic cases"""
    print("=== Synthetic Test Cases ===")
    
    test_cases = [
        {
            'name': 'Simple 2x3',
            'A': np.array([[1, 1, 1], [2, 1, 3]], dtype=int),
            'd': np.array([2, 4], dtype=int)
        },
        {
            'name': 'Small 3x4', 
            'A': np.array([[1, 2, 1, 3], [2, 1, 3, 1], [1, 1, 1, 1]], dtype=int),
            'd': np.array([5, 7, 3], dtype=int)
        }
    ]
    
    for case in test_cases:
        print(f"\nTesting {case['name']}: A.shape={case['A'].shape}")
        A, d = case['A'], case['d']
        
        try:
            # Test original
            print("  Original version...")
            start = time.time()
            ms_orig = MarketSplit(A, d, max_sols=10, debug=False)
            orig_solutions = ms_orig.enumerate()
            orig_time = time.time() - start
            
            # Test fixed Numba
            print("  Fixed Numba version...")
            start = time.time()
            numba_result = ms_run_numba_fixed(A, d, case['name'], max_sols=10, debug=False)
            numba_time = time.time() - start
            
            # Compare
            print(f"    Original:  {len(orig_solutions)} solutions in {orig_time:.4f}s")
            print(f"    Numba:     {numba_result['solutions_count']} solutions in {numba_time:.4f}s")
            print(f"    Match:     {'✓' if len(orig_solutions) == numba_result['solutions_count'] else '✗'}")
            
            if numba_result['solutions']:
                x = numba_result['solutions'][0]
                verification = np.array_equal(A @ x, d)
                print(f"    Valid:     {'✓' if verification else '✗'}")
                if not verification:
                    print(f"      A @ x = {A @ x}, d = {d}")
            
        except Exception as e:
            print(f"    Error: {e}")

def test_real_data():
    """Test with real data"""
    print("\n=== Real Data Test ===")
    
    try:
        data_path = "ms_instance/01-marketsplit/instances"
        sol_path = "ms_instance/01-marketsplit/solutions"
        ms_data = MSData(data_path, sol_path)
        
        # Test small instances first
        for m in [3, 4]:
            instances = ms_data.get(m=m)
            if instances:
                inst = instances[0]  # First instance
                A, d = inst['A'], inst['d']
                instance_id = inst['id']
                opt_sol = ms_data.get_solution(instance_id)
                
                print(f"\nTesting {instance_id} (m={m}, n={inst['n']})")
                
                try:
                    # Original
                    start = time.time()
                    ms_orig = MarketSplit(A, d, max_sols=10, debug=False)
                    orig_solutions = ms_orig.enumerate()
                    orig_time = time.time() - start
                    
                    # Fixed Numba (first run includes compilation)
                    start = time.time()
                    numba_result1 = ms_run_numba_fixed(A, d, instance_id, opt_sol, max_sols=10)
                    numba_time1 = time.time() - start
                    
                    # Fixed Numba (second run, compiled)
                    start = time.time()
                    numba_result2 = ms_run_numba_fixed(A, d, instance_id, opt_sol, max_sols=10)
                    numba_time2 = time.time() - start
                    
                    speedup1 = orig_time / numba_time1 if numba_time1 > 0 else float('inf')
                    speedup2 = orig_time / numba_time2 if numba_time2 > 0 else float('inf')
                    
                    print(f"  Original:       {len(orig_solutions)} solutions in {orig_time:.4f}s")
                    print(f"  Numba (+comp):  {numba_result1['solutions_count']} solutions in {numba_time1:.4f}s (speedup: {speedup1:.1f}x)")
                    print(f"  Numba:          {numba_result2['solutions_count']} solutions in {numba_time2:.4f}s (speedup: {speedup2:.1f}x)")
                    print(f"  Solutions match: {'✓' if len(orig_solutions) == numba_result2['solutions_count'] else '✗'}")
                    print(f"  Optimal found:   {'✓' if numba_result2['optimal_found'] else '✗'}")
                    
                except Exception as e:
                    print(f"  Error: {e}")
                    import traceback
                    traceback.print_exc()
                    
    except Exception as e:
        print(f"Could not load real data: {e}")
        print("Make sure data files are in the correct path")

def stress_test():
    """Test larger instances"""
    print("\n=== Stress Test ===")
    
    try:
        data_path = "ms_instance/01-marketsplit/instances"
        sol_path = "ms_instance/01-marketsplit/solutions"
        ms_data = MSData(data_path, sol_path)
        
        for m in [5, 6]:  # Larger instances
            instances = ms_data.get(m=m)
            if instances:
                inst = instances[0]
                A, d = inst['A'], inst['d']
                instance_id = inst['id']
                opt_sol = ms_data.get_solution(instance_id)
                
                print(f"\nStress testing {instance_id} (m={m}, n={inst['n']})")
                
                try:
                    start = time.time()
                    result = ms_run_numba_fixed(A, d, instance_id, opt_sol, max_sols=5)
                    elapsed = time.time() - start
                    
                    print(f"  Time: {elapsed:.4f}s")
                    print(f"  Solutions: {result['solutions_count']}")
                    print(f"  Success: {'✓' if result['success'] else '✗'}")
                    print(f"  Optimal found: {'✓' if result['optimal_found'] else '✗'}")
                    
                    if result['success']:
                        print(f"  Backtrack loops: {result['backtrack_loops']:,}")
                        print(f"  Pruning: 1st={result['first_pruning_effect_count']:,}, "
                              f"2nd={result['second_pruning_effect_count']:,}, "
                              f"3rd={result['third_pruning_effect_count']:,}")
                        
                except Exception as e:
                    print(f"  Error: {e}")
                    
    except Exception as e:
        print(f"Could not run stress test: {e}")

if __name__ == "__main__":
    print("Testing Fixed Numba Market Split Solver")
    print("=" * 50)
    
    # Test synthetic cases first
    test_synthetic()
    
    # Test with real data
    test_real_data()
    
    # Stress test larger instances
    stress_test()
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("If you see good speedups (5x+) on real data, the optimization is working.")