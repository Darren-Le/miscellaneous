#!/usr/bin/env python3
"""
Performance comparison script for original vs Numba-optimized Market Split solver
"""

import time
import numpy as np
from ms_data import MSData
from ms_solve import MarketSplit  # Original version
from ms_solve_numba import MarketSplitNumba, ms_run_numba  # Numba version

def performance_test():
    print("=== Numba Performance Test ===\n")
    
    # Load data
    data_path = "ms_instance/01-marketsplit/instances"
    sol_path = "ms_instance/01-marketsplit/solutions"
    ms_data = MSData(data_path, sol_path)
    
    # Test different problem sizes
    test_cases = []
    for m in [3, 4, 5]:  # Start with smaller instances
        instances = ms_data.get(m=m)
        if instances:
            test_cases.extend(instances[:2])  # First 2 instances of each size
    
    results = []
    
    for inst in test_cases:
        A, d = inst['A'], inst['d']
        instance_id = inst['id']
        opt_sol = ms_data.get_solution(instance_id)
        m, n = A.shape
        
        print(f"Testing {instance_id} (m={m}, n={n})")
        
        try:
            # Original version
            print("  Running original version...")
            start_time = time.time()
            ms_orig = MarketSplit(A, d, max_sols=100, debug=False)
            orig_solutions = ms_orig.enumerate()
            orig_time = time.time() - start_time
            
            # Numba version (first run includes compilation time)
            print("  Running Numba version (includes compilation)...")
            start_time = time.time()
            numba_result = ms_run_numba(A, d, instance_id, opt_sol, max_sols=100, debug=False)
            numba_time_with_compile = time.time() - start_time
            
            # Numba version (second run, no compilation)
            print("  Running Numba version (compiled)...")
            start_time = time.time()
            numba_result2 = ms_run_numba(A, d, instance_id, opt_sol, max_sols=100, debug=False)
            numba_time = time.time() - start_time
            
            # Verify results match
            solutions_match = len(orig_solutions) == numba_result['solutions_count']
            optimal_match = (opt_sol is not None and 
                           any(np.array_equal(sol, opt_sol) for sol in orig_solutions) == 
                           numba_result['optimal_found'])
            
            speedup_with_compile = orig_time / numba_time_with_compile if numba_time_with_compile > 0 else float('inf')
            speedup = orig_time / numba_time if numba_time > 0 else float('inf')
            
            result = {
                'instance': instance_id,
                'size': f"({m},{n})",
                'orig_time': orig_time,
                'numba_time_with_compile': numba_time_with_compile,
                'numba_time': numba_time,
                'speedup_with_compile': speedup_with_compile,
                'speedup': speedup,
                'solutions_match': solutions_match,
                'optimal_match': optimal_match,
                'orig_solutions': len(orig_solutions),
                'numba_solutions': numba_result['solutions_count']
            }
            
            results.append(result)
            
            print(f"    Original:     {orig_time:.4f}s ({len(orig_solutions)} solutions)")
            print(f"    Numba (+comp): {numba_time_with_compile:.4f}s ({numba_result['solutions_count']} solutions)")
            print(f"    Numba:        {numba_time:.4f}s ({numba_result2['solutions_count']} solutions)")
            print(f"    Speedup:      {speedup:.1f}x")
            print(f"    Correct:      {'✓' if solutions_match and optimal_match else '✗'}")
            print()
            
        except Exception as e:
            print(f"    Error: {e}")
            print()
    
    # Summary
    print("=== SUMMARY ===")
    print(f"{'Instance':<15} {'Size':<8} {'Original':<10} {'Numba':<10} {'Speedup':<8} {'Status'}")
    print("-" * 65)
    
    total_orig_time = 0
    total_numba_time = 0
    
    for r in results:
        total_orig_time += r['orig_time']
        total_numba_time += r['numba_time']
        status = "✓" if r['solutions_match'] and r['optimal_match'] else "✗"
        print(f"{r['instance']:<15} {r['size']:<8} {r['orig_time']:<10.4f} {r['numba_time']:<10.4f} {r['speedup']:<8.1f} {status}")
    
    overall_speedup = total_orig_time / total_numba_time if total_numba_time > 0 else float('inf')
    print("-" * 65)
    print(f"{'TOTAL':<15} {'':>8} {total_orig_time:<10.4f} {total_numba_time:<10.4f} {overall_speedup:<8.1f}")
    
    print(f"\nOverall speedup: {overall_speedup:.1f}x")
    print(f"Average per-instance speedup: {np.mean([r['speedup'] for r in results]):.1f}x")
    
    return results

def stress_test():
    """Test on larger instances to see real performance gains"""
    print("\n=== Stress Test (Larger Instances) ===\n")
    
    data_path = "ms_instance/01-marketsplit/instances"
    sol_path = "ms_instance/01-marketsplit/solutions"
    ms_data = MSData(data_path, sol_path)
    
    # Test progressively larger instances
    for m in [5, 6, 7]:
        instances = ms_data.get(m=m)
        if instances:
            inst = instances[0]  # Test first instance of each size
            A, d = inst['A'], inst['d']
            instance_id = inst['id']
            opt_sol = ms_data.get_solution(instance_id)
            
            print(f"Stress testing {instance_id} (m={m}, n={inst['n']})")
            
            try:
                # Only test Numba version for larger instances
                start_time = time.time()
                result = ms_run_numba(A, d, instance_id, opt_sol, max_sols=10, debug=False)
                solve_time = time.time() - start_time
                
                print(f"  Numba: {solve_time:.4f}s")
                print(f"  Solutions: {result['solutions_count']}")
                print(f"  Backtrack loops: {result['backtrack_loops']:,}")
                print(f"  Success: {'✓' if result['success'] else '✗'}")
                
                if result['success'] and result['solutions_count'] > 0:
                    print(f"  First solution: {result['first_solution_time']:.4f}s")
                    print(f"  Pruning effects: 1st={result['first_pruning_effect_count']:,}, "
                          f"2nd={result['second_pruning_effect_count']:,}, "
                          f"3rd={result['third_pruning_effect_count']:,}")
                
            except Exception as e:
                print(f"  Error: {e}")
            
            print()

if __name__ == "__main__":
    # Run performance comparison
    results = performance_test()
    
    # Run stress test on larger instances
    stress_test()
    
    print("Test completed! If you see significant speedups (10x+), the Numba optimization is working well.")
    print("Note: First run includes JIT compilation time, subsequent runs will be faster.")