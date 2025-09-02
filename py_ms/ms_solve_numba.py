import numpy as np
from numba import njit
from fpylll import IntegerMatrix, LLL, BKZ
from math import gcd
from functools import reduce
import logging
import time
from datetime import datetime
import argparse
from ms_data import MSData

# Numba-optimized helper functions
@njit(fastmath=True)
def compute_l1_norm(arr):
    """Compute L1 norm manually for Numba compatibility"""
    result = 0.0
    for i in range(len(arr)):
        result += abs(arr[i])
    return result

@njit(fastmath=True)
def vector_add_scaled_inplace(result, vec, scale):
    """result += scale * vec (in-place)"""
    for i in range(len(result)):
        result[i] += scale * vec[i]

@njit(fastmath=True)
def vector_copy_inplace(dst, src):
    """Copy src to dst"""
    for i in range(len(dst)):
        dst[i] = src[i]

@njit(fastmath=True)
def check_solution_validity(v, rmax, n, tol=1e-10):
    """Check if v satisfies the solution constraints"""
    # Check v[-1] = rmax
    if abs(v[-1] - rmax) > tol:
        return False
    
    # Check -rmax <= v[i] <= rmax for i = 0, ..., n-1
    for i in range(n):
        if v[i] < -rmax - tol or v[i] > rmax + tol:
            return False
    return True

@njit(fastmath=True)
def extract_solution_inplace(v, c, rmax, n, x_out):
    """Extract x from v given the matrix structure"""
    for i in range(n):
        x_out[i] = int(round((v[i] + rmax) / (2.0 * c[i])))

@njit(fastmath=True)
def verify_solution(A, d, x):
    """Verify that Ax = d"""
    m, n = A.shape
    for i in range(m):
        sum_val = 0
        for j in range(n):
            sum_val += A[i, j] * x[j]
        if sum_val != d[i]:
            return False
    return True

@njit(fastmath=True)
def enumerate_solutions_iterative(basis, b_hat, b_hat_norms_sq, mu, 
                                 u_global_bounds, A, d, c, rmax, 
                                 max_sols, n_basis, n, m):
    """
    Iterative enumeration to avoid recursion issues with Numba
    Uses explicit stack to simulate recursion
    """
    # Constants
    c_val = (n + 1) * rmax * rmax
    max_total_solutions = 10000 if max_sols == -1 else max_sols
    
    # Results
    solutions_flat = np.zeros(max_total_solutions * n, dtype=np.int32)
    solution_count = 0
    
    # Statistics
    backtrack_loops = 0
    dive_loops = 0
    first_sol_bt_loops = 0
    first_pruning_effect_count = 0
    second_pruning_effect_count = 0
    third_pruning_effect_count = 0
    first_solution_found = False
    
    # Working arrays
    u_values = np.zeros(n_basis, dtype=np.int32)
    v = np.zeros(n + 1, dtype=np.float64)
    curr_w = np.zeros(n + 1, dtype=np.float64)
    prev_w = np.zeros(n + 1, dtype=np.float64)
    x = np.zeros(n, dtype=np.int32)
    
    # Explicit stack for iterative implementation
    # Stack entry: [idx, u_min, u_max, u_current, mu_sum, w_norm_sq_at_level]
    max_stack_size = n_basis * 1000  # Should be enough for most problems
    stack = np.zeros((max_stack_size, 6), dtype=np.float64)
    stack_ptr = 0
    
    # Initialize first level
    if n_basis > 0:
        idx = n_basis - 1
        
        # Compute initial bounds for top level
        u_min_pruning2 = int(np.floor(-u_global_bounds[idx]))
        u_max_pruning2 = int(np.ceil(u_global_bounds[idx]))
        
        # For the first level, prev_w_norm_sq = 0, mu_sum = 0
        bound_sq = c_val / b_hat_norms_sq[idx]
        bound = np.sqrt(max(0.0, bound_sq))
        u_min_pruning1 = int(np.floor(-bound))
        u_max_pruning1 = int(np.ceil(bound))
        
        u_min = max(u_min_pruning1, u_min_pruning2)
        u_max = min(u_max_pruning1, u_max_pruning2)
        
        if u_min <= u_max:
            stack[0, 0] = idx        # idx
            stack[0, 1] = u_min      # u_min
            stack[0, 2] = u_max      # u_max  
            stack[0, 3] = u_min      # u_current
            stack[0, 4] = 0.0        # mu_sum
            stack[0, 5] = 0.0        # w_norm_sq_at_level
            stack_ptr = 1
    
    while stack_ptr > 0:
        backtrack_loops += 1
        
        # Pop current state
        stack_ptr -= 1
        idx = int(stack[stack_ptr, 0])
        u_min = int(stack[stack_ptr, 1])
        u_max = int(stack[stack_ptr, 2])
        u_current = int(stack[stack_ptr, 3])
        mu_sum = stack[stack_ptr, 4]
        prev_w_norm_sq = stack[stack_ptr, 5]
        
        if u_current > u_max:
            continue  # This level is done
        
        # Update u_current for this level and push back if not done
        if u_current < u_max:
            stack[stack_ptr, 3] = u_current + 1  # Next u value
            stack_ptr += 1  # Push back for next iteration
        
        u_values[idx] = u_current
        
        if idx == 0:  # Base case - we have a complete assignment
            dive_loops += 1
            
            # Compute v = sum(u_i * basis_i)
            for i in range(n + 1):
                v[i] = 0.0
                for j in range(n_basis):
                    v[i] += u_values[j] * basis[j, i]
            
            # Check constraints
            if check_solution_validity(v, rmax, n):
                extract_solution_inplace(v, c, rmax, n, x)
                
                if verify_solution(A, d, x):
                    if not first_solution_found:
                        first_solution_found = True
                        first_sol_bt_loops = backtrack_loops
                    
                    # Store solution
                    if solution_count < max_total_solutions:
                        for i in range(n):
                            solutions_flat[solution_count * n + i] = x[i]
                        solution_count += 1
                        
                        if max_sols > 0 and solution_count >= max_sols:
                            break
            
            continue
        
        # Compute current w for this level
        coeff = u_current + mu_sum
        
        # Reconstruct prev_w from u_values at higher levels
        for i in range(n + 1):
            prev_w[i] = 0.0
        
        for level in range(idx + 1, n_basis):
            level_coeff = u_values[level]
            level_mu_sum = 0.0
            for j in range(level + 1, n_basis):
                level_mu_sum += u_values[j] * mu[j, level]
            level_coeff += level_mu_sum
            
            vector_add_scaled_inplace(prev_w, b_hat[level], level_coeff)
        
        # curr_w = coeff * b_hat[idx] + prev_w  
        vector_copy_inplace(curr_w, prev_w)
        vector_add_scaled_inplace(curr_w, b_hat[idx], coeff)
        
        curr_w_norm_sq = 0.0
        for i in range(len(curr_w)):
            curr_w_norm_sq += curr_w[i] * curr_w[i]
        
        # First pruning condition
        if curr_w_norm_sq > c_val + 1e-10:
            first_pruning_effect_count += 1
            continue
        
        # Third pruning condition
        curr_w_norm_l1 = compute_l1_norm(curr_w)
        if curr_w_norm_sq > rmax * curr_w_norm_l1 + 1e-10:
            third_pruning_effect_count += 1
            continue
        
        # Push next level onto stack
        next_idx = idx - 1
        if next_idx >= 0 and stack_ptr < max_stack_size - 1:
            # Compute mu_sum for next level
            next_mu_sum = 0.0
            for j in range(next_idx + 1, n_basis):
                next_mu_sum += u_values[j] * mu[j, next_idx]
            
            # Compute bounds for next level
            bound_sq = (c_val - curr_w_norm_sq) / b_hat_norms_sq[next_idx]
            bound = np.sqrt(max(0.0, bound_sq))
            
            next_u_min_pruning1 = int(np.floor(-bound - next_mu_sum))
            next_u_max_pruning1 = int(np.ceil(bound - next_mu_sum))
            
            next_u_min_pruning2 = int(np.floor(-u_global_bounds[next_idx]))
            next_u_max_pruning2 = int(np.ceil(u_global_bounds[next_idx]))
            
            next_u_min = max(next_u_min_pruning1, next_u_min_pruning2)
            next_u_max = min(next_u_max_pruning1, next_u_max_pruning2)
            
            # Count second pruning effect
            original_range = next_u_max_pruning1 - next_u_min_pruning1 + 1
            final_range = max(0, next_u_max - next_u_min + 1)
            if final_range < original_range:
                second_pruning_effect_count += 1
            
            if next_u_min <= next_u_max:
                stack[stack_ptr, 0] = next_idx
                stack[stack_ptr, 1] = next_u_min
                stack[stack_ptr, 2] = next_u_max
                stack[stack_ptr, 3] = next_u_min
                stack[stack_ptr, 4] = next_mu_sum
                stack[stack_ptr, 5] = curr_w_norm_sq
                stack_ptr += 1
    
    return solutions_flat, solution_count, (backtrack_loops, dive_loops, first_sol_bt_loops,
                                           first_pruning_effect_count, second_pruning_effect_count,
                                           third_pruning_effect_count)

class MarketSplit:
    def __init__(self, A, d, r=None, max_sols=-1, debug=False):
        # Set up logger
        self.logger = logging.getLogger(__name__)
        if debug:
            self.logger.setLevel(logging.DEBUG)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('%(message)s'))
                self.logger.addHandler(handler)
        else:
            self.logger.setLevel(logging.WARNING)
        
        self.A = A
        self.d = d
        self.m, self.n = A.shape
        self.n_basis = self.n - self.m + 1
        self.r = r if r else np.ones(self.n, dtype=int)

        self.backtrack_loops = 0
        self.first_sol_bt_loops = 0
        self.dive_loops = 0
        self.first_solution_time = None
        self.max_sols = max_sols  # -1 means find all solutions
        
        self.first_pruning_effect_count = 0 
        self.second_pruning_effect_count = 0
        self.third_pruning_effect_count = 0 
        
        self.rmax = None
        self.c = None
        self._basis = None  # Store as row vector
        self._b_hat = None  # Store as row vector
        self._b_bar = None  # Store as row vector
        self._b_hat_norms_sq = None
        self._b_bar_norms = None
        self._mu = None

        # coords[i] is the coordinates of basis[i]: self.L @ coords[i].T = basis[i].T
        # it is a solution of the homogenous system: (A, -d) @ coords[i].T = 0
        self._coords = None 

        # Run preprocessing
        self._get_extended_matrix()
        self._get_reduced_basis()
        self._get_gso()
        self._compute_dual_norms()
        self._get_coordinates()
        
        # Make arrays read-only
        self._basis.flags.writeable = False
        self._b_hat.flags.writeable = False
        self._b_hat_norms_sq.flags.writeable = False
        self._b_bar.flags.writeable = False
        self._mu.flags.writeable = False
        self._coords.flags.writeable = False
        
        self.verify_gso()
        self.verify_dual()

    @property
    def basis(self):
        return self._basis

    @property  
    def b_hat(self):
        return self._b_hat

    @property
    def b_hat_norms_sq(self):
        return self._b_hat_norms_sq

    @property
    def b_bar(self):
        return self._b_bar

    @property
    def b_bar_norms(self):
        return self._b_bar_norms

    @property
    def mu(self):
        return self._mu
    
    @property
    def coords(self):
        return self._coords

    def _compute_lcm(self, nums):
        def __lcm(a, b):
            return abs(a * b) // gcd(a, b)
        return reduce(__lcm, nums)
    
    def _get_extended_matrix(self, N=None):
        if N is None:
            pows = len(str(np.max(np.abs(self.A)))) + len(str(np.max(np.abs(self.d)))) + 2
            N = 10 ** pows
        
        self.rmax = self._compute_lcm(self.r)
        self.c = np.array([self.rmax // ri for ri in self.r])
        self.L = np.zeros((self.m + self.n + 1, self.n + 1), dtype=int)
        
        # First column
        self.L[:self.m, 0] = -N * self.d
        self.L[self.m:self.m + self.n, 0] = -self.rmax
        self.L[self.m + self.n, 0] = self.rmax
        
        # Top-right block
        self.L[:self.m, 1:] = N * self.A
        
        # Middle diagonal block
        for i in range(self.n):
            self.L[self.m + i, 1 + i] = 2 * self.c[i]
        

    def _get_reduced_basis(self):
        ext_m, ext_n = self.L.shape
        L_lll = IntegerMatrix.from_matrix(self.L.T.tolist())
        # LLL.reduction(L_lll)
        # Use BKZ instead of LLL for shorter basis
        BKZ.reduction(L_lll, BKZ.Param(block_size=min(30, ext_n//2))) 
        L_reduced = np.array(
            [[L_lll[i][j] for j in range(ext_m)] for i in range(ext_n)], dtype=int
        )
        
        b = []
        for j in range(self.n + 1):
            if np.all(L_reduced[j, :self.m] == 0):
                col = L_reduced[j, self.m:]
                b.append(col)
        assert(len(b) == self.n - self.m + 1)
        self._basis = np.array(b, dtype=np.float64)  # Use float64 for Numba compatibility

    
    def _get_gso(self):
        assert(self._basis.shape[0] == self.n_basis)
        assert(self._basis.shape[1] == self.n + 1)
        
        # Manual Gram-Schmidt orthogonalization
        self._b_hat = np.zeros((self.n_basis, self.n + 1), dtype=float)
        self._mu = np.zeros((self.n_basis, self.n_basis), dtype=float)
        self._b_hat_norms_sq = np.zeros(self.n_basis, dtype=float)
        
        for i in range(self.n_basis):
            # Start with the original basis vector
            self._b_hat[i] = self._basis[i].astype(float)
            
            # Subtract projections onto previous orthogonal vectors
            for j in range(i):
                # mu[i,j] = <basis[i], b_hat[j]> / ||b_hat[j]||²
                self._mu[i, j] = np.dot(self._basis[i], self._b_hat[j]) / self._b_hat_norms_sq[j]
                # Subtract projection
                self._b_hat[i] -= self._mu[i, j] * self._b_hat[j]
            
            # Set diagonal mu to 1
            self._mu[i, i] = 1.0
            
            # Compute squared norm
            self._b_hat_norms_sq[i] = np.dot(self._b_hat[i], self._b_hat[i])
        
    def _compute_dual_norms(self):
        B = self._basis.T
        B_T = self._basis
        gram = B_T @ B
        gram_inv = np.linalg.inv(gram)
        self._b_bar = (B @ gram_inv).T
        
        self._b_bar_norms = {
            'l2': np.array([np.linalg.norm(self._b_bar[i, :], ord=2) for i in range(self.n_basis)]),
            'l1': np.array([np.linalg.norm(self._b_bar[i, :], ord=1) for i in range(self.n_basis)])
        }
    
    def _get_coordinates(self):
        L_bottom = self.L[self.m:, :]
        coordinates = []
        for i in range(self.n_basis):
            x = np.linalg.solve(L_bottom, self.basis[i, :])
            coordinates.append(x)
        self._coords = np.array(coordinates)
    
    def verify_gso(self, tol=1e-10):
        """Verify that b_hat is the correct GSO of basis"""
        # Check orthogonality
        for i in range(self.n_basis):
            for j in range(i + 1, self.n_basis):
                dot_product = np.dot(self.b_hat[i], self.b_hat[j])
                if abs(dot_product) > tol:
                    self.logger.debug(f"Orthogonality failed: b_hat[{i}] · b_hat[{j}] = {dot_product}")
                    return False
        
        # Check GSO formula: basis[i] = b_hat[i] + sum(mu[i,j] * b_hat[j] for j < i)
        for i in range(self.n_basis):
            reconstructed = self.b_hat[i].copy()
            for j in range(i):
                reconstructed += self.mu[i, j] * self.b_hat[j]
            
            if not np.allclose(self.basis[i], reconstructed, atol=tol):
                self.logger.debug(f"GSO formula failed for basis[{i}]")
                return False
        
        self.logger.debug("GSO verification passed")
        return True

    def verify_dual(self, tol=1e-10):
        """Verify that b_bar is the dual of basis: <b_bar[i], basis[j]> = δ_ij"""
        for i in range(self.n_basis):
            for j in range(self.n_basis):
                dot_product = np.dot(self.b_bar[i], self.basis[j])
                expected = 1.0 if i == j else 0.0
                if abs(dot_product - expected) > tol:
                    self.logger.debug(f"Dual property failed: <b_bar[{i}], basis[{j}]> = {dot_product}, expected {expected}")
                    return False
        
        self.logger.debug("Dual verification passed")
        return True

    def enumerate(self):
        """Numba-optimized enumeration using iterative approach"""
        start_time = time.time()
        
        # Prepare global bounds for second pruning strategy
        sqrt_c = np.sqrt((self.n + 1) * self.rmax ** 2)
        u_bounds_l2 = self.b_bar_norms['l2'] * sqrt_c
        u_bounds_l1 = self.b_bar_norms['l1'] * self.rmax
        u_global_bounds = np.minimum(u_bounds_l2, u_bounds_l1)
        
        # Call Numba-optimized function
        solutions_flat, solution_count, stats = enumerate_solutions_iterative(
            self.basis, self.b_hat, self.b_hat_norms_sq, self.mu,
            u_global_bounds, self.A.astype(np.int32), self.d.astype(np.int32), 
            self.c.astype(np.float64), float(self.rmax), 
            self.max_sols, self.n_basis, self.n, self.m
        )
        
        # Update statistics
        (self.backtrack_loops, self.dive_loops, self.first_sol_bt_loops,
         self.first_pruning_effect_count, self.second_pruning_effect_count,
         self.third_pruning_effect_count) = stats
        
        # Convert solutions back to list format
        solutions = []
        for i in range(solution_count):
            sol = solutions_flat[i * self.n:(i + 1) * self.n].copy()
            solutions.append(sol)
        
        if solution_count > 0 and self.first_solution_time is None:
            self.first_solution_time = time.time() - start_time
        
        return solutions

def ms_run(A, d, instance_id, opt_sol=None, max_sols=-1, debug=False):
    try:
        start_time = time.time()
        init_start = time.time()
        ms = MarketSplit(A, d, debug=debug, max_sols=max_sols)
        init_time = time.time() - init_start
        solutions = ms.enumerate()
        solve_time = time.time() - start_time
        
        found_opt = False
        if opt_sol is not None:
            found_opt = any(np.array_equal(sol, opt_sol) for sol in solutions)
        
        return {
            'id': instance_id,
            'solutions_count': len(solutions),
            'solutions': solutions,
            'optimal_found': found_opt,
            'backtrack_loops': ms.backtrack_loops,
            'first_sol_bt_loops': ms.first_sol_bt_loops,
            'dive_loops': ms.dive_loops,
            'first_pruning_effect_count': ms.first_pruning_effect_count,  
            'second_pruning_effect_count': ms.second_pruning_effect_count,
            'third_pruning_effect_count': ms.third_pruning_effect_count,  
            'solve_time': solve_time,
            'first_solution_time': ms.first_solution_time or 0,
            'init_time': init_time,
            'success': True
        }
    except Exception as e:
        return {
            'id': instance_id,
            'solutions_count': 0,
            'solutions': [],
            'optimal_found': False,
            'backtrack_loops': 0,
            'first_sol_bt_loops': 0,
            'dive_loops': 0,
            'first_pruning_effect_count': 0,
            'second_pruning_effect_count': 0,
            'third_pruning_effect_count': 0,
            'solve_time': 0,
            'first_solution_time': 0,
            'init_time': 0, 
            'success': False,
            'error': str(e)
        }

def print_and_log(text, file_handle):
    print(text)
    file_handle.write(text + "\n")

# 主函数部分 - 使用数据文件 (Same as original, now with Numba optimization)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Market Split Solver (Numba-Optimized)')
    parser.add_argument('--data_path', type=str, default="ms_instance/01-marketsplit/instances", help='Path to instance data')
    parser.add_argument('--sol_path', type=str, default="ms_instance/01-marketsplit/solutions", help='Path to solution data')
    parser.add_argument('--max_sols', type=int, default=-1, help='Maximum number of solutions to find (-1 for all)')
    parser.add_argument('--debug', action='store_true', help='Enable debug print')
    args = parser.parse_args()

    data_path = args.data_path
    sol_path = args.sol_path
    ms_data = MSData(data_path, sol_path)

    debug_mode = args.debug
    max_sols = args.max_sols
    test_m_values = [3, 4, 5, 6, 7]
    all_results = []

    print("Market Split Solver (Numba-Optimized Version)")
    print("=" * 50)
    print("First run includes JIT compilation time...")
    print()

    for m in test_m_values:
        instances = ms_data.get(m=m)
        print(f"Testing {len(instances)} instances with m = {m}")
        
        for inst in instances:
            A, d = inst['A'], inst['d']
            opt_sol = ms_data.get_solution(inst['id'])
            result = ms_run(A, d, inst['id'], opt_sol, max_sols, debug=debug_mode)
            all_results.append(result)
            
            status = "✓" if result['success'] else "✗"
            opt_status = "✓" if result['optimal_found'] else "✗"
            # Dynamic printing
            print(f"{status} {result['id']}: {result['solutions_count']} solutions, "
                f"optimal: {opt_status}, bt_loops: {result['backtrack_loops']}, "
                f"dive_loops: {result['dive_loops']}, 1st_prune: {result['first_pruning_effect_count']}, "
                f"2nd_prune: {result['second_pruning_effect_count']}, 3rd_prune: {result['third_pruning_effect_count']}, "
                f"time: {result['solve_time']:.4f}s, "
                f"1st_sol: {result['first_solution_time']:.4f}s, 1st_bt: {result['first_sol_bt_loops']}, "
                f"init: {result['init_time']:.4f}s")
        print()

    # Results table - print and save simultaneously
    m_choices = "_".join(map(str, test_m_values))
    now = datetime.now()
    time_str = f"{now.day}d{now.hour}h{now.minute}m{now.second}s"
    log_filename = f"res_numba_{m_choices}_{time_str}.log"


    with open(log_filename, 'w') as f:
        print_and_log("=" * 136, f)
        print_and_log("RESULTS (NUMBA-OPTIMIZED)", f)
        print_and_log("=" * 136, f)
        print_and_log("", f)
        
        print_and_log(f"{'ID':<15} {'Size':<8} {'Status':<8} {'Optimal':<8} {'Time(s)':<10} {'1st_Sol(s)':<10} {'1st_Prune':<10} {'2nd_Prune':<10} {'3rd_Prune':<10} {'Init(s)':<10} {'Solutions':<10} {'1st_BT':<8} {'BT_Loops':<12} {'Dive_Loops':<12}", f)
        print_and_log("-" * 136, f)

        for result in all_results:
            inst = ms_data.get(id=result['id'])
            m, n = inst['A'].shape
            size = f"({m},{n})"
            status = "SUCCESS" if result['success'] else "FAILED"
            optimal = "✓" if result['optimal_found'] else "✗"
            print_and_log(f"{result['id']:<15} {size:<8} {status:<8} {optimal:<8} {result['solve_time']:<10.4f} {result['first_solution_time']:<10.4f} {result['first_pruning_effect_count']:<10} {result['second_pruning_effect_count']:<10} {result['third_pruning_effect_count']:<10} {result['init_time']:<10.4f} {result['solutions_count']:<10} {result['first_sol_bt_loops']:<8} {result['backtrack_loops']:<12} {result['dive_loops']:<12}", f)
            
        print_and_log("", f)
        print_and_log("=" * 136, f)

    print(f"Results saved to {log_filename}")

    # Write solutions to file
    sol_filename = log_filename.replace('.log', '.sol')
    with open(sol_filename, 'w') as f:
        for result in all_results:
            if result['success'] and result['solutions']:
                f.write(f"======={result['id']}=============\n")
                for i, solution in enumerate(result['solutions'], 1):
                    f.write(f"-----------{i}-th solution------------\n")
                    f.write(' '.join(map(str, solution)) + '\n')
                f.write('\n')

    print(f"Solutions saved to {sol_filename}")
    print()
    print("Numba optimization complete!")
    print("Note: First run includes JIT compilation time.")
    print("Subsequent runs of the same problem sizes will be faster.")