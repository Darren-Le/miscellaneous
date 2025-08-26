import numpy as np
from numba import njit, types
from numba.typed import List
import time
from fpylll import IntegerMatrix, LLL, BKZ
from math import gcd
from functools import reduce
import logging
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
def compute_l2_norm_sq(arr):
    """Compute L2 norm squared manually for Numba compatibility"""
    result = 0.0
    for i in range(len(arr)):
        result += arr[i] * arr[i]
    return result

@njit(fastmath=True)
def dot_product(a, b):
    """Manual dot product for Numba"""
    result = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

@njit(fastmath=True)
def vector_add_scaled(result, vec, scale):
    """result += scale * vec (in-place)"""
    for i in range(len(result)):
        result[i] += scale * vec[i]

@njit(fastmath=True)
def vector_copy(dst, src):
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
def extract_solution(v, c, rmax, n):
    """Extract x from v given the matrix structure"""
    x = np.zeros(n, dtype=np.int32)
    for i in range(n):
        x[i] = int(round((v[i] + rmax) / (2.0 * c[i])))
    return x

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

@njit(fastmath=True, cache=True)
def enumerate_solutions_numba(basis, b_hat, b_hat_norms_sq, mu, 
                             u_global_bounds, A, d, c, rmax, 
                             max_sols, n_basis, n, m):
    """
    Numba-optimized enumeration function
    Returns: solutions (as flat array), solution_count, stats
    """
    # Pre-allocate arrays to avoid allocations in hot loop
    max_total_solutions = 10000 if max_sols == -1 else max_sols
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
    
    c_val = (n + 1) * rmax * rmax  # Constant for pruning
    sqrt_c = np.sqrt(c_val)
    
    # Working arrays (reused to avoid allocations)
    u_values = np.zeros(n_basis, dtype=np.int32)
    v = np.zeros(n + 1, dtype=np.float64)
    curr_w = np.zeros(n + 1, dtype=np.float64)
    x = np.zeros(n, dtype=np.int32)
    
    def backtrack(idx, prev_w, prev_w_norm_sq):
        nonlocal solution_count, backtrack_loops, dive_loops, first_sol_bt_loops
        nonlocal first_pruning_effect_count, second_pruning_effect_count, third_pruning_effect_count
        nonlocal first_solution_found
        
        backtrack_loops += 1
        
        if idx == -1:
            dive_loops += 1
            
            # Compute v = sum(u_i * basis_i)
            for i in range(n + 1):
                v[i] = 0.0
                for j in range(n_basis):
                    v[i] += u_values[j] * basis[j, i]
            
            # Check constraints
            if check_solution_validity(v, rmax, n):
                # Extract solution
                x = extract_solution(v, c, rmax, n)
                
                # Verify solution
                if verify_solution(A, d, x):
                    if not first_solution_found:
                        first_solution_found = True
                        first_sol_bt_loops = backtrack_loops
                    
                    # Store solution
                    if solution_count < max_total_solutions:
                        for i in range(n):
                            solutions_flat[solution_count * n + i] = x[i]
                        solution_count += 1
                        
                        # Check if we've found enough solutions
                        if max_sols > 0 and solution_count >= max_sols:
                            return True
            return False
        
        # First pruning condition
        if prev_w_norm_sq > c_val + 1e-10:
            dive_loops += 1
            first_pruning_effect_count += 1
            return False
        
        # Compute mu_sum = sum_{i=idx+1}^{n_basis-1} u_i * mu_{i,idx}
        mu_sum = 0.0
        for j in range(idx + 1, n_basis):
            mu_sum += u_values[j] * mu[j, idx]
        
        # Compute bounds for u[idx] from first pruning condition
        bound_sq = (c_val - prev_w_norm_sq) / b_hat_norms_sq[idx]
        bound = np.sqrt(max(0.0, bound_sq))
        
        u_min_pruning1 = int(np.floor(-bound - mu_sum))
        u_max_pruning1 = int(np.ceil(bound - mu_sum))
        
        # Second pruning: global bounds
        u_min_pruning2 = int(np.floor(-u_global_bounds[idx]))
        u_max_pruning2 = int(np.ceil(u_global_bounds[idx]))
        
        # Take intersection
        u_min = max(u_min_pruning1, u_min_pruning2)
        u_max = min(u_max_pruning1, u_max_pruning2)
        
        # Count second pruning effect
        original_range = u_max_pruning1 - u_min_pruning1 + 1
        final_range = max(0, u_max - u_min + 1)
        if final_range < original_range:
            second_pruning_effect_count += 1
        
        # Third pruning strategy with Holder's inequality
        for u_val in range(u_min, u_max + 1):
            u_values[idx] = u_val
            
            # Compute w^(idx) = (u_val + mu_sum) * b_hat[idx] + prev_w
            coeff = u_val + mu_sum
            
            # curr_w = coeff * b_hat[idx] + prev_w
            vector_copy(curr_w, prev_w)
            vector_add_scaled(curr_w, b_hat[idx], coeff)
            
            # Third pruning condition: ||w||_2^2 <= rmax * ||w||_1
            w_norm_sq = coeff * coeff * b_hat_norms_sq[idx] + prev_w_norm_sq
            w_norm_l1 = compute_l1_norm(curr_w)
            
            if w_norm_sq > rmax * w_norm_l1 + 1e-10:
                if coeff > 0:
                    # Skip all remaining u_val + r (r > 0)
                    third_pruning_effect_count += (u_max - u_val)
                    break
                # Skip only current value if coeff <= 0
                third_pruning_effect_count += 1
                continue
            
            if backtrack(idx - 1, curr_w, w_norm_sq):
                return True
        
        return False
    
    # Initialize
    prev_w = np.zeros(n + 1, dtype=np.float64)
    backtrack(n_basis - 1, prev_w, 0.0)
    
    return solutions_flat, solution_count, (backtrack_loops, dive_loops, first_sol_bt_loops,
                                           first_pruning_effect_count, second_pruning_effect_count,
                                           third_pruning_effect_count)

class MarketSplitNumba:
    def __init__(self, A, d, r=None, max_sols=-1, debug=False):
        # Set up logger (same as original)
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
        self.max_sols = max_sols
        
        self.first_pruning_effect_count = 0 
        self.second_pruning_effect_count = 0
        self.third_pruning_effect_count = 0 
        
        # Same preprocessing as original (can't easily optimize with Numba)
        self.rmax = None
        self.c = None
        self._basis = None
        self._b_hat = None
        self._b_bar = None
        self._b_hat_norms_sq = None
        self._b_bar_norms = None
        self._mu = None
        self._coords = None

        # Run preprocessing (keep original logic)
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

    # Keep all the preprocessing methods from original class
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
        self._basis = np.array(b, dtype=np.float64)  # Use float64 for Numba

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
        """Same as original"""
        for i in range(self.n_basis):
            for j in range(i + 1, self.n_basis):
                dot_product = np.dot(self.b_hat[i], self.b_hat[j])
                if abs(dot_product) > tol:
                    self.logger.debug(f"Orthogonality failed: b_hat[{i}] · b_hat[{j}] = {dot_product}")
                    return False
        
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
        """Same as original"""
        for i in range(self.n_basis):
            for j in range(self.n_basis):
                dot_product = np.dot(self.b_bar[i], self.basis[j])
                expected = 1.0 if i == j else 0.0
                if abs(dot_product - expected) > tol:
                    self.logger.debug(f"Dual property failed: <b_bar[{i}], basis[{j}]> = {dot_product}, expected {expected}")
                    return False
        
        self.logger.debug("Dual verification passed")
        return True

    # Properties (same as original)
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

    def enumerate(self):
        """Numba-optimized enumeration"""
        start_time = time.time()
        
        # Prepare global bounds for second pruning strategy
        sqrt_c = np.sqrt((self.n + 1) * self.rmax ** 2)
        u_bounds_l2 = self.b_bar_norms['l2'] * sqrt_c
        u_bounds_l1 = self.b_bar_norms['l1'] * self.rmax
        u_global_bounds = np.minimum(u_bounds_l2, u_bounds_l1)
        
        # Call Numba-optimized function
        solutions_flat, solution_count, stats = enumerate_solutions_numba(
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

# Update the ms_run function to use Numba version
def ms_run_numba(A, d, instance_id, opt_sol=None, max_sols=-1, debug=False):
    try:
        start_time = time.time()
        init_start = time.time()
        ms = MarketSplitNumba(A, d, debug=debug, max_sols=max_sols)
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

# Test function
if __name__ == "__main__":
    print("Testing Numba-optimized Market Split solver...")
    
    # Load a small test case
    data_path = "ms_instance/01-marketsplit/instances"
    sol_path = "ms_instance/01-marketsplit/solutions"
    ms_data = MSData(data_path, sol_path)
    
    # Test on a small instance first
    instances = ms_data.get(m=3)
    if instances:
        inst = instances[0]
        A, d = inst['A'], inst['d']
        opt_sol = ms_data.get_solution(inst['id'])
        
        print(f"Testing instance {inst['id']} with shape {A.shape}")
        
        # Compare original vs Numba
        print("Running original version...")
        start = time.time()
        ms_orig = MarketSplit(A, d, debug=False, max_sols=10)
        orig_sols = ms_orig.enumerate()
        orig_time = time.time() - start
        
        print("Running Numba version...")
        start = time.time()
        result = ms_run_numba(A, d, inst['id'], opt_sol, max_sols=10, debug=False)
        numba_time = result['solve_time']
        
        print(f"Original: {len(orig_sols)} solutions in {orig_time:.4f}s")
        print(f"Numba:    {result['solutions_count']} solutions in {numba_time:.4f}s")
        print(f"Speedup:  {orig_time/numba_time:.2f}x")
        print(f"Solutions match: {len(orig_sols) == result['solutions_count']}")