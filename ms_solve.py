import numpy as np
from fpylll import IntegerMatrix, LLL
from math import gcd
from functools import reduce
import logging
import time
from ms_data import MSData

class MarketSplit:
    def __init__(self, A, d, r=None, debug=False):
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
        self.dive_loops = 0

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
        LLL.reduction(L_lll)
        L_reduced = np.array(
            [[L_lll[i][j] for j in range(ext_m)] for i in range(ext_n)], dtype=int
        )
        
        b = []
        for j in range(self.n + 1):
            if np.all(L_reduced[j, :self.m] == 0):
                col = L_reduced[j, self.m:]
                b.append(col)
        assert(len(b) == self.n - self.m + 1)
        self._basis = np.array(b)

    
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
        sols = []
        c = (self.n + 1) * self.rmax ** 2  # 预计算常数

        def backtrack(idx, u_values, prev_w):
            """
            回溯算法
            idx: 当前要确定的 u 的索引 (从 n_basis-1 递减到 0)
            u_values: 长度为 n_basis 的数组
            prev_w: w^(idx+1)
            """
            self.backtrack_loops += 1

            if idx == -1:
                self.dive_loops += 1
                # 计算 v = sum(u_i * basis_i)
                v = np.zeros(self.n + 1)
                for i in range(self.n_basis):
                    v += u_values[i] * self.basis[i]
                
                # 检查约束: v[-1] = rmax 且 -rmax <= v[i] <= rmax for i = 0, ..., n-1
                if abs(v[-1] - self.rmax) < 1e-10:  # v0 = rmax => x0 = 1
                    if np.all(v[:-1] >= -self.rmax - 1e-10) and np.all(v[:-1] <= self.rmax + 1e-10):
                        # 由于矩阵M的特殊结构，我们可以直接恢复解
                        # x0 = 1 (由 v0 = rmax 保证)
                        # x_i = (v_{i-1} + rmax) / (2 * c_{i-1}) for i = 1, ..., n
                        
                        x = np.zeros(self.n, dtype=int)
                        for i in range(self.n):
                            x[i] = int(round((v[i] + self.rmax) / (2 * self.c[i])))
                        
                        # 验证解
                        if np.allclose(self.A @ x, self.d):
                            sols.append(x.copy())
                return
            
            # 第一个剪枝条件：检查 ||w^(idx+1)||_2^2 是否已经超过界限
            prev_w_norm_sq = np.dot(prev_w, prev_w)
            if prev_w_norm_sq > c:
                self.dive_loops += 1
                return  # 剪枝
            
            # 计算 sum_{i=idx+1}^{n_basis-1} u_i * mu_{i,idx}
            mu_sum = 0.0
            for j in range(idx + 1, self.n_basis):
                mu_sum += u_values[j] * self.mu[j, idx]
            
            # 根据第一个剪枝条件计算 u[idx] 的范围
            # (u_idx + mu_sum)^2 <= (c - ||w^(idx+1)||_2^2) / ||b_hat^(idx)||_2^2
            bound_sq = (c - prev_w_norm_sq) / self.b_hat_norms_sq[idx]
            bound = np.sqrt(max(0, bound_sq))
            
            # u_idx 的范围是: -bound - mu_sum <= u_idx <= bound - mu_sum
            u_min = int(np.floor(-bound - mu_sum))
            u_max = int(np.ceil(bound - mu_sum))
            
            for u_val in range(u_min, u_max + 1):
                u_values[idx] = u_val
                
                # 计算 w^(idx) = (u_val + mu_sum) * b_hat[idx] + prev_w
                coeff = u_val + mu_sum
                curr_w = coeff * self.b_hat[idx] + prev_w
                # assert(np.dot(curr_w, curr_w) <= c)
                
                backtrack(idx - 1, u_values, curr_w)
        
        # Start backtracking
        u_values = np.zeros(self.n_basis, dtype=int)
        initial_w = np.zeros(self.n + 1)
        backtrack(self.n_basis - 1, u_values, initial_w)
        
        return sols

def ms_run(A, d, instance_id, opt_sol=None, debug=False):
    try:
        start_time = time.time()
        ms = MarketSplit(A, d, debug=debug)
        solutions = ms.enumerate()
        solve_time = time.time() - start_time
        
        found_opt = False
        if opt_sol is not None:
            found_opt = any(np.array_equal(sol, opt_sol) for sol in solutions)
        
        return {
            'id': instance_id,
            'solutions_count': len(solutions),
            'optimal_found': found_opt,
            'backtrack_loops': ms.backtrack_loops,
            'dive_loops': ms.dive_loops,
            'solve_time': solve_time,
            'success': True
        }
    except Exception as e:
        return {
            'id': instance_id,
            'solutions_count': 0,
            'optimal_found': False,
            'backtrack_loops': 0,
            'dive_loops': 0,
            'solve_time': 0,
            'success': False,
            'error': str(e)
        }


# 主函数部分 - 使用数据文件
if __name__ == "__main__":
    data_path = "ms_instance/01-marketsplit/instances"
    sol_path = "ms_instance/01-marketsplit/solutions"
    ms_data = MSData(data_path, sol_path)

    debug_mode = False
    test_m_values = [3, 4, 5]
    all_results = []

    for m in test_m_values:
        instances = ms_data.get(m=m)
        print(f"Testing {len(instances)} instances with m = {m}")
        
        for inst in instances:
            A, d = inst['A'], inst['d']
            opt_sol = ms_data.get_solution(inst['id'])
            result = ms_run(A, d, inst['id'], opt_sol, debug=debug_mode)
            all_results.append(result)
            
            # Dynamic printing
            status = "✓" if result['success'] else "✗"
            opt_status = "✓" if result['optimal_found'] else "✗"
            print(f"{status} {result['id']}: {result['solutions_count']} solutions, "
                  f"optimal: {opt_status}, bt_loops: {result['backtrack_loops']}, "
                  f"dive_loops: {result['dive_loops']}, time: {result['solve_time']:.4f}s")
        print()

    # Results table
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"{'ID':<15} {'Status':<8} {'Time(s)':<10} {'Solutions':<10} {'BT_Loops':<12} {'Dive_Loops':<12}")
    print("-" * 80)

    for result in all_results:
        status = "SUCCESS" if result['success'] else "FAILED"
        print(f"{result['id']:<15} {status:<8} {result['solve_time']:<10.4f} {result['solutions_count']:<10} {result['backtrack_loops']:<12} {result['dive_loops']:<12}")

    print("\n" + "=" * 80)