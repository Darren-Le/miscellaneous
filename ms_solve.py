import numpy as np
from fpylll import IntegerMatrix, LLL, BKZ
from math import gcd
from functools import reduce
import logging
import time
from datetime import datetime
import argparse
import highspy
from ms_data import MSData

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
        
        self.build_original_lp()

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

    def build_original_lp(self, save_path=None):
        """
        构建并保存原始线性规划问题
        
        原始问题形式:
        min x[0] + x[1] + ... + x[n-1]
        subject to:
            Ax = d
            0 <= x[i] <= r[i] for i = 0, ..., n-1
            x ∈ Z (整数约束)
        
        Args:
            save_path: LP文件保存路径，如果为None则使用默认路径
        """
        if not HIGHSPY_AVAILABLE:
            print("highspy不可用，跳过LP文件生成")
            return None
            
        # 初始化HiGHS模型
        h = highspy.Highs()
        
        # 添加决策变量 x[0], x[1], ..., x[n-1]
        x_vars = []
        for i in range(self.n):
            # 添加变量：下界为0，上界为r[i]，整数类型
            var = h.addVariable(
                lb=0, 
                ub=int(self.r[i]), 
                name=f"x_{i}",
                vtype=highspy.HighsVarType.kInteger  # 设置为整数变量
            )
            x_vars.append(var)
        
        # 添加等式约束 Ax = d
        for i in range(self.m):
            # 构建约束表达式: A[i,0]*x[0] + A[i,1]*x[1] + ... + A[i,n-1]*x[n-1] = d[i]
            constraint_expr = sum(int(self.A[i, j]) * x_vars[j] for j in range(self.n))
            h.addConstr(constraint_expr == int(self.d[i]))
        
        # 设置目标函数: min x[0] + x[1] + ... + x[n-1]
        objective_expr = sum(x_vars)
        h.minimize(objective_expr)
        
        # 生成保存路径
        if save_path is None:
            # 使用当前时间和问题规模生成默认文件名
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"original_lp_{self.m}x{self.n}_{timestamp}.lp"
        
        # 写入LP文件
        try:
            h.writeModel(save_path)
            print(f"原始LP问题已保存到: {save_path}")
            
            # 打印问题信息
            print(f"问题规模: {self.m} 个约束, {self.n} 个变量")
            print(f"目标函数: minimize ∑x_i")
            print(f"约束条件: Ax = d")
            print(f"变量边界: 0 ≤ x_i ≤ r_i")
            print(f"变量类型: 整数")
            
        except Exception as e:
            print(f"保存LP文件时出错: {e}")
        
        return save_path

    def enumerate(self):
        start_time = time.time()  # Track start time
        sols = []
        c = (self.n + 1) * self.rmax ** 2  # 预计算常数

        # 第二个剪枝策略：预计算每个 u_i 的全局界
        sqrt_c = np.sqrt(c)
        u_bounds_l2 = self.b_bar_norms['l2'] * sqrt_c
        u_bounds_l1 = self.b_bar_norms['l1'] * self.rmax
        # 取最紧界
        u_global_bounds = np.minimum(u_bounds_l2, u_bounds_l1)

        def backtrack(idx, u_values, prev_w, prev_w_norm_sq):
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
                            # Track first solution time
                            if self.first_solution_time is None:
                                self.first_solution_time = time.time() - start_time
                                self.first_sol_bt_loops = self.backtrack_loops
                            
                            sols.append(x.copy())

                            # Check if we've found enough solutions
                            if self.max_sols > 0 and len(sols) >= self.max_sols:
                                return True  # Signal to stop search
                return False
            
            # 第一个剪枝条件：检查 ||w^(idx+1)||_2^2 是否已经超过界限
            # prev_w_norm_sq = np.dot(prev_w, prev_w)
            if prev_w_norm_sq > c + 1e-10:
                self.dive_loops += 1
                self.first_pruning_effect_count += 1 
                return False # 剪枝
            
            # 计算 sum_{i=idx+1}^{n_basis-1} u_i * mu_{i,idx}
            mu_sum = 0.0
            for j in range(idx + 1, self.n_basis):
                mu_sum += u_values[j] * self.mu[j, idx]
            
            # 根据第一个剪枝条件计算 u[idx] 的范围
            # (u_idx + mu_sum)^2 <= (c - ||w^(idx+1)||_2^2) / ||b_hat^(idx)||_2^2
            bound_sq = (c - prev_w_norm_sq) / self.b_hat_norms_sq[idx]
            bound = np.sqrt(max(0, bound_sq))
            
            # u_idx 的范围是: -bound - mu_sum <= u_idx <= bound - mu_sum
            u_min_pruning1 = int(np.floor(-bound - mu_sum))
            u_max_pruning1 = int(np.ceil(bound - mu_sum))
            
            # 第二个剪枝策略：应用全局界
            u_min_pruning2 = int(np.floor(-u_global_bounds[idx]))
            u_max_pruning2 = int(np.ceil(u_global_bounds[idx]))
            
            # 取两个界的交集
            u_min = max(u_min_pruning1, u_min_pruning2)
            u_max = min(u_max_pruning1, u_max_pruning2)
            
            # 检查第二个剪枝策略是否生效
            original_range = u_max_pruning1 - u_min_pruning1 + 1
            final_range = max(0, u_max - u_min + 1)

            if final_range < original_range:
                self.second_pruning_effect_count += 1
            
            # 第三个剪枝策略：使用定理2进行剪枝
            for u_val in range(u_min, u_max + 1):
                u_values[idx] = u_val
                
                # 计算 w^(idx) = (u_val + mu_sum) * b_hat[idx] + prev_w
                coeff = u_val + mu_sum
                curr_w = coeff * self.b_hat[idx] + prev_w
                
                # 第三个剪枝条件：检查 ||w||_2^2 <= rmax * ||w||_1
                # ||w||_2^2 = coeff^2 * ||b_hat[idx]||_2^2 + ||prev_w||_2^2  
                w_norm_sq = coeff * coeff * self.b_hat_norms_sq[idx] + prev_w_norm_sq
                w_norm_l1 = np.linalg.norm(curr_w, ord=1)
                
                if w_norm_sq > self.rmax * w_norm_l1 + 1e-10:
                    # 如果当前w不满足条件，跳过所有满足 coeff * r > 0 的 u + r
                    if coeff > 0:
                        # 跳过所有 u_val + r (r > 0)，即跳过剩余的循环
                        self.third_pruning_effect_count += (u_max - u_val)
                        break
                    # 如果 coeff <= 0，只跳过当前值
                    self.third_pruning_effect_count += 1
                    continue
                
                if backtrack(idx - 1, u_values, curr_w, w_norm_sq):
                    return True
            
            return False
        
        # 开始回溯
        u_values = np.zeros(self.n_basis, dtype=int)
        initial_w = np.zeros(self.n + 1)
        backtrack(self.n_basis - 1, u_values, initial_w, 0.0)
        
        return sols

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

# 主函数部分 - 使用数据文件
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Market Split Solver')
    parser.add_argument('--data_path', type=str, default="../ms_instance/01-marketsplit/instances", help='Path to instance data')
    parser.add_argument('--sol_path', type=str, default="../ms_instance/01-marketsplit/solutions", help='Path to solution data')
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
    log_filename = f"res_{m_choices}_{time_str}.log"


    with open(log_filename, 'w') as f:
        print_and_log("=" * 136, f)
        print_and_log("RESULTS", f)
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