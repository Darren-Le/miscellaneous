import numpy as np
from fpylll import IntegerMatrix, LLL, GSO
from math import gcd
from functools import reduce
from ms_data import MSData

class MarketSplit:
    def __init__(self, A, d, r=None):
        self.A = A
        self.d = d
        self.m, self.n = A.shape
        self.n_basis = self.n - self.m + 1
        self.r = r if r else np.ones(self.n, dtype=int)

        self.rmax = None
        self.c = None
        self.basis = None  # Store as row vector
        self.b_hat = None  # Store as row vector
        self.b_bar = None  # Store as row vector
        self.mu = None

        # coords[i] is the coordinates of basis[i]: self.L @ coords[i].T = basis[i].T
        # it is a solution of the homogenous system: (A, -d) @ coords[i].T = 0
        self.coords = None 

        # Run preprocessing
        self._get_extended_matrix()
        self._get_reduced_basis()
        self._get_gso()
        self._compute_dual_norms()
        self._get_coordinates()
    
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
        self.basis = np.array(b)

    
    def _get_gso(self):
        assert(self.basis.shape[0] == self.n_basis)
        assert(self.basis.shape[1] == self.n + 1)
        
        B_T = IntegerMatrix.from_matrix(self.basis.T.tolist())
        M = GSO.Mat(B_T)
        M.update_gso()
        
        # Extract mu coefficients
        self.mu = np.zeros((self.n_basis, self.n_basis))
        for i in range(self.n_basis):
            self.mu[i, i] = 1.0
            for j in range(i):
                self.mu[i, j] = M.get_mu(i, j)
        
        # Extract squared norms
        self.b_hat_norms_sq = np.zeros(self.n_basis)
        for i in range(self.n_basis):
            self.b_hat_norms_sq[i] = M.get_r(i, i)
        
        # Compute orthogonal vectors
        self.b_hat = []
        for i in range(self.n_basis):
            b_hat_i = self.basis[i].astype(float)
            for j in range(i):
                b_hat_i = b_hat_i - self.mu[i, j] * self.b_hat[j]
            self.b_hat.append(b_hat_i)
        
        self.b_hat = np.array(self.b_hat)

    def _compute_dual_norms(self):
        B = self.basis.T
        B_T = self.basis
        gram = B_T @ B
        gram_inv = np.linalg.inv(gram)
        self.b_bar = (B @ gram_inv).T
        
        self.b_bar_norms = {
            'l2': np.array([np.linalg.norm(self.b_bar[i, :], ord=2) for i in range(self.n_basis)]),
            'l1': np.array([np.linalg.norm(self.b_bar[i, :], ord=1) for i in range(self.n_basis)])
        }
    
    def _get_coordinates(self):
        L_bottom = self.L[self.m:, :]
        coordinates = []
        for i in range(self.n_basis):
            x = np.linalg.solve(L_bottom, self.basis[i, :])
            coordinates.append(x)
        self.coords = np.array(coordinates)

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
            if idx == -1:
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
            
            backtrack(idx - 1, u_values, curr_w)
        
        return sols

# 主函数部分 - 使用数据文件
if __name__ == "__main__":
    # 1. 加载数据
    data_path = "path/to/data"  # 替换为你的数据文件夹路径
    ms_data = MSData(data_path)
    
    # 2. 获取特定实例
    instance_id = "ms_03_050_002"
    inst = ms_data.get(instance_id)
    
    # 3. 提取A和d矩阵
    A = inst['A']
    d = inst['d']
    
    print(f"Loaded instance {instance_id}:")
    print(f"  Matrix A shape: {A.shape}")
    print(f"  Vector d shape: {d.shape}")
    print(f"  A = \n{A}")
    print(f"  d = {d}")
    
    # 4. 创建MarketSplit实例并求解
    try:
        ms = MarketSplit(A, d)
        print(f"\nExtended matrix L shape: {ms.L.shape}")
        print(f"Basis shape: {ms.basis.shape}")
        print(f"Coordinates shape: {ms.coords.shape}")
        
        # 可选：运行枚举算法
        # print("\nRunning enumeration...")
        # solutions = ms.enumerate()
        # print(f"Found {len(solutions)} solutions")
        
    except Exception as e:
        print(f"Error creating MarketSplit instance: {e}")