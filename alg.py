import numpy as np
from fpylll import IntegerMatrix, LLL, GSO
from math import gcd
from functools import reduce

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
            b_hat_i = self.basis[:, i].astype(float)
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
        
        def backtrack(idx, u_values, prev_w):
            """
            回溯算法
            idx: 当前要确定的 u 的索引 (从 n_basis-1 递减到 0)
            u_values: 长度为 n_basis 的数组，存储 u_0 到 u_{n_basis-1} 的值
            prev_w: w^(idx+1) - 上一层的投影向量
            """
            if idx == -1:
                # 所有 u_i 都已确定，构建解向量 v = sum(u_i * basis_i)
                v = np.zeros(self.n + 1)
                for i in range(self.n_basis):
                    v += u_values[i] * self.basis[i]
                
                # 检查是否满足约束: v_0 = rmax 且 -rmax <= v_i <= rmax for all i
                if abs(v[0] - self.rmax) < 1e-10:  # v_0 应该等于 rmax
                    if np.all(v >= -self.rmax - 1e-10) and np.all(v <= self.rmax + 1e-10):
                        # 转为整数解
                        v_int = np.round(v).astype(int)
                        # 从 v 恢复原始解 x
                        x = v_int[1:] // (2 * self.c) + self.r // 2
                        # 验证解
                        if np.allclose(self.A @ x, self.d):
                            sols.append(x.copy())
                return
            
            # 枚举 u[idx] 的可能值
            # TODO: 这里应该用剪枝条件计算精确范围
            u_range = 100  # 临时范围
            
            for u_val in range(-u_range, u_range + 1):
                u_values[idx] = u_val
                
                # 计算 w^(idx) 使用递推公式 (12)
                # w^(idx) = (sum_{i=idx}^{n_basis-1} u_i * mu_{i,idx}) * b_hat[idx] + w^(idx+1)
                coeff = u_val * self.mu[idx, idx]  # mu[idx,idx] = 1
                for j in range(idx + 1, self.n_basis):
                    coeff += u_values[j] * self.mu[j, idx]
                
                curr_w = coeff * self.b_hat[idx] + prev_w
                
                # 递归到下一层
                backtrack(idx - 1, u_values, curr_w)
        
        # 初始化
        u_initial = np.zeros(self.n_basis, dtype=int)
        w_initial = np.zeros(self.n + 1)
        
        # 从最高索引开始回溯 (对应论文中从 u_{n-m+1} 开始)
        backtrack(self.n_basis - 1, u_initial, w_initial)
        
        return sols

A = np.array([[1,4]], dtype=int)
d = np.array([4], dtype=int)
ms = MarketSplit(A, d)
print(ms.L)
print(ms.basis)
print(ms.coords)
