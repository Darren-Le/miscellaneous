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

        # Run preprocessing
        self._get_extended_matrix()
        self._get_reduced_basis()
        self._get_gso()
        self._compute_dual_norms()
    
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
    

