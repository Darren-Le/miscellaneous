import numpy as np
from pathlib import Path

class MSData:
    def __init__(self, path, sol_path=None):
        self.path = Path(path)
        self.sol_path = Path(sol_path) if sol_path else self.path
        self.data = []
        self.by_size = {}
        self.by_id = {}
        self.solutions = {}
        self._load()
        self._load_solutions()
    
    def _load(self):
        for f in self.path.glob('*.dat'):
            try:
                with open(f, 'r') as file:
                    lines = [l.strip() for l in file if l.strip() and not l.startswith('#')]
                
                m, n = map(int, lines[0].split())
                A = np.zeros((m, n), dtype=int)
                d = np.zeros(m, dtype=int)
                
                for i in range(m):
                    vals = list(map(int, lines[i + 1].split()))
                    A[i, :] = vals[:n]
                    d[i] = vals[n]
                
                id = f.stem
                inst = {'id': id, 'm': m, 'n': n, 'A': A, 'd': d}
                
                self.data.append(inst)
                self.by_id[id] = inst
                
                size = (m, n)
                if size not in self.by_size:
                    self.by_size[size] = []
                self.by_size[size].append(inst)
            except:
                pass
    
    def _load_solutions(self):
        for f in self.sol_path.glob('*.opt.sol'):
            try:
                with open(f, 'r') as file:
                    lines = [l.strip() for l in file if l.strip() and not l.startswith('#')]
                
                x_vars = {}
                for line in lines:
                    if line.startswith('x#'):
                        parts = line.split()
                        var_num = int(parts[0][2:])
                        var_val = int(parts[1])
                        x_vars[var_num] = var_val
                
                if x_vars:
                    n = max(x_vars.keys())
                    x = np.zeros(n, dtype=int)
                    for i in range(1, n + 1):
                        x[i - 1] = x_vars.get(i, 0)
                    
                    id = f.stem.replace('.opt', '')
                    self.solutions[id] = x
            except:
                pass
    
    def get(self, m=None, n=None, id=None):
        if id is not None:
            return self.by_id.get(id)
        
        results = self.data
        if m is not None:
            results = [inst for inst in results if inst['m'] == m]
        if n is not None:
            results = [inst for inst in results if inst['n'] == n]
        
        return results
    
    def get_solution(self, id):
        return self.solutions.get(id)
    
    def size(self, m, n):
        return self.by_size.get((m, n), [])
    
    def sizes(self):
        return list(self.by_size.keys())
    
    def stats(self):
        if not self.data:
            return {}
        sizes = list(self.by_size.keys())
        ms_data = [m for m, n in sizes]
        ns = [n for m, n in sizes]
        
        print(f"Total instances: {len(self.data)}")
        print(f"M range: {min(ms_data)}-{max(ms_data)}")
        print(f"N range: {min(ns)}-{max(ns)}")
        print(f"Solutions loaded: {len(self.solutions)}")
        print("Instances per size:")
        for size in sorted(sizes):
            count = len(self.by_size[size])
            print(f"  {size[0]:2d} x {size[1]:3d}: {count}")
        
        return {
            'total': len(self.data),
            'm_range': (min(ms_data), max(ms_data)),
            'n_range': (min(ns), max(ns)),
            'per_size': {size: len(self.by_size[size]) for size in sizes}
        }
    
    def column_filter(self, instance_id):
    """Filter A matrix columns based on optimal solution (keep only columns where opt_sol == 1)"""
        inst = self.get(id=instance_id)
        if inst is None:
            return None
            
        opt_sol = self.get_solution(instance_id)
        if opt_sol is None:
            return None
        
        # Keep only columns where optimal solution is 1
        mask = opt_sol == 1
        filtered_A = inst['A'][:, mask]
        
        return filtered_A
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]

# Usage
if __name__ == "__main__":
    data_path = "path/to/data"
    sol_path = "path/to/sols"
    ms_data = MSData(sol_path, data_path)
    print(f"Loaded {len(ms_data)} instances")
    ms_data.stats()
    
    # Get by ID
    inst = ms_data.get(id="ms_03_050_009")
    if inst:
        print(f"Instance {inst['id']}: {inst['m']}x{inst['n']}")
    
    # Get solution
    sol = ms_data.get_solution("ms_03_050_009")
    if sol is not None:
        print(f"Solution: {sol}")
        print(f"Solution length: {len(sol)}")
    
    # Get by m
    m3_instances = ms_data.get(m=3)
    print(f"Found {len(m3_instances)} instances with m=3")
    
    # Get by n
    n20_instances = ms_data.get(n=20)
    print(f"Found {len(n20_instances)} instances with n=20")
    
    # Get by size
    m3_n20_instances = ms_data.get(m=3, n=20)
    print(f"Found {len(m3_n20_instances)} instances with m=3 and n=20")
    if m3_n20_instances:
        print("m = 3, n = 20 Instance IDs:", [inst['id'] for inst in m3_n20_instances])
    
    # Access matrix and vector
    if inst:
        A, d = inst['A'], inst['d']
        print(f"Matrix shape: {A.shape}, Vector shape: {d.shape}")

    # Test column filter
    filtered_A = ms_data.column_filter("ms_03_050_009")
    if filtered_A is not None:
        print(f"Original A shape: {inst['A'].shape}, Filtered A shape: {filtered_A.shape}")