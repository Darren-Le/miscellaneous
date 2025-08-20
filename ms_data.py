import os
import numpy as np
from pathlib import Path

class MSData:
    def __init__(self, path):
        self.path = Path(path)
        self.data = []
        self.by_size = {}
        self.by_id = {}
        self._load()
    
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
    
    def get(self, id):
        return self.by_id.get(id)
    
    def size(self, m, n):
        return self.by_size.get((m, n), [])
    
    def sizes(self):
        return list(self.by_size.keys())
    
    def stats(self):
        if not self.data:
            return {}
        sizes = list(self.by_size.keys())
        ms = [m for m, n in sizes]
        ns = [n for m, n in sizes]
        
        print(f"Total instances: {len(self.data)}")
        print(f"M range: {min(ms)}-{max(ms)}")
        print(f"N range: {min(ns)}-{max(ns)}")
        print("Instances per size:")
        for size in sorted(sizes):
            count = len(self.by_size[size])
            print(f"  {size[0]:2d} x {size[1]:3d}: {count}")
        
        return {
            'total': len(self.data),
            'm_range': (min(ms), max(ms)),
            'n_range': (min(ns), max(ns)),
            'per_size': {size: len(self.by_size[size]) for size in sizes}
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]

# Usage
if __name__ == "__main__":
    ms = MSData(r"path/to/data")
    print(f"Loaded {len(ms)} instances")
    ms.stats()
    
    # Get by ID
    inst = ms.get("ms_15_200_003")
    if inst:
        print(f"Instance {inst['id']}: {inst['m']}x{inst['n']}")
    
    # Get by size
    instances = ms.size(3, 20)
    print(f"Found {len(instances)} instances of size 3x20")
    if instances:
        print("Instance IDs:", [inst['id'] for inst in instances])
    
    # Access matrix and vector
    if inst:
        A, d = inst['A'], inst['d']
        print(f"Matrix shape: {A.shape}, Vector shape: {d.shape}")